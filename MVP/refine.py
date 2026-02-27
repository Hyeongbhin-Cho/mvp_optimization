import torch
import importlib
import os
from torch.utils.data import DataLoader
from setup import init_config
from easydict import EasyDict as edict
from metric_utils import export_results, summarize_evaluation

import torch.nn.functional as F
from gsplat.strategy.default_sh_opacity import SHOpacityStrategy
from gsplat import rasterization
import torch.optim as optim
import time
from metric_utils import (compute_psnr, compute_lpips, compute_ssim)
import json

@torch.no_grad()
def render_images(gaussians, target_data, config, chunk_size=4):
    _, _, _, h, w = target_data["image"].size()
    
    target_c2w = target_data["c2w"][0]
    render_w2c = torch.inverse(target_c2w).float()
    
    target_intr = target_data["fxfycxcy"][0]
    V = target_intr.shape[0]
    render_intr_i = torch.zeros((V, 3, 3), device=render_w2c.device, dtype=render_w2c.dtype)
    render_intr_i[:, 0, 0] = target_intr[:, 0]
    render_intr_i[:, 1, 1] = target_intr[:, 1]
    render_intr_i[:, 0, 2] = target_intr[:, 2]
    render_intr_i[:, 1, 2] = target_intr[:, 3]
    render_intr_i[:, 2, 2] = 1.0
    
    with torch.no_grad():
        render_scales = torch.exp(gaussians["scales"])
        render_quats = F.normalize(gaussians["quats"], p=2, dim=-1)
        
        final_render_images = []
        
        for i in range(0, V, chunk_size):
            end_i = min(i+chunk_size, V)
            
            render_images, _, _ = rasterization(
                gaussians["means"], render_quats, render_scales,
                gaussians["opacities"], gaussians["features"],
                render_w2c[i: end_i],       
                render_intr_i[i: end_i],    
                w, h,
                sh_degree=config.model.gaussians.sh_degree,
                near_plane=config.model.gaussians.near_plane,
                far_plane=config.model.gaussians.far_plane,
                sh_degree_opacity=config.model.gaussians.opacity_degree,
                packed=False,
                backgrounds=torch.ones((end_i - i, 3), device=render_w2c.device), 
                rasterize_mode='classic'
            )
            
            final_render_images.append(render_images)
        
    return torch.concat(final_render_images, dim=0).unsqueeze(0).permute(0, 1, 4, 2, 3)

def refine_gaussians(gaussians, input_data, target_data, config, iterations=2000, compute_metrics=False, metrics_every=100, chunk_size=4):
    _, _, _, h, w = target_data["image"].size()
    
    # Extract Parameter
    with torch.no_grad():
        init_xyz = gaussians["xyz"][0].detach().clone().float().requires_grad_(True)
        init_feature = gaussians["feature"][0].detach().clone().float().requires_grad_(True)
        init_scale = gaussians["scale"][0].detach().clone().float().requires_grad_(True)
        init_rotation = gaussians["rotation"][0].detach().clone().float().requires_grad_(True)
        init_opacity = gaussians["opacity"][0].detach().clone().float().requires_grad_(True)

    # Set ParameterDict
    params = torch.nn.ParameterDict({
        "means": torch.nn.Parameter(init_xyz),
        "features": torch.nn.Parameter(init_feature),
        "scales": torch.nn.Parameter(init_scale),
        "quats": torch.nn.Parameter(init_rotation),
        "opacities": torch.nn.Parameter(init_opacity)
    })

    # Set Optmization
    optimizers = {
        "means": optim.Adam([params["means"]], lr=1.6e-5), 
        "features": optim.Adam([params["features"]], lr=2.5e-3),
        "opacities": optim.Adam([params["opacities"]], lr=1e-3), 
        "scales": optim.Adam([params["scales"]], lr=1e-3),
        "quats": optim.Adam([params["quats"]], lr=1e-4),
    }

    # Init strategy
    strategy = SHOpacityStrategy(reset_every=config.inference.reset_every)
    strategy.check_sanity(params, optimizers)
    strategy_state = strategy.initialize_state()
    
    input_c2w = input_data["c2w"][0]
    test_w2c = torch.inverse(input_c2w).float()
    
    input_intr = input_data["fxfycxcy"][0]
    V_input = input_intr.shape[0]
    test_intr_i = torch.zeros((V_input, 3, 3), device=test_w2c.device, dtype=test_w2c.dtype)
    test_intr_i[:, 0, 0] = input_intr[:, 0]
    test_intr_i[:, 1, 1] = input_intr[:, 1]
    test_intr_i[:, 0, 2] = input_intr[:, 2]
    test_intr_i[:, 1, 2] = input_intr[:, 3]
    test_intr_i[:, 2, 2] = 1.0
    
    input_images = input_data["image"][0].permute(0, 2, 3, 1).contiguous()
    target_images = target_data["image"].contiguous()
    
    with torch.enable_grad():
        for step in range(iterations):
            indices = torch.randperm(V_input, device=test_w2c.device)[:chunk_size]
            
            active_scales = torch.exp(params["scales"])
            active_quats = F.normalize(params["quats"], p=2, dim=-1)
                    
            input_render_images, _, info  = rasterization(
                params["means"], active_quats, active_scales,
                params["opacities"], params["features"],
                test_w2c[indices],
                test_intr_i[indices],
                w, h,
                sh_degree=config.model.gaussians.sh_degree,
                near_plane=config.model.gaussians.near_plane,
                far_plane=config.model.gaussians.far_plane,
                sh_degree_opacity=config.model.gaussians.opacity_degree,
                packed=False,
                absgrad=False,
                sparse_grad=False,                                        
                render_mode="RGB",
                backgrounds=torch.ones(chunk_size, 3).to(test_intr_i.device),
                rasterize_mode='classic'
            )
            
            strategy.step_pre_backward(params, optimizers, strategy_state, step, info)

            loss = F.l1_loss(input_render_images, input_images[indices])
            loss.backward()

            strategy.step_post_backward(params, optimizers, strategy_state, step, info)
            
            for opt in optimizers.values():
                opt.step()
                opt.zero_grad()
            
            metrics = []
            if (step % metrics_every == 0) & compute_metrics:
                target_render_images = render_images(params, target_data, config, chunk_size)
                
                psnr = float(compute_psnr(target_images[indices], target_render_images).mean())
                lpips = float(compute_lpips(target_images[indices], target_render_images).mean())
                ssim = float(compute_ssim(target_images[indices], target_render_images).mean())
                
                metrics.append({
                    "step": step,
                    "psnr": psnr,
                    "lpips": lpips,
                    "ssim": ssim
                })
                
    with torch.no_grad():
        target_render_images = render_images(params, target_data, config, chunk_size)
        
        result = edict(
                input=input_data,
                target=target_data,
                render=target_render_images,
                gaussians=gaussians,
                metrics=metrics
                )

    return result

if __name__ == "__main__":
    config = init_config()

    os.environ["OMP_NUM_THREADS"] = str(config.inference.get("num_threads", 1))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set up tf32
    torch.backends.cuda.matmul.allow_tf32 = config.inference.use_tf32
    torch.backends.cudnn.allow_tf32 = config.inference.use_tf32
    amp_dtype_mapping = {
        "fp16": torch.float16, 
        "bf16": torch.bfloat16, 
        "fp32": torch.float32, 
        'tf32': torch.float32
    }

    # Load data
    dataset_name = config.inference.get("dataset_name", "data.dataset.Dataset")
    module, class_name = dataset_name.rsplit(".", 1)
    Dataset = importlib.import_module(module).__dict__[class_name]
    dataset = Dataset(config)

    dataloader = DataLoader(
        dataset,
        batch_size=config.inference.batch_size_per_gpu,
        shuffle=False,
        num_workers=config.inference.num_workers,
        prefetch_factor=config.inference.prefetch_factor,
        persistent_workers=True,
        pin_memory=False,
    )
    dataloader_iter = iter(dataloader)


    # Import model and load checkpoint
    module, class_name = config.model.class_name.rsplit(".", 1)
    MVP = importlib.import_module(module).__dict__[class_name]
    model = MVP(config).to(device)
    msg = model.load_ckpt(config.inference.ckpt_path)
    print(msg)

    print(f"Running inference; save results to: {config.inference.out_dir}")
    import warnings
    warnings.filterwarnings('ignore', category=FutureWarning)

    model.eval()
    cnt = 0
    with torch.no_grad(), torch.autocast(
        enabled=config.inference.use_amp,
        device_type="cuda",
        dtype=amp_dtype_mapping[config.inference.amp_dtype],
    ):
        for batch in dataloader:
            batch = {k: v.to(device) if type(v) == torch.Tensor else v for k, v in batch.items()}
            print(cnt)
            cnt += 1
            input_data_dict = {key: value[:, :config.data.num_input_frames] if type(value) == torch.Tensor else value for key, value in batch.items()}
            target_data_dict = {key: value[:, config.data.num_input_frames:] if type(value) == torch.Tensor else None for key, value in batch.items()}
            
            load_path = os.path.join(config.inference.precom_path, f"gaussians_{cnt:04d}.pt")
    
            if not os.path.exists(load_path):
                continue
            
            gaussians = torch.load(load_path, map_location="cuda")
            
            start_time = time.time()
            result = refine_gaussians(gaussians,
                                      input_data_dict,
                                      target_data_dict,
                                      config,
                                      iterations=config.inference.refine_iters,
                                      compute_metrics=config.inference.compute_metrics,
                                      metrics_every=config.inference.metrics_every,
                                      chunk_size=config.inference.chunk_size
            )
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"[{cnt:04d}] Scene Overfitting Time: {elapsed_time:.2f} seconds")
            
            metrics_dir = os.path.join(config.inference.out_dir, "metrics")
            os.makedirs(metrics_dir, exist_ok=True)
            
            json_path = os.path.join(metrics_dir, f"gaussian_{cnt:04d}.json")
            
            save_data = {
                "overfitting_time_seconds": elapsed_time,
                "metrics_history": result.metrics 
            }
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, indent=4)           
        
            export_results(result, config.inference.out_dir, 
                        compute_metrics=config.inference.get("compute_metrics"), 
                        uid=cnt)
            
        torch.cuda.empty_cache()


    if config.inference.get("compute_metrics", False):
        summarize_evaluation(config.inference.out_dir)
    exit(0)