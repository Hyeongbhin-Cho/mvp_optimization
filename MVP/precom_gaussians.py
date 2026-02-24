import importlib
import os
import torch
from torch.utils.data import DataLoader
from setup import init_config
from metric_utils import export_results, summarize_evaluation

import torch.nn.functional as F
from gsplat import rasterization

@torch.no_grad()
def prune_gaussians(gaussians, threshold=0.001):
    raw_opacity = gaussians["opacity"][0]
    
    base_opacity = torch.sigmoid(raw_opacity.abs().sum(dim=(1, 2)))
    mask = base_opacity > threshold
    
    before_cnt = raw_opacity.shape[0]
    after_cnt = mask.sum().item()
    print(f"Pruning: {before_cnt} -> {after_cnt} ({(after_cnt/before_cnt)*100:.1f}%)")
    
    pruned_gaussians = {k: v[0][mask].unsqueeze(0).detach().cpu().half() for k, v in gaussians.items()}
    
    return pruned_gaussians

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
            _, _, _, h, w = target_data_dict["image"].size()
            with torch.no_grad():
                result = model(input_data_dict, target_data_dict)
                gaussians = result.gaussians
                pruned_gaussians = prune_gaussians(gaussians, config.inference.prune_threshold)

                target_c2w = target_data_dict["c2w"][0]
                test_w2c = torch.inverse(target_c2w).float()
                
                target_intr = target_data_dict["fxfycxcy"][0]
                V_target = target_intr.shape[0]
                test_intr_i = torch.zeros((V_target, 3, 3), device=test_w2c.device, dtype=test_w2c.dtype)
                test_intr_i[:, 0, 0] = target_intr[:, 0]
                test_intr_i[:, 1, 1] = target_intr[:, 1]
                test_intr_i[:, 0, 2] = target_intr[:, 2]
                test_intr_i[:, 1, 2] = target_intr[:, 3]
                test_intr_i[:, 2, 2] = 1.0
                
                means_cuda = pruned_gaussians["xyz"][0].to(device).float()
                quats_cuda = pruned_gaussians["rotation"][0].to(device).float()
                scales_cuda = pruned_gaussians["scale"][0].to(device).float()
                opacities_cuda = pruned_gaussians["opacitie"][0].to(device).float()
                features_cuda = pruned_gaussians["feature"][0].to(device).float()
                
                active_scales = torch.exp(scales_cuda)
                active_quats = F.normalize(quats_cuda, p=2, dim=-1)
                
                render_images=[]
                for i in range(V_target):
                    render, _, _= rasterization(
                        means_cuda, active_quats, active_scales,
                        opacities_cuda, features_cuda,
                        test_w2c[i],
                        test_intr_i[i],
                        w, h,
                        sh_degree=config.model.gaussians.sh_degree,
                        near_plane=config.model.gaussians.near_plane,
                        far_plane=config.model.gaussians.far_plane,
                        sh_degree_opacity=config.model.gaussians.opacity_degree,
                        packed=False,
                        absgrad=False,
                        sparse_grad=False,                                        
                        render_mode="RGB",
                        backgrounds=torch.ones(V_target, 3).to(test_intr_i.device),
                        rasterize_mode='classic'
                    )
                    render_images.append(render)

                render_images = torch.concat(render_images, dim=0).unsqueeze(0).permute(0, 1, 4, 2, 3)
                
            save_path = os.path.join("/home/gudqls22/data/gaussians_eval", f"gaussians_{cnt:04d}.pt")
            torch.save(pruned_gaussians, save_path)
            
            result.render = render_images
        
            export_results(result, config.inference.out_dir, 
                        compute_metrics=config.inference.get("compute_metrics"), 
                        uid=cnt)
            
        torch.cuda.empty_cache()


    if config.inference.get("compute_metrics", False):
        summarize_evaluation(config.inference.out_dir)
    exit(0)