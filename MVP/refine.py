import torch
import torch.nn.functional as F
from gsplat.strategy.default_sh_opacity import SHOpacityStrategy
from gsplat import rasterization
import torch.optim as optim

def refine_gaussians(gaussians, target_data, config, iterations=2000):
    _, _, _, h, w = target_data["image"].size()
    
    # Extract Parameter
    with torch.no_grad():
        init_xyz = gaussians["xyz"][0].detach().clone()
        init_feature = gaussians["feature"][0].detach().clone()
        init_scale = gaussians["scale"][0].detach().clone()
        init_rotation = gaussians["rotation"][0].detach().clone()
        init_opacity = gaussians["opacity"][0].detach().clone()

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
        k: optim.Adam([v], lr=1e-3) for k, v in params.items() if v.requires_grad
    }

    # Init strategy
    strategy = SHOpacityStrategy()
    strategy.check_sanity(params, optimizers)
    strategy_state = strategy.initialize_state()
    
    target_c2w = target_data["c2w"][0]
    test_w2c = torch.inverse(target_c2w).float()
    
    target_intr = target_data["fxfycxcy"][0]
    V_target = target_intr.shape[0]
    test_intr_i = torch.zeros((V_target, 3, 3), device=test_w2c.device, dtype=test_w2c.dtype)
    test_intr_i[:, 0, 0] = target_intr[:, 0]
    test_intr_i[:, 1, 1] = target_intr[:, 1]
    test_intr_i[:, 0, 2] = target_intr[:, 2]
    test_intr_i[:, 1, 2] = target_intr[:, 3]
    test_intr_i[:, 2, 2] = 1.0
    
    target_images = target_data["image"][0].permute(0, 2, 3, 1).contiguous()

    for step in range(iterations):
        active_scales = torch.exp(params["scales"])
        active_quats = F.normalize(params["quats"], p=2, dim=-1)
                
        render_image, _, info  = rasterization(
            params["means"], active_quats, active_scales,
            params["opacities"], params["features"],
            test_w2c, test_intr_i,
            w, h,
            sh_degree=config.model.gaussians.sh_degree,
            near_plane=config.model.gaussians.near_plane,
            far_plane=config.model.gaussians.far_plane,
            sh_degree_opacity=config.model.gaussians.opacity_degree,
            packed=False,
            absgrad=False,
            sparse_grad=False,                                        
            render_mode="RGB",
            backgrounds=torch.ones(1, 3).to(test_intr_i.device),
            rasterize_mode='classic'
        )
        
        strategy.step_pre_backward(params, optimizers, strategy_state, step, info)

        loss = F.l1_loss(render_image, target_images)
        loss.backward()

        strategy.step_post_backward(params, optimizers, strategy_state, step, info)
        
        for opt in optimizers.values():
            opt.step()
            opt.zero_grad()

    return render_image, params