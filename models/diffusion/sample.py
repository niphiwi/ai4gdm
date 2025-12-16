import torch

@torch.no_grad()
def sample_timestep(x, cond_mask, cond_vals, t, model, scheduler):
    betas_t = scheduler._extract(scheduler.betas, t, x.shape)
    sqrt_one_minus_cumprod_t = scheduler._extract(scheduler.sqrt_one_minus_alphas_cumprod, t, x.shape)
    sqrt_recip_alpha_t = scheduler._extract(scheduler.sqrt_recip_alphas, t, x.shape)

    # Prepare model input
    model_input = torch.cat([x, cond_mask, cond_vals], dim=1)
    pred_noise = model(model_input, t)

    # DDPM mean
    model_mean = sqrt_recip_alpha_t * (x - betas_t * pred_noise / sqrt_one_minus_cumprod_t)

    if t[0].item() == 0:
        return model_mean
    else:
        noise = torch.randn_like(x)
        posterior_var_t = scheduler._extract(scheduler.posterior_variance, t, x.shape)
        return model_mean + torch.sqrt(posterior_var_t) * noise

def create_random_mask(image_size=64, num_known=50, device='cuda'):
    """
    Create a random mask of known pixels of shape [1, 1, image_size, image_size].
    """
    mask = torch.zeros((1, 1, image_size, image_size), dtype=torch.float32, device=device)
    known_indices = torch.randperm(image_size * image_size)[:num_known]
    ys = known_indices // image_size
    xs = known_indices % image_size
    mask[0, 0, ys, xs] = 1.0  # Set known pixels to 1
    return mask

@torch.no_grad()                                                                                                                                                                                                                                                                                                                                                                                                                                        
def sample_masked_input(x0, scheduler, num_known):
    device = x0.device
    B, C, H, W = x0.shape
    x0 = x0.to(device)  # [B, 1, 64, 64]

    # Sample diffusion timestep
    t = torch.randint(0, scheduler.T, (B,), device=device)

    mask = create_random_mask(image_size=H, num_known=num_known)
    cond_vals = mask * x0
    cond_mask = mask.expand(B, 1, H, W)

    # Forward diffusion (sample x_t and keep noise for training)
    x_t, noise = scheduler.q_sample(x0, t)
    model_input = torch.cat([x_t, cond_mask, cond_vals], dim=1)  # [B, 3, 64, 64]


    return model_input, noise, t