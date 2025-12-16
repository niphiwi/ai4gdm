import torch
import torch.nn.functional as F

class NoiseScheduler:
    @torch.no_grad()
    def __init__(self, T=100, beta_start=0.0001, beta_end=0.01, device="cpu"):
        self.T = T
        self.device = device

        # Linear beta schedule
        self.betas = torch.linspace(beta_start, beta_end, T).to(device)

        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0).to(device)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0).to(device)

        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas).to(device)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod).to(device)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod).to(device)

        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

    def q_sample(self, x0, t, noise=None):
        """
        Generate x_t from x_0 using the diffusion formula.
        """
        if noise is None:
            noise = torch.randn_like(x0)

        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x0.shape).to(x0.device)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape).to(x0.device)

        return (
            sqrt_alphas_cumprod_t * x0 +
            sqrt_one_minus_alphas_cumprod_t * noise,
            noise
        )

    def _extract(self, vals, t, shape):
        """
        Get index t from vals for a batch, reshape to match `shape`.
        """
        # t.to(vals.device)
        out = vals.gather(0, t).reshape(-1, *[1] * (len(shape) - 1))
        return out.to(self.device)