import time
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime

from data.dataset import GDMDataset
from models.diffusion.model import SimpleUnet
from models.diffusion.noise_scheduler import NoiseScheduler
from models.diffusion.sample import sample_timestep
from neuralop.models import FNO

from models.sota.kernel_dmv.wrapper import KernelDMV
from models.sota.dares.wrapper import Dares

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SCHEDULER = NoiseScheduler(T=100, beta_start=1e-4, beta_end=0.2, device=DEVICE)

torch.manual_seed(1337)

def build_dataloader(data_path: str, batch_size: int = 1) -> DataLoader:
    dataset = GDMDataset(data_path, mode="test")
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def load_fno_model(weights_path: str) -> FNO:
    model = FNO(
        n_modes=(16, 16),
        in_channels=2,
        out_channels=1,
        hidden_channels=32,
        projection_channel_ratio=2,
    ).to(DEVICE)
    state = torch.load(weights_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(state)
    model.eval()
    return model

def load_diffusion_model(weights_path: str) -> SimpleUnet:
    model = SimpleUnet(dims=(32, 64, 64), in_dim=3).to(DEVICE)
    state = torch.load(weights_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(state)
    model.eval()
    return model

def load_kdm_model(kernel_size: int = 4):
    model = KernelDMV(x_range=[0,63], y_range=[0,63], cell_size=1, kernel_size=kernel_size)
    return model

def load_dares_model():
    return Dares()


def prepare_sota_inputs(X, batch_size: int):
    if batch_size != 1:
        raise ValueError("batch_size must be 1 for SOTA models evaluation.")

    mask = X[:, 0:1, :, :]  # [batch_size, 1, H, W]
    vals = X[:, 1:2, :, :]  # [batch_size, 1, H, W]

    mask_2d = mask[0][0]  # [H, W]
    y_coords, x_coords = torch.where(mask_2d == 1)
    positions = torch.stack([y_coords, x_coords], dim=1)
    measurements = vals[0][0][mask_2d.bool()]

    return positions.cpu(), measurements.cpu()

def sample_diffusion_model(X: torch.Tensor, diffusion_model: torch.nn.Module) -> torch.Tensor:
    batch_size, _, H, W = X.shape
    x_t = torch.randn((batch_size, 1, H, W), device=DEVICE)

    mask = X[:, 0:1, :, :]  # [batch_size, 1, H, W]
    vals = X[:, 1:2, :, :]  # [batch_size, 1, H, W]

    for t in reversed(range(SCHEDULER.T)):
        t_tensor = torch.full((1,), t, device=DEVICE, dtype=torch.long)
        with torch.no_grad():
            x_t = sample_timestep(x_t, mask, vals, t_tensor, diffusion_model, SCHEDULER)
    return x_t


def evaluate(dataloader: DataLoader, batch_size: int, eval_fno: bool, eval_diffusion: bool, eval_kdm: bool, eval_dares: bool, fno_model: torch.nn.Module = None, diffusion_model: torch.nn.Module = None, kdm_model = None, dares_model = None) -> None:
    times_fno, times_diffusion = [], []
    rmses_fno, rmses_diffusion = [], []
    times_kdm, rmses_kdm = [], []
    times_dares, rmses_dares = [], []

    with torch.no_grad():
        for X, y in tqdm(dataloader, mininterval=60):
            X = X.to(DEVICE)
            y = y.to(DEVICE)

            if eval_kdm or eval_dares:
                positions, measurements = prepare_sota_inputs(X, batch_size)

            # FNO
            if eval_fno:
                start = time.time()
                pred_fno = fno_model(X)
                times_fno.append(time.time() - start)
                rmses_fno.append(torch.sqrt(torch.mean((pred_fno - y) ** 2)).item())

            # Diffusion
            if eval_diffusion:
                start = time.time()
                pred_diffusion = sample_diffusion_model(X, diffusion_model)
                times_diffusion.append(time.time() - start)
                rmses_diffusion.append(torch.sqrt(torch.mean((pred_diffusion - y) ** 2)).item())

            # KDM+V
            if eval_kdm:
                if batch_size != 1:
                    raise ValueError("batch_size must be 1 for SOTA models evaluation.")
                start = time.time()
                kdm_model.set_measurements(positions, measurements)
                pred_kdm = kdm_model.predict()
                times_kdm.append(time.time() - start)
                rmses_kdm.append(torch.sqrt(torch.mean((pred_kdm.to(DEVICE) - y[0].to(DEVICE))**2)).item())

            # DARES
            if eval_dares:
                if batch_size != 1:
                    raise ValueError("batch_size must be 1 for SOTA models evaluation.")
                start = time.time()
                dares_model.set_measurements(positions, measurements)
                pred_dares = dares_model.predict()
                times_dares.append(time.time() - start)
                rmses_dares.append(torch.sqrt(torch.mean((pred_dares.to(DEVICE) - y[0].to(DEVICE))**2)).item())

    if eval_fno:
        print(f"FNO       - Avg Time: {sum(times_fno)/len(times_fno):.4f}s, Avg RMSE: {sum(rmses_fno)/len(rmses_fno):.4f}")
    if eval_diffusion:
        print(f"Diffusion - Avg Time: {sum(times_diffusion)/len(times_diffusion):.4f}s, Avg RMSE: {sum(rmses_diffusion)/len(rmses_diffusion):.4f}")
    if eval_kdm:
        print(f"KDM+V     - Avg Time: {sum(times_kdm)/len(times_kdm):.4f}s, Avg RMSE: {sum(rmses_kdm)/len(rmses_kdm):.4f}")
    if eval_dares:
        print(f"DARES     - Avg Time: {sum(times_dares)/len(times_dares):.4f}s, Avg RMSE: {sum(rmses_dares)/len(rmses_dares):.4f}")

    # write results to file
    with open("evaluation_results.txt", "a") as f:
        # add header containing date
        f.write(f"Evaluation Results - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        if eval_fno:
            f.write(f"FNO       - Avg Time: {sum(times_fno)/len(times_fno):.4f}s, Avg RMSE: {sum(rmses_fno)/len(rmses_fno):.4f}\n")
        if eval_diffusion:
            f.write(f"Diffusion - Avg Time: {sum(times_diffusion)/len(times_diffusion):.4f}s, Avg RMSE: {sum(rmses_diffusion)/len(rmses_diffusion):.4f}\n")
        if eval_kdm:
            f.write(f"KDM+V     - Avg Time: {sum(times_kdm)/len(times_kdm):.4f}s, Avg RMSE: {sum(rmses_kdm)/len(rmses_kdm):.4f}\n")
        if eval_dares:
            f.write(f"DARES     - Avg Time: {sum(times_dares)/len(times_dares):.4f}s, Avg RMSE: {sum(rmses_dares)/len(rmses_dares):.4f}\n")

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate models on synthetic test data")
    parser.add_argument("--fno", action="store_true", help="Evaluate FNO model")
    parser.add_argument("--diffusion", action="store_true", help="Evaluate Diffusion model")
    parser.add_argument("--kdm", action="store_true", help="Evaluate KDM+V model")
    parser.add_argument("--dares", action="store_true", help="Evaluate DARES model")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for evaluation (default: 1)")
    args = parser.parse_args()

    # if no models are selected, evaluate all
    if not (args.fno or args.diffusion or args.kdm or args.dares):
        args.fno = True
        args.diffusion = True
        args.kdm = True
        args.dares = True

    dataloader = build_dataloader("../gas-distribution-datasets/synthetic/test/test90.pt", batch_size=args.batch_size)
    
    fno_model = load_fno_model("models/fno/weights.pth") if args.fno else None
    diffusion_model = load_diffusion_model("models/diffusion/weights.pt") if args.diffusion else None
    kdm_model = load_kdm_model() if args.kdm else None
    dares_model = load_dares_model() if args.dares else None

    evaluate(dataloader, args.batch_size, args.fno, args.diffusion, args.kdm, args.dares, fno_model, diffusion_model, kdm_model, dares_model)


if __name__ == "__main__":
    main()