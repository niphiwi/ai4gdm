import torch
from torch.utils.data import Dataset
from data.transforms import CreateInput
import random

def upscale_sensor_grid(sensor_data):
    """
    Upscale sensor data from 6x5 grid to 64x64 grid with correct spatial positioning.
    
    Args:
        sensor_data: torch.Tensor of shape [30, 3800, 6, 5]
        
    Returns:
        output: torch.Tensor of shape [30, 3800, 64, 64]
        mask: torch.Tensor of shape [64, 64] (boolean), True where sensors are located
    """
    batch, time, height, width = sensor_data.shape  # [30, 3800, 6, 5]
    
    # Physical parameters
    area_height = 9.0  # meters (vertical)
    area_width = 7.5   # meters (horizontal)
    sensor_spacing = 1.5  # meters (same in both directions)
    
    # Calculate grid resolution
    grid_height = 64
    grid_width = 64
    pixel_size_h = area_height / grid_height  # ~0.140625 m
    pixel_size_w = area_height / grid_height  # ~0.140625 m
    
    # Calculate sensor positions (centered in their 1.5m cells)
    # Height: 6 sensors with 1.5m spacing = 6 * 1.5 = 9m total
    # Sensors at: 0.75m, 2.25m, 3.75m, 5.25m, 6.75m, 8.25m
    # Width: 5 sensors with 1.5m spacing = 5 * 1.5 = 7.5m total  
    # Sensors at: 0.75m, 2.25m, 3.75m, 5.25m, 6.75m
    sensor_positions_h = torch.arange(height, dtype=torch.float32) * sensor_spacing + sensor_spacing / 2
    sensor_positions_w = torch.arange(width, dtype=torch.float32) * sensor_spacing + sensor_spacing / 2
    
    # Convert to pixel indices in 64x64 grid
    sensor_pixels_h = sensor_positions_h / pixel_size_h  # positions in pixel coordinates
    sensor_pixels_w = sensor_positions_w / pixel_size_w
    
    # Create output tensor (initialized with zeros)
    output = torch.zeros(batch, time, grid_height, grid_width, dtype=sensor_data.dtype, device=sensor_data.device)
    
    # Create mask tensor (boolean)
    mask = torch.zeros(grid_height, grid_width, dtype=torch.bool, device=sensor_data.device)
    
    # Place sensor values at their corresponding positions
    # Round to nearest pixel
    sensor_indices_h = torch.round(sensor_pixels_h).long()
    sensor_indices_w = torch.round(sensor_pixels_w).long()
    
    # Ensure indices are within bounds
    sensor_indices_h = torch.clamp(sensor_indices_h, 0, grid_height - 1)
    sensor_indices_w = torch.clamp(sensor_indices_w, 0, grid_width - 1)
    
    # Map sensor values to grid and update mask
    for i, idx_h in enumerate(sensor_indices_h):
        for j, idx_w in enumerate(sensor_indices_w):
            output[:, :, idx_h, idx_w] = sensor_data[:, :, i, j]
            mask[idx_h, idx_w] = True
    
    return output, mask


class TokyoDataset(Dataset):
    def __init__(self, root, transform=None):
        """
        ...
        train: if True returns training samples, else validation samples
        split_ratio: float in (0,1), percentage of training samples
        crop_size: size of the square crop to extract from each frame
        ...
        """

        data = torch.load(root, weights_only=True)

        # Z-normalize
        mean = data.mean()
        std = data.std()
        data = (data - mean) / std

        # Upscale sensor grid from 6x5 to 64x64
        data, self.mask = upscale_sensor_grid(data)
        data = data.reshape(-1, 64, 64)

        self.data = data


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        X = self.data[idx]

        # concatenate mask as an additional channel
        mask_channel = self.mask.unsqueeze(0).to(X.device).float()  # shape
        X = torch.cat([mask_channel, X.unsqueeze(0)], dim=0)  # shape [2, 64, 64]

        return X
    