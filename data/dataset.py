import torch
from torch.utils.data import Dataset
from data.transforms import CreateInput
import random

class GDMDataset(Dataset):
    def __init__(self, root, mode: str, split_ratio: float=1, num_known=50, crop_size=64, mean=1.813, std=3.730, transform=None):
        """
        ...
        train: if True returns training samples, else validation samples
        split_ratio: float in (0,1), percentage of training samples
        crop_size: size of the square crop to extract from each frame
        ...
        """
        assert crop_size <= 90, "Crop size must be smaller than or equal to 90"

        if mode not in ["train", "val", "test"]:
            raise ValueError("mode must be 'train', 'val', or 'test'")
        if (split_ratio <= 0 or split_ratio > 1):
            raise ValueError("split_ratio must be in (0, 1]")

        self.root = root
        self.crop_size = crop_size
        self.mode = mode
        self.split_ratio = split_ratio
        self.mean = mean
        self.std = std
        self.transform = transform
        self.num_known = num_known

        data_tensor = torch.load(root, weights_only=True)
        # Remove the first 30 seconds
        data_tensor = data_tensor.reshape(-1, 600, 90, 90)  # now shape [360, 600, 90, 90]
        data = data_tensor[:, 30:, :, :]  # now shape [360, 570, 90, 90]
        N, T, H, W = data.shape

        # Log-transform and clamp
        log_data = torch.clamp(torch.log(data), min=0)

        # Z-normalize
        self.data = (log_data - self.mean) / self.std

        # Flatten all snapshots into one dimension: [360 * 570, 90, 90]
        self.data = self.data.reshape(N * T, H, W)

        # Determine indices for split
        num_total = len(self.data)
        split_index = int(self.split_ratio * num_total)

        if self.mode == "train":
            self.indices = list(range(0, split_index))
        elif self.mode == "val":
            self.indices = list(range(split_index, num_total))
        elif self.mode =="test":
            # get a random subset (according to the split_ratio) of the full dataset for testing
            num_test = int(split_ratio * num_total)
            self.indices = random.sample(range(num_total), num_test)
        else: 
            # This should never happen due to the earlier check
            raise ValueError("mode must be 'train', 'val', or 'test'")


    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        image = self.data[self.indices[idx]]

        while True:
            # Random crop
            top = random.randint(0, image.shape[0] - self.crop_size)
            left = random.randint(0, image.shape[1] - self.crop_size)
            crop = image[top:top + self.crop_size, left:left + self.crop_size]

            # Ensure the crop has some variance
            if crop.var() >= 1e-3:
                break

        if self.transform:
            # Add a channel dimension for torchvision transforms
            crop = crop.unsqueeze(0)  # [1, H, W]
            crop = self.transform(crop)
            crop = crop.squeeze(0) # Back to [H, W]
       
        while True:
            X, y = CreateInput(num_known=self.num_known)(crop.unsqueeze(0))  # Add channel dim for CreateInput   

            # Check to only return samples that contain any meaningful measurements
            mask = X[0:1, :, :]
            vals = X[1:2, :, :]
            if vals[mask.bool()].var() > 1e-3: 
                break

        return X, y