import torch

def create_random_mask(image_size=64, num_known=50, device='cuda'):
    """
    Create a random mask of known pixels of shape [1, image_size, image_size].
    """
    mask = torch.zeros((1, image_size, image_size), dtype=torch.float32, device=device)
    known_indices = torch.randperm(image_size * image_size)[:num_known]
    ys = known_indices // image_size
    xs = known_indices % image_size
    mask[0, ys, xs] = 1.0  # Set known pixels to 1
    return mask

# Create a transform that creates and applies a mask to the input images
def apply_mask(image, mask):
    """
    Apply a mask to the input image.
    """ 
    masked_image = image * mask
    return masked_image

class CreateInput:
    """
    Custom transform to create a masked input image.
    """
    def __init__(self, num_known):
        self.num_known = num_known

    def __call__(self, base_image):
        mask = create_random_mask(image_size=base_image.shape[-1], num_known=self.num_known, device=base_image.device)
        masked_image = apply_mask(base_image, mask)

        # concatenate mask and masked image along channel dimension
        input_image = torch.cat([mask, masked_image], dim=0)

        return input_image, base_image  # X_mask, X_samples, y