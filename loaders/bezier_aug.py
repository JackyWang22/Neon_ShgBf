import torch
from PIL import Image
import numpy as np


def factorial(n):
    """Compute factorial using a loop in PyTorch for stability."""
    if n == 0 or n == 1:
        return torch.tensor(1)
    result = torch.tensor(1)
    for i in range(2, n + 1):
        result *= i
    return result

def binomial_coefficient(n, k):
    """Compute binomial coefficient using factorials."""
    return factorial(n) // (factorial(k) * factorial(n - k))

def bernstein_polynomial(n, i, t):
    """Calculate the Bernstein polynomial value."""
    bin_coeff = binomial_coefficient(n, i)
    return bin_coeff * (t ** i) * ((1 - t) ** (n - i))

def bezier_curve(control_points, t):
    """Compute the Bézier curve value at each t using Bernstein polynomials."""
    n = len(control_points) - 1
    curve_val = torch.zeros_like(t)
    for i in range(n + 1):
        bern_poly = bernstein_polynomial(n, i, t)
        curve_val += control_points[i] * bern_poly
    return curve_val

def stochastic_intensity_transformation(image):
    """
    Apply stochastic non-linear intensity transformation to an image.
    Args:
    - image (torch.Tensor): Input image tensor of shape (C, H, W) or (B, C, H, W)
    - threshold (float): Threshold for deciding whether to invert intensities.
    """
    device = image.device
    b, c, h, w = image.shape if image.ndim == 4 else (1, *image.shape)
    image = image.view(b, c, h * w)  # Flatten the spatial dimensions

    # Generate control points for the Bézier curve uniformly sampled from [0, 1]
    control_points = torch.rand(b, 3, device=device)  # 10 control points for each batch item

    # Apply the Bézier curve transformation to each pixel
    for i in range(b):
        for j in range(c):
            image[i, j] = bezier_curve(control_points[i], image[i, j])

    # Reshape back to original image dimensions
    image = image.view(b, c, h, w)

    return image



# img = Image.open('1B_H4_SHG.tif')
# img = np.array(img)
# img = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0).contiguous() / 255.0
# img = img.mean(dim=1, keepdim=True)
# min_val = img.min()
# max_val = img.max()
# img = (img - min_val) / (max_val - min_val)
# Image.fromarray((img.squeeze().numpy() * 255).astype(np.uint8)).save('original_image.tif')

# print(img.max(), img.min())
# # Example usage:
# intensity_transformed_image = stochastic_intensity_transformation(img)
# print(intensity_transformed_image.max(), intensity_transformed_image.min())


# intensity_transformed_image = intensity_transformed_image.squeeze().numpy()
# min_val = intensity_transformed_image.min()
# max_val = intensity_transformed_image.max()
# intensity_transformed_image = (intensity_transformed_image - min_val) / (max_val - min_val)
# Image.fromarray((intensity_transformed_image * 255).astype(np.uint8)).save('intensity_transformed_image.tif')