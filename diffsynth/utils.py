import torch
import numpy as np

def preprocess_image(image):
    image = torch.Tensor(np.array(image, dtype=np.float32) * (2 / 255) - 1).permute(2, 0, 1).unsqueeze(0)
    return image

def preprocess_images(images):
    return [preprocess_image(image) for image in images]
