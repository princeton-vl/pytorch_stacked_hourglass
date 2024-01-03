import os
import numpy as np
import scipy.stats as st
import pandas as pd
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from multiprocessing import Pool
from tqdm import tqdm

# Assuming TF refers to torchvision.transforms.functional
import torchvision.transforms.functional as TF

class CoordinateDataset(Dataset):
    def __init__(self, root_dir, im_sz, output_res, augment=False, num_workers=32, only10=False):
        self.root_dir = root_dir
        self.im_sz = im_sz
        self.output_res = output_res
        self.augment = augment
        csv_file = os.path.join(root_dir, 'Data.csv')
        self.data_frame = pd.read_csv(csv_file, header=0).head(10) if only10 else pd.read_csv(csv_file, header=0)

        image_paths = [os.path.join(self.root_dir, img_name) for img_name in self.data_frame.iloc[:, 0]]
        with Pool(num_workers) as pool:
            self.images = list(tqdm(pool.imap(Image.open, image_paths), total=len(image_paths)))

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        image = self.images[idx]
        points = self.data_frame.iloc[idx, 1:].values.astype('float').reshape(-1, 2)

        if self.augment:
            image, points = custom_transform(image, points)

        image_tensor = transforms.Compose([
            transforms.Resize((self.im_sz, self.im_sz)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
        ])(image)

        heatmaps = self.generate_heatmaps(points, self.output_res, use_gaussian=True)

        return image_tensor, heatmaps
    
    def generate_gaussian_heatmap(self, size, sigma):
        """Generate a 2D Gaussian heatmap."""
        interval = (2*sigma+1.)/(size)
        x = np.linspace(-sigma-interval/2., sigma+interval/2., size+1)
        kern1d = np.diff(st.norm.cdf(x))
        kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
        kernel = kernel_raw/kernel_raw.sum()
        return kernel

    def generate_heatmaps(self, points, output_res, use_gaussian=False, sigma=1):
        num_keypoints = len(points)
        heatmaps = np.zeros((num_keypoints, output_res, output_res), dtype=np.float32)
        size = 6*sigma + 1  # for use_gaussian=True

        for i in range(num_keypoints):
            x, y = int(points[i, 0] * output_res), int(points[i, 1] * output_res)
            if 0 <= x < output_res and 0 <= y < output_res:
                # Simple binary heatmap; consider using Gaussian distribution for better results
                if not use_gaussian:
                    heatmaps[i, y, x] = 1
                else:
                    gaussian = self.generate_gaussian_heatmap(size, sigma)
                    # Ensure the Gaussian is placed correctly on the heatmap
                    x_start, y_start = x - size // 2, y - size // 2
                    x_end, y_end = x_start + size, y_start + size
                    # Handle edge cases
                    x_start, y_start = max(0, x_start), max(0, y_start)
                    x_end, y_end = min(output_res, x_end), min(output_res, y_end)
                    heatmap_part = heatmaps[i, y_start:y_end, x_start:x_end]
                    gaussian_part = gaussian[:y_end-y_start, :x_end-x_start]

                    heatmaps[i, y_start:y_end, x_start:x_end] = np.maximum(heatmap_part, gaussian_part)

        return torch.tensor(heatmaps, dtype=torch.float32)

def custom_transform(image, points, degree_range=(-15, 15), translate_range=(0.1, 0.1), scale_range=(0.8, 1.2)):
    angle = random.uniform(*degree_range)
    translations = (random.uniform(-translate_range[0], translate_range[0]) * image.width,
                    random.uniform(-translate_range[1], translate_range[1]) * image.height)
    scale = random.uniform(*scale_range)

    transformed_image = TF.affine(image, angle=angle, translate=translations, scale=scale, shear=0)

    # Update points according to transformations
    cos_a, sin_a = np.cos(np.radians(angle)), np.sin(np.radians(angle))
    rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])

    # Apply rotation matrix to each point individually
    transformed_points = np.zeros_like(points)
    for i, point in enumerate(points):
        shifted_point = (point - 0.5) * scale
        rotated_point = np.dot(shifted_point, rotation_matrix)
        transformed_points[i] = rotated_point + 0.5 + np.array(translations) / np.array([image.width, image.height])

    transformed_points = np.clip(transformed_points, 0, 1)
    transformed_image = transforms.ColorJitter(contrast=(0.8, 1.2))(transformed_image)

    return transformed_image, transformed_points
