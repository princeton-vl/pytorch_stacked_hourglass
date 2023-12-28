import os
import numpy as np
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
    def __init__(self, root_dir, output_res, augment=False, num_workers=32, only10=False):
        self.root_dir = root_dir
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
            transforms.Resize((self.output_res, self.output_res)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
        ])(image)

        heatmaps = self.generate_heatmaps(points, self.output_res)

        return image_tensor, heatmaps

    def generate_heatmaps(self, points, output_res):
        num_keypoints = len(points) // 2
        heatmaps = np.zeros((num_keypoints, output_res, output_res), dtype=np.float32)

        for i in range(num_keypoints):
            x, y = int(points[i, 0] * output_res), int(points[i, 1] * output_res)
            if 0 <= x < output_res and 0 <= y < output_res:
                # Simple binary heatmap; consider using Gaussian distribution for better results
                heatmaps[i, y, x] = 1

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
    points = (points - 0.5) * rotation_matrix * scale + 0.5 + np.array(translations) / np.array([image.width, image.height])
    points = np.clip(points, 0, 1)

    transformed_image = transforms.ColorJitter(contrast=(0.8, 1.2))(transformed_image)

    return transformed_image, points
