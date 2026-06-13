from numpy import dtype
import torchvision.transforms as transforms
import torch
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image


class DreamDriveDataset(Dataset):
    def __init__(self, csv_file,image_size=224):
        self.df = pd.read_csv(csv_file)
        self.image_size = image_size

        # Image pre-processing for ResNet backbone
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Load Image
        image = Image.open(row['image_path']).convert("RGB")

        # Extract and Normalise coordinates (0.0 to 1.0)
        norm_x = row['x_pixel']/ row['width']
        norm_y = row['y_pixel']/ row['height']

        # Transform Image
        image_tensor = self.transform(image)
        target_tensor = torch.tensor([norm_x, norm_y], dtype=torch.float32)

        return image_tensor, target_tensor


