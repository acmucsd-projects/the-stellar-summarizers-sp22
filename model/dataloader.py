import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Resize, ConvertImageDtype
from torchvision.io import read_image

import pandas as pd
import os
    
class SumMeDataset(Dataset):
    """Custom Dataset class wrapper for loading frame features and ground truth importance score"""
    def __init__(self, annotations_filename, img_dir, transform=None, target_transform=None, device='cpu'):
        self.annotation_filename = annotations_filename
        self.annotation = pd.read_csv(annotations_filename, header=0)
        self.img_dir = img_dir
        self.video_name = img_dir.split('/')[-1]
        self.frame_labels = self.annotation[self.annotation['video_name'] == self.video_name]['gt_score']
        self.transform = transform
        self.target_transform = target_transform
        self.device = device

    def __len__(self):
        return len(self.frame_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, 'img_' + str(idx + 1).zfill(5) + '.jpg')
        image = read_image(img_path)
        label = self.frame_labels.iloc[idx]
        if self.transform:
            image = self.transform(image)
        image = image.to(self.device)
        return image, label, idx



if __name__ == '__main__':
    
    image_transformation = nn.Sequential(
        Resize(size=(140, 320)),
        ConvertImageDtype(torch.float),
    )

    batch_size = 10
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using: {device.upper()}")

    # load the data and label
    annotations_filename = './frames/annotation.csv'
    videos_root = './frames/Jumps'
    dataset = SumMeDataset(annotations_filename, videos_root, transform=image_transformation)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    print(f'{videos_root} has {len(dataset)} samples')
    features, labels, index = next(iter(dataloader))
    print("DataLoader:")
    print(f"- loaded batch of frames with index: {index}")
    print(f"- feature batch shape: {features.size()}")
    print(f"- labels batch shape: {labels.size()}")