# from video_dataset import  VideoFrameDataset, ImglistToTensor
# from torchvision import transforms
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
import matplotlib.pyplot as plt
import pandas as pd
import os


# created using https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#creating-a-custom-dataset-for-your-files
class SumMeDataset(Dataset):
    def __init__(self, annotations_filename, img_dir, transform=None, target_transform=None):

        self.annotation_filename = annotations_filename
        self.annotation = pd.read_csv(annotations_filename, header=0)

        self.img_dir = img_dir
        self.video_name = img_dir.split('/')[-1]
        self.frame_labels = self.annotation[self.annotation['video_name'] == self.video_name]['gt_score']
        print(self.frame_labels)

        # not implemented yet
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.frame_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, 'img_' + str(idx + 1).zfill(5) + '.jpg')
        image = read_image(img_path)
        label = self.frame_labels[idx]

        # not implemented yet
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label


annotations_filename = './frames/annotation.csv'
img_root = './frames'

if __name__ == '__main__':

    videos_root = './frames/Bike_Polo'

    dataset = SumMeDataset(annotations_filename, videos_root)

    # demo, shows the 1000th frame of the Bike_Polo video
    sample = dataset[0]
    print(sample)
    plt.imshow(sample[0].permute(1, 2, 0))  # put channel data as last dimension
    plt.show()
