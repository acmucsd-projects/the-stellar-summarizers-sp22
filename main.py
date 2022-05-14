from lib.video_dataset import  VideoFrameDataset, ImglistToTensor
from torchvision import transforms
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

import cv2
import requests
import os


if __name__ == '__main__':
    videos_root = 'frames'
    
    dataset = VideoFrameDataset(
        root_path=videos_root,
        annotationfile_path=annotation_file,
        num_segments=5,
        frames_per_segment=1,
        imagefile_template='img_{:0.5d}.jpg',
        transform=None,
        test_mode=False
    )