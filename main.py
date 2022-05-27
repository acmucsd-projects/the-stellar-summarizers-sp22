import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import torch
import torchvision.transforms.functional as F
import torchvision.models as models
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, ToPILImage
from torchvision.io import read_image
from torchvision.utils import make_grid



# dataloader
class SumMeDataset(Dataset):
    def __init__(self, annotations_filename, img_dir, transform=None, target_transform=None):

        self.annotation_filename = annotations_filename
        self.annotation = pd.read_csv(annotations_filename, header=0)

        self.img_dir = img_dir
        self.video_name = img_dir.split('/')[-1]
        self.frame_labels = self.annotation[self.annotation['video_name'] == self.video_name]['gt_score']

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.frame_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, 'img_' + str(idx + 1).zfill(5) + '.jpg')
        image = read_image(img_path)
        label = self.frame_labels.iloc[idx]

        if self.transform:
            image = self.transform(ToPILImage()(image.to('cpu'))) # transform to float point 
        if self.target_transform:
            label = self.target_transform(label)

        return image, label


# encoder
class new_resNext(nn.Module):
    def __init__(self, fc_size=2048, large=False, pretrained=True):
        super(encoder,self).__init__()
        if large:
            self.model = models.resnext101_32x8d(pretrained=pretrained)
        else:
            self.model = models.resnext50_32x4d(pretrained=pretrained)
        self.model.fc = nn.Linear(fc_size, 10)

    def forward(self,x):
        x = self.model(x)
        return x


class Model(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, n_layers):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        # define the layers
        self.rnn = nn.LSTM(input_size, hidden_size, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input):
        # batch_size = input.size(0)
        # print(f'batch size = {batch_size}')
        # hidden = self.initHidden(batch_size)
        # print(hidden)
        output, hidden = self.rnn(input)
        output = self.fc(output)
        return output, hidden

    def initHidden(self, batch_size):
        # TO BE IMPLEMENTED (Q: does LSTM need this?)
        hidden = torch.zeros(1, self.hidden_size)
        return hidden


def show(imgs):
    """Helper function to convert tensor images to PIL images and display them"""
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False, figsize=(20, 16))
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])


if __name__ == '__main__':

    # get device for training
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using: {device.upper()}")

    # set filepath for dataloader
    annotations_filename = './frames/annotation.csv'
    videos_root = './frames/Cooking' # change this to choose which video to load

    # instantiating the dataset 
    dataset = SumMeDataset(annotations_filename, videos_root, transform=ToTensor())
    print(f'This dataset has {len(dataset)} samples')

    # data loader
    batch_size = 64
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    features, labels = next(iter(dataloader))
    print("Data Loader:")
    print(f"Feature batch shape: {features.size()}")
    print(f"Labels batch shape: {labels.size()}")

    encoder = new_resNext().to(device) # models move to different device inplace

    model = Model(10, 1, 10, 1) # TODO: tuning
    model.to(device)

    # define loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    """Training"""
    num_epochs = 10
    loss_history = [] # TO BE IMPLEMENTED

    for epoch in range(num_epochs):
        print(f'epoch: {epoch}')

        for features, labels in dataloader:

            # move features and labels to GPU
            features = features.to(device) # data doesn't move to different device inplace ! 
            labels = labels.to(device)
            labels = torch.reshape(labels, (1, batch_size, 1))

            optimizer.zero_grad()

            # encode image features using resNext and reshape
            encoded_features = encoder(features)
            batched_seq = torch.reshape(encoded_features, (1, batch_size, 10))
            batched_seq.to(device)

            output, hidden = model(batched_seq) # forward pass
            loss = criterion(output, labels.float()) # calcualte MSE
            loss.backward()
            optimizer.step()


    """Testing"""
    test_loss = 0
    tensor_preds = []
    with torch.no_grad():
        for features, labels in dataloader:

            # encode the features
            features = features.to(device)
            labels = labels.to(device)
            labels = torch.reshape(labels, (1, batch_size, 1))

            encoded_features = encoder(features)
            batched_seq = torch.reshape(encoded_features, (1, batch_size, 10))

            pred, _= model(batched_seq)
            test_loss += criterion(pred, labels.float()).item()
            tensor_preds += torch.flatten(pred)

    test_loss /= batch_size
    print(f'test_loss = {test_loss}')

    # parse predictions into a numpy array
    preds = []
    for pred in tensor_preds:
        preds.append(pred.item())
    preds = np.array(preds)

    """Qualitative Evaluation"""
    # get N top ranked frames
    best_feature = []
    top_imgs_i = np.sort(np.argsort(preds)[-10:]) # sort by predicted values then sort the index so the frames are in order
    for img_i in top_imgs_i:
        img_path = os.path.join(videos_root, 'img_' + str(img_i + 1).zfill(5) + '.jpg')
        best_feature.append(read_image(img_path))

    grid = make_grid(best_feature, nrow=5)
    show(grid)