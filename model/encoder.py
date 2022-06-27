# encoder
import torch
from torchvision.models import resnext50_32x4d, resnext101_32x8d

class Encoder(torch.nn.Module):
    """Pretrained ResNeXt for encoding frame features into latent features"""
    def __init__(self, output_size, fc_size=2048, large=False, pretrained=True):
        super(Encoder, self).__init__()
        if large:
            self.model = resnext101_32x8d(pretrained=pretrained)
        else:
            self.model = resnext50_32x4d(pretrained=pretrained)

        self.model.fc = torch.nn.Linear(fc_size, output_size)

    def forward(self, x):
        x = self.model(x)
        return x

# test code
if __name__ == '__main__':
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using: {device.upper()}")

    output_feature_size = 256
    new_resNext = Encoder(output_size=output_feature_size).to(device)