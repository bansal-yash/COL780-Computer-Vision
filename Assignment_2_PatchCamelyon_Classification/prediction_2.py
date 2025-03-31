import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.v2 as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


class PCamDataset(Dataset):
    def __init__(self, image_dir, transform):
        self.image_dir = image_dir
        self.transform = transform
        self.image_filenames = os.listdir(image_dir)
        self.length = len(self.image_filenames)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.image_filenames[index])
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        return img


def test_model(model, test_loader):
    test_preds, test_probs = [], []

    model.eval()
    with torch.no_grad():
        for images in test_loader:
            images = images.to(device)
            outputs = model(images).squeeze(dim=1)
            probs = torch.sigmoid(outputs)

            test_preds.extend(probs.cpu().numpy())
            test_probs.extend(probs.cpu().numpy())

    final_preds = (np.array(test_probs) > 0.5).astype(int)

    return final_preds


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            stride,
            padding,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.relu(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        super(ResidualBlock, self).__init__()
        stride = 2 if downsample else 1
        self.conv = DepthwiseSeparableConv(
            in_channels, out_channels, kernel_size=3, stride=stride
        )
        self.shortcut = (
            nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=stride, bias=False
            )
            if in_channels != out_channels or downsample
            else nn.Identity()
        )

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)


class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InceptionModule, self).__init__()
        self.conv1x1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, padding=0, bias=False
        )
        self.conv3x3 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.conv5x5 = nn.Conv2d(
            in_channels, out_channels, kernel_size=5, padding=2, bias=False
        )

    def forward(self, x):
        x1 = self.conv1x1(x)
        x2 = self.conv3x3(x)
        x3 = self.conv5x5(x)
        return torch.cat([x1, x2, x3], dim=1)


class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        self.init_conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.res_block1 = ResidualBlock(64, 64)
        self.res_block2 = ResidualBlock(64, 128, downsample=True)
        self.res_block3 = ResidualBlock(128, 256, downsample=True)
        self.inception = InceptionModule(256, 256)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((6, 6))
        self.fc = nn.Sequential(
            nn.Linear(256 * 3 * 6 * 6, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        x = self.init_conv(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.inception(x)
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


if __name__ == "__main__":
    x_test_path = "test_images"
    test_transform = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.Resize((96, 96)),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    test_dataset = PCamDataset(x_test_path, transform=test_transform)

    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
    )

    model = CustomCNN()
    model = model.to(device)

    checkpoint_path = "trained_models/custom_non_comp.pth"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_preds_custom = test_model(model, test_loader)

    pred_labels = ["positive" if pred == 1 else "negative" for pred in test_preds_custom]

    results_path = "results.txt"
    with open(results_path, "w") as f:
        for label in pred_labels:
            f.write(label + "\n")

    print(f"Results saved to {results_path}")
