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
        self.dropout = nn.Dropout2d(0.2)

    def forward(self, x):
        out = self.conv(x) + self.shortcut(x)
        return self.dropout(out)


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
        self.bn = nn.BatchNorm2d(out_channels * 3)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout2d(0.2)

    def forward(self, x):
        x1 = self.conv1x1(x)
        x2 = self.conv3x3(x)
        x3 = self.conv5x5(x)
        x = torch.cat([x1, x2, x3], dim=1)
        x = self.bn(x)
        x = self.relu(x)
        return self.dropout(x)


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

    model_resnet = models.resnet50()
    num_ftrs = model_resnet.fc.in_features
    model_resnet.fc = nn.Linear(num_ftrs, 1)
    model_resnet = model_resnet.to(device)
    checkpoint_path = "trained_models/resnet_comp.pth"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model_resnet.load_state_dict(checkpoint["model_state_dict"])
    test_preds_res = test_model(model_resnet, test_loader)

    model_vgg = models.vgg13()
    features = list(model_vgg.features)
    modified_features = []
    i = 0
    while i < len(features):
        layer = features[i]
        modified_features.append(layer)

        if isinstance(layer, nn.Conv2d):
            if i + 1 < len(features) and not isinstance(
                features[i + 1], nn.BatchNorm2d
            ):
                modified_features.append(nn.BatchNorm2d(layer.out_channels))

        if isinstance(layer, nn.MaxPool2d) and i > 6:
            dropout_prob = 0.2 if i < 20 else 0.3
            modified_features.append(nn.Dropout2d(dropout_prob))

        i += 1

    model_vgg.features = nn.Sequential(*modified_features)
    num_ftrs = model_vgg.classifier[6].in_features
    model_vgg.classifier[6] = nn.Linear(num_ftrs, 1)
    model_vgg = model_vgg.to(device)
    checkpoint_path = "trained_models/vgg_comp.pth"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model_vgg.load_state_dict(checkpoint["model_state_dict"])
    test_preds_vgg = test_model(model_vgg, test_loader)

    model_custom = CustomCNN()
    model_custom = model_custom.to(device)
    checkpoint_path = "trained_models/custom_comp.pth"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model_custom.load_state_dict(checkpoint["model_state_dict"])
    test_preds_cus = test_model(model_custom, test_loader)

    test_preds_res = np.array(test_preds_res)
    test_preds_vgg = np.array(test_preds_vgg)
    test_preds_cus = np.array(test_preds_cus)

    predictions_stack = np.stack(
        [test_preds_res, test_preds_vgg, test_preds_cus], axis=1
    )

    ensemble_preds = np.apply_along_axis(
        lambda x: np.bincount(x, minlength=2).argmax(), axis=1, arr=predictions_stack
    )

    pred_labels = ["positive" if pred == 1 else "negative" for pred in ensemble_preds]

    results_path = "results.txt"
    with open(results_path, "w") as f:
        for label in pred_labels:
            f.write(label + "\n")

    print(f"Ensemble results saved to {results_path}")
