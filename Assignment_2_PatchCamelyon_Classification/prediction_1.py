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

    model = models.vgg16()
    num_ftrs = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_ftrs, 1)
    model = model.to(device)

    checkpoint_path = "trained_models/vgg_non_comp.pth"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_preds_vgg = test_model(model, test_loader)

    pred_labels = ["positive" if pred == 1 else "negative" for pred in test_preds_vgg]

    results_path = "results.txt"
    with open(results_path, "w") as f:
        for label in pred_labels:
            f.write(label + "\n")

    print(f"Results saved to {results_path}")
