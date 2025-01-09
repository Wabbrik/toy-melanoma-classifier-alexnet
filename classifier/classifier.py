import os
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

from classifier.dataset import ClassifierClasses, labeled_dataset, test_dataset, train_dataset

# Hyperparameters
BATCH_SIZE: int = 32
NUM_EPOCHS: int = 10
LEARNING_RATE: float = 0.001

DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data transformations
TRANSFORM: transforms.Compose = transforms.Compose(
    [
        transforms.Resize((227, 227)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)


class CustomModel(nn.Module):
    def __init__(self, num_classes: int):
        super(CustomModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.AdaptiveAvgPool2d((6, 6)),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 6 * 6, 4096),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(-1, 6 * 6 * 256)
        x = self.classifier(x)
        return x


def initialize_model() -> nn.Module:
    model: nn.Module = CustomModel(num_classes=len(ClassifierClasses))
    return model


def train_model(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    num_epochs: int,
    tqdm_adapter: Any,
) -> nn.Module:
    model.train()
    for epoch in range(num_epochs):
        running_loss: float = 0.0
        with tqdm_adapter(data_loader, desc=f"Epoch {epoch + 1}/{num_epochs}") as progress_bar:
            for inputs, labels in progress_bar:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()

                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # Backward pass and optimize
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                progress_bar.set_postfix(loss=(running_loss / len(data_loader.dataset)))

    return model


def save_model(model: nn.Module, path: str) -> None:
    torch.save(model.state_dict(), path)


def load_model(path: str) -> nn.Module:
    model: nn.Module = initialize_model()
    model.load_state_dict(torch.load(path, weights_only=True))
    model = model.to(DEVICE)
    model.eval()
    return model


def prepare_data_loader(dataset_path: str, transform: transforms.Compose, tqdm_adapter: Any) -> DataLoader:
    labeled_data = labeled_dataset(dataset_path)
    images, labels = [], []

    for img_path, label in tqdm_adapter(labeled_data.items(), desc="Preparing data loader"):
        image = transform(Image.open(img_path).convert("RGB"))
        images.append(image)
        labels.append(label)

    tensor_images = torch.stack(images)
    tensor_labels = torch.tensor([int(label == ClassifierClasses.malignant.name) for label in labels])

    dataset = torch.utils.data.TensorDataset(tensor_images, tensor_labels)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)


def classify_image(model: nn.Module, image: Image.Image) -> dict[str, float]:
    image_tensor: torch.Tensor = TRANSFORM(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs: torch.Tensor = model(image_tensor)

        return {
            classification.name: confidence.item()
            for classification, confidence in zip(ClassifierClasses, outputs.softmax(1)[0])
        }


def try_load_cached_model(tqdm_adapter: Any):
    if os.path.exists("model.pth"):
        print("Loading saved model...")
        model: nn.Module = load_model("model.pth")
    else:
        train_data_loader = prepare_data_loader(train_dataset(), TRANSFORM, tqdm_adapter)
        print("Training model...")
        model = initialize_model()
        model = model.to(DEVICE)

        # Loss and optimizer
        criterion: nn.Module = nn.CrossEntropyLoss()
        optimizer: optim.Optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

        model = train_model(model, train_data_loader, criterion, optimizer, NUM_EPOCHS, tqdm_adapter)

        print("Saving model...")
        save_model(model, "model.pth")
    return model
