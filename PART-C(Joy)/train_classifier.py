"""
Step 2 of Part C.

Trains a DenseNet-121 to predict, from a chest X-ray alone, the
probability of each of the 14 NIH pathologies. This is the VISUAL half of
the "multi-modal" model described in the problem statement - it does not
yet combine clinical text (ClinicalBERT) with the image. That text-fusion
part is a separate next step (see README) since it needs paired
image+clinical-note data we don't have set up yet.

This is a multi-label problem (an X-ray can show more than one pathology
at once), which is why we use a sigmoid output + BCE loss per class,
instead of the single softmax you'd use for a "pick one class" problem.
"""

from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.models import densenet121, DenseNet121_Weights
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent

LABELS_CSV = BASE_DIR / "data" / "labels_multilabel.csv"

# ---- LOCAL path (default) ----
IMAGES_DIR = BASE_DIR / "data" / "images"

# ---- KAGGLE path (use this instead if running in a Kaggle notebook) ----
# IMAGES_DIR = Path("/kaggle/input/nih-chest-xrays/sample/images/")

MODEL_OUTPUT_PATH = BASE_DIR / "model" / "densenet_multilabel.pt"

PATHOLOGIES = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
    "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema",
    "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia",
]

NUM_IMAGES = 1000  # small subset for a same-day baseline
EPOCHS = 5         # kept small today - raise once the pipeline is confirmed working
BATCH_SIZE = 16

device = "cuda" if torch.cuda.is_available() else "cpu"


class ChestXrayDataset(Dataset):
    def __init__(self, dataframe, images_dir, transform):
        self.df = dataframe
        self.images_dir = images_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(self.images_dir / row["Image Index"]).convert("L")
        image = self.transform(image)
        labels = torch.tensor(row[PATHOLOGIES].values.astype("float32"))
        return image, labels


def build_model():
    model = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
    # replace the final layer: instead of 1000 ImageNet classes, output
    # one score per pathology, with a sigmoid so each is an independent
    # 0-1 probability (a scan can have more than one finding at once)
    model.classifier = nn.Sequential(
        nn.Linear(model.classifier.in_features, len(PATHOLOGIES)),
        nn.Sigmoid(),
    )
    return model.to(device)


def main():
    MODEL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    labels_df = pd.read_csv(LABELS_CSV).head(NUM_IMAGES)

    transform = T.Compose([
        T.Resize((224, 224)),
        T.Grayscale(num_output_channels=3),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = ChestXrayDataset(labels_df, IMAGES_DIR, transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = build_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.BCELoss()  # binary cross-entropy - standard choice for multi-label problems

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for images, labels in tqdm(dataloader, desc=f"epoch {epoch + 1}/{EPOCHS}"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            predictions = model(images)
            loss = loss_fn(predictions, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"epoch {epoch + 1}: avg loss = {total_loss / len(dataloader):.4f}")

    torch.save(model.state_dict(), MODEL_OUTPUT_PATH)
    print("saved model to", MODEL_OUTPUT_PATH)


if __name__ == "__main__":
    main()
