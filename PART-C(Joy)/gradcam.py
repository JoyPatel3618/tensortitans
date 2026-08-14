"""
Step 3 of Part C.

Grad-CAM answers "which part of the image made the model predict this
pathology?" - it produces a heatmap over the X-ray highlighting the
regions the model paid the most attention to. This is what makes the
model's decision explainable to a doctor instead of being a black box.

Run this after train_classifier.py, on any single image you want to
inspect.
"""

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

from train_classifier import build_model, PATHOLOGIES, MODEL_OUTPUT_PATH

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "results"

device = "cuda" if torch.cuda.is_available() else "cpu"


def generate_gradcam(image_path, pathology_name):
    model = build_model()
    model.load_state_dict(torch.load(MODEL_OUTPUT_PATH, map_location=device))
    model.eval()

    # hook into the last convolutional block - that's where spatial
    # information is still preserved (before it gets flattened for
    # classification), which is what Grad-CAM needs
    activations = {}
    gradients = {}

    def save_activation(module, input, output):
        activations["value"] = output

    def save_gradient(module, grad_input, grad_output):
        gradients["value"] = grad_output[0]

    target_layer = model.features[-1]
    target_layer.register_forward_hook(save_activation)
    target_layer.register_full_backward_hook(save_gradient)

    transform = T.Compose([
        T.Resize((224, 224)),
        T.Grayscale(num_output_channels=3),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    original_image = Image.open(image_path).convert("L")
    tensor = transform(original_image).unsqueeze(0).to(device)
    tensor.requires_grad_()

    output = model(tensor)
    class_index = PATHOLOGIES.index(pathology_name)
    score = output[0, class_index]

    model.zero_grad()
    score.backward()

    # average the gradients to get an "importance weight" per feature
    # channel, then combine with the activations to build the heatmap
    pooled_gradients = torch.mean(gradients["value"], dim=[0, 2, 3])
    activation_map = activations["value"].squeeze(0)

    for i in range(activation_map.shape[0]):
        activation_map[i, :, :] *= pooled_gradients[i]

    heatmap = torch.mean(activation_map, dim=0).detach().cpu().numpy()
    heatmap = np.maximum(heatmap, 0)  # only keep positive influence
    heatmap /= (heatmap.max() + 1e-8)  # normalize to 0-1

    return heatmap, float(score.item())


def save_overlay(image_path, heatmap, output_path):
    original = cv2.imread(str(image_path))
    heatmap_resized = cv2.resize(heatmap, (original.shape[1], original.shape[0]))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(original, 0.6, heatmap_colored, 0.4, 0)
    cv2.imwrite(str(output_path), overlay)


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(exist_ok=True)

    # example usage - point this at any X-ray + pathology you want to inspect
    example_image = BASE_DIR / "data" / "images" / "example.png"
    heatmap, confidence = generate_gradcam(example_image, "Pneumonia")

    print(f"predicted confidence: {confidence:.3f}")
    save_overlay(example_image, heatmap, OUTPUT_DIR / "gradcam_example.png")
    print("saved heatmap overlay to", OUTPUT_DIR / "gradcam_example.png")
