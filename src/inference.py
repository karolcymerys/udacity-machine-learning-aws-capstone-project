import io
import json
import os

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def model_fn(model_dir):
    version = os.environ['RESNET_VERSION']

    if version == 'resnet18':
        model = models.resnet18(weights=None)
    elif version == 'resnet34':
        model = models.resnet34(weights=None)
    elif version == 'resnet50':
        model = models.resnet50(weights=None)
    elif version == 'resnet101':
        model = models.resnet101(weights=None)
    elif version == 'resnet152':
        model = models.resnet152(weights=None)
    else:
        raise ValueError('Unknown version')

    model.fc = nn.Linear(model.fc.in_features, 1)
    weights = torch.load(os.path.join(model_dir, "model.pth"), map_location=DEVICE, weights_only=True)
    model.load_state_dict(weights)
    return model.to(DEVICE).eval()


def input_fn(request_body, content_type):
    if content_type in ("application/x-image", "image/jpeg", "image/png", "application/octet-stream"):
        image = Image.open(io.BytesIO(request_body)).convert("RGB")
        return transform(image).unsqueeze(0)

    raise ValueError(f"Unsupported content type: {content_type}")


def predict_fn(input_data, model):
    with torch.no_grad():
        output = model(input_data.to(DEVICE)).squeeze(1)

    return round_to_closest_non_negative_integer(output).cpu().tolist()


def round_to_closest_non_negative_integer(batch):
    return batch.round().clamp(min=0).long()


def output_fn(prediction, accept):
    return json.dumps({"predictions": prediction}), "application/json"
