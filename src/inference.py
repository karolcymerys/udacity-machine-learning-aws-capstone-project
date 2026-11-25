import io
import json
import os
from concurrent.futures import ThreadPoolExecutor
from itertools import repeat

import boto3
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from botocore.config import Config

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_WORKERS = 16

s3 = boto3.client("s3", config=Config(max_pool_connections=MAX_WORKERS))

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
        return {
            'keys': [''],
            'batch': transform(image).unsqueeze(0)
        }
    elif content_type == "application/json":
        if isinstance(request_body, bytes):
            request_body = request_body.decode("utf-8")

        payload = json.loads(request_body)
        bucket, keys = payload["bucket"], payload["keys"]

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            tensors = list(pool.map(load_image_from_s3, repeat(bucket), keys))

        return {
            "keys": keys,
            "batch": torch.stack(tensors)
        }

    raise ValueError(f"Unsupported content type: {content_type}")


def load_image_from_s3(bucket, key):
    body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    image = Image.open(io.BytesIO(body)).convert("RGB")
    return transform(image)


def predict_fn(input_data, model):
    batch = input_data["batch"]
    predictions = []

    with torch.no_grad():
        for i in range(0, len(batch), 64):
            out = model(batch[i:i + 64].to(DEVICE)).squeeze(1)
            predictions.extend(round_to_closest_non_negative_integer(out).cpu().tolist())

    return {
        "keys": input_data["keys"],
        "predictions": predictions
    }


def round_to_closest_non_negative_integer(batch):
    return batch.round().clamp(min=0).long()


def output_fn(predictions, accept):
    return json.dumps(predictions), "application/json"
