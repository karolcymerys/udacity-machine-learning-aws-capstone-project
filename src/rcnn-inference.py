import io
import json
import os

import torch
from PIL import Image
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights, maskrcnn_resnet50_fpn_v2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRANSFORM = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT.transforms()

def model_fn(model_dir):
    model = maskrcnn_resnet50_fpn_v2(weights=None)
    model.load_state_dict(torch.load(os.path.join(model_dir, "model.pth"), map_location=DEVICE, weights_only=True))
    return model.to(DEVICE).eval()


def input_fn(request_body, content_type):
    if content_type in ("application/x-image", "image/jpeg", "image/png", "application/octet-stream"):
        return [TRANSFORM(Image.open(io.BytesIO(request_body)).convert("RGB"))]

    raise ValueError(f"Unsupported content type: {content_type}")


def predict_fn(input_data, model):
    with torch.no_grad():
        output = model(input_data.to(DEVICE))
    return output[0]['scores'].float().cpu().tolist()


def output_fn(prediction, accept):
    return json.dumps({"predictions": prediction}), "application/json"
