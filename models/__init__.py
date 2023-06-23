from .resnet import ResNet50, ResNet101, ResNet152
from .vit import VIT

MAP = {
    "resnet50": ResNet50,
    "resnet101": ResNet101,
    "resnet152": ResNet152,
    "vit": VIT,
}


def get_model(name: str, **model_config):
    return MAP[name](model_config)
