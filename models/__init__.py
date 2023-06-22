from .resnet import ResNet50, ResNet101, ResNet152

MAP = {
    "resnet50": ResNet50,
    "resnet101": ResNet101,
    "resnet152": ResNet152,
}


def get_model(name: str, num_classes, channels):
    return MAP[name](num_classes=num_classes, channels=channels)
