import torch.nn as nn
from torchvision import models


def create_backbone(name: str, pretrained: bool = True):
    name = name.lower()

    model = None
    
    factories = {
        "resnet18": models.resnet18,
        "resnet34": models.resnet34,
        "resnet50": models.resnet50,
        "resnet101": models.resnet101,
        "mobilenet_v2": models.mobilenet_v2,
        "densenet121": models.densenet121,
        "efficientnet_b0": getattr(models, "efficientnet_b0", None),
        "vit_b_16": getattr(models, "vit_b_16", None),
        "convnext_tiny": getattr(models, "convnext_tiny", None),
    }

    if name in factories and factories[name] is not None:
        factory = factories[name]
        # muitas versões do torchvision aceitam pretrained kw; se falhar, tenta sem
        try:
            model = factory(pretrained=pretrained)
        except TypeError:
            model = factory()
    else:
        # tentativa direta: se existir no módulo models
        if hasattr(models, name):
            factory = getattr(models, name)
            try:
                model = factory(pretrained=pretrained)
            except TypeError:
                model = factory()
        else:
            raise ValueError(
                f"Backbone '{name}' não encontrado em torchvision. Use use_timm=True para timm.")
    return model
