from model.backbone.resnet.resnet import resnet18, resnet34, resnet50
from model.backbone.efficientnet.efficientnet import EfficientNet


def get_backbone(name, kwargs):
    if name == 'resnet18':
        return resnet18()

    if name == 'resnet34':
        return resnet34()

    if name == 'resnet50-pysot':
        return resnet50()

    if name.startswith('efficientnet'):
        return EfficientNet.from_name(name, **kwargs)

    raise RuntimeError("No backbone for {}".format(name))
