from __future__ import absolute_import
import torch

from .pcb import *
from .mlfn import *
from .hacnn import *
from .osnet import *
from .senet import *
from .mudeep import *
from .nasnet import *
from .resnet import *
from .densenet import *
from .xception import *
from .osnet_ain import *
from .resnetmid import *
from .shufflenet import *
from .squeezenet import *
from .inceptionv4 import *
from .mobilenetv2 import *
from .resnet_ibn_a import *
from .resnet_ibn_b import *
from .shufflenetv2 import *
from .inceptionresnetv2 import *

__model_factory = {
    # image classification models
    'resnet18': resnet18,
    'resnet50': resnet50,
    'osnet_ibn_x1_0': osnet_ibn_x1_0,
    'osnet_x0_75': osnet_x0_75,
    'osnet_x0_5': osnet_x0_5,
    'osnet_x0_25': osnet_x0_25,
}


def show_avai_models():
    """Displays available models.

    Examples::
        >>> from torchreid import models
        >>> models.show_avai_models()
    """
    print(list(__model_factory.keys()))


def build_model(
    name, num_classes, loss='softmax', pretrained=True, use_gpu=True
):
    """A function wrapper for building a model.

    Args:
        name (str): model name.
        num_classes (int): number of training identities.
        loss (str, optional): loss function to optimize the model. Currently
            supports "softmax" and "triplet". Default is "softmax".
        pretrained (bool, optional): whether to load ImageNet-pretrained weights.
            Default is True.
        use_gpu (bool, optional): whether to use gpu. Default is True.

    Returns:
        nn.Module

    Examples::
        >>> from torchreid import models
        >>> model = models.build_model('resnet50', 751, loss='softmax')
    """
    avai_models = list(__model_factory.keys())
    if name not in avai_models:
        raise KeyError(
            'Unknown model: {}. Must be one of {}'.format(name, avai_models)
        )
    return __model_factory[name](
        num_classes=num_classes,
        loss=loss,
        pretrained=pretrained,
        use_gpu=use_gpu
    )
