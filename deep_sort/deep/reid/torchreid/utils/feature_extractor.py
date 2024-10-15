# from __future__ import absolute_import
# import numpy as np
import torch
# import torchvision.transforms as T
# import cv2
# from PIL import Image

# from torchreid.utils import (
#     check_isfile, load_pretrained_weights, compute_model_complexity
# )
from torchreid.utils import (
    check_isfile, load_pretrained_weights)
from torchreid.models import build_model


class FeatureExtractor(object):
    """A simple API for feature extraction.

    FeatureExtractor can be used like a python function, which
    accepts input of the following types:
        - a list of strings (image paths)
        - a list of numpy.ndarray each with shape (H, W, C)
        - a single string (image path)
        - a single numpy.ndarray with shape (H, W, C)
        - a torch.Tensor with shape (B, C, H, W) or (C, H, W)

    Returned is a torch tensor with shape (B, D) where D is the
    feature dimension.

    Args:
        model_name (str): model name.
        model_path (str): path to model weights.
        image_size (sequence or int): image height and width.
        pixel_mean (list): pixel mean for normalization.
        pixel_std (list): pixel std for normalization.
        pixel_norm (bool): whether to normalize pixels.
        device (str): 'cpu' or 'cuda' (could be specific gpu devices).
        verbose (bool): show model details.

    Examples::

        from torchreid.utils import FeatureExtractor

        extractor = FeatureExtractor(
            model_name='osnet_x1_0',
            model_path='a/b/c/model.pth.tar',
            device='cuda'
        )

        image_list = [
            'a/b/c/image001.jpg',
            'a/b/c/image002.jpg',
            'a/b/c/image003.jpg',
            'a/b/c/image004.jpg',
            'a/b/c/image005.jpg'
        ]

        features = extractor(image_list)
        print(features.shape) # output (5, 512)
    """

    def __init__(
        self,
        model_name='',
        model_path='',
        # image_size=(256, 128), # --> for PIL image
        # image_size=(128, 256),  # for cv2 image
        # pixel_mean=[0.485, 0.456, 0.406],
        # pixel_std=[0.229, 0.224, 0.225],
        # pixel_norm=True,
        device='cuda',
        # verbose=True
    ):
        # Build model
        model = build_model(
            model_name,
            num_classes=1,
            pretrained=True,
            use_gpu=device.startswith('cuda')
        )
        model.eval()

        # if verbose:
        #     num_params, flops = compute_model_complexity(
        #         model, (1, 3, image_size[0], image_size[1])
        #     )
        #     print('Model: {}'.format(model_name))
        #     print('- params: {:,}'.format(num_params))
        #     print('- flops: {:,}'.format(flops))

        if model_path and check_isfile(model_path):
            load_pretrained_weights(model, model_path)

        # Build transform functions
        # transforms = []
        # transforms += [T.Resize(image_size)]  # takes size sequence as (h,w)
        # transforms += [T.ToTensor()]
        # if pixel_norm:
        #     transforms += [T.Normalize(mean=pixel_mean, std=pixel_std)]
        # preprocess = T.Compose(transforms)

        # to_pil = T.ToPILImage()

        device = torch.device(device)
        model.to(device)

        # Class attributes
        self.model      = model
        # self.preprocess = preprocess
        # self.to_pil     = to_pil
        self.device     = device
        # self.image_size = image_size
        # self.pixel_mean = pixel_mean
        # self.pixel_std  = pixel_std

    def __call__(self, input):
        # if isinstance(input, list):
        if isinstance(input, torch.Tensor):    
            # images = []

            # image_size=(128, 256)
            
            # pixel_mean=[0.485, 0.456, 0.406],
            # pixel_std=[0.229, 0.224, 0.225],

            # mean = torch.tensor(pixel_mean).view(3, 1, 1)
            # std = torch.tensor(pixel_std).view(3, 1, 1)
            
            # for element in input:
            #     # if isinstance(element, str):
            #     #     image = Image.open(element).convert('RGB')

            #     # elif isinstance(element, np.ndarray):
            #     #     image = self.to_pil(element)
            #     if isinstance(element, np.ndarray):     #element is HWC
            #         # image = self.to_pil(element)        #converted to WHC and RGB
            #         image = cv2.cvtColor(element, cv2.COLOR_BGR2RGB)

            #     else:
            #         raise TypeError(
            #             'Type of each element must belong to [str | numpy.ndarray]'
            #         )

            #     image = cv2.resize(image, image_size)
            #     image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

            #     # mean = torch.tensor(self.pixel_mean).view(3, 1, 1)
            #     # std = torch.tensor(self.pixel_std).view(3, 1, 1)

            #     image = (image - mean) / std        # tensor --> [c,h,w]
            #     # image = self.preprocess(image)      # tensor --> [c,h,w]
            #     images.append(image)            

            # images = torch.stack(images, dim=0)     # tensor --> [N,c,h,w]  --> [4,3,256,128]
            # images = images.to(self.device)
            images = input.to(self.device)     

        # elif isinstance(input, str):
        #     image = Image.open(input).convert('RGB')
        #     image = self.preprocess(image)
        #     images = image.unsqueeze(0).to(self.device)

        # elif isinstance(input, np.ndarray):
        #     image = self.to_pil(input)
        #     image = self.preprocess(image)
        #     images = image.unsqueeze(0).to(self.device)

        # elif isinstance(input, torch.Tensor):
        #     if input.dim() == 3:
        #         input = input.unsqueeze(0)
        #     images = input.to(self.device)

        else:
            raise NotImplementedError

        with torch.no_grad():
            features = self.model(images)

        return features