import os
from typing import Union, List, Tuple

import cv2
import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt
from omegaconf import OmegaConf


def water_to_mask(image_path: str, size: Tuple[int, int], pretrained_path: str, device: str, c: int = 12) -> torch.Tensor:
    """
    Creates a binary mask from an image using deeplab with weights pre-trained for river detection.
    https://researchdata.reading.ac.uk/282/
    Parameters
    ----------
    image_path : str
        path to reference image
    size : Tuple[int, int]
        target size of mask (ignoring the first two channels)
    pretrained_path : str
        path to pre-trained models so that deeplab can be loaded
    device : str
        whether to use cuda
    c : int
        number of channels the mask should have in the end. It will always affect just the first 3.

    Returns
    -------
    torch.Tensor
        the mask in shape (1, 12, size)
    """

    with torch.no_grad():
        pretrained_path = os.path.join(pretrained_path, "deeplabv3")
        CONFIG = OmegaConf.load(os.path.join(pretrained_path, "water.yaml"))
        with open(os.path.join(pretrained_path, CONFIG.DATASET.LABELS)) as f:
            classes = {}
            for label in f:
                label = label.rstrip().split("\t")
                classes[int(label[0])] = label[1].split(",")[0]
        model = torch.hub.load("kazuto1011/deeplab-pytorch", "deeplabv2_resnet101", pretrained=None, n_classes=2)
        # path can be: laura_whole.pth / lago_2steps.pth / water_whole.pth / water_2steps.pth
        state_dict = torch.load(os.path.join(pretrained_path, "laura_whole.pth"), map_location=lambda storage, loc: storage)
        model.load_state_dict(state_dict)
        model.eval()
        model.to(device)
        # getting the image:
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        scale = CONFIG.IMAGE.SIZE.TEST / max(image.shape[:2])
        print(type(image))
        image = cv2.resize(image, dsize=None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        # Subtract mean values, as the original preprocess has this
        image = image.astype(np.float32)
        image -= np.array(
            [
                float(CONFIG.IMAGE.MEAN.B),
                float(CONFIG.IMAGE.MEAN.G),
                float(CONFIG.IMAGE.MEAN.R),
            ]
        )

        # Convert to torch.Tensor and add "batch" axis
        image = torch.from_numpy(image.transpose(2, 0, 1)).float().unsqueeze(0)
        image = image.to(device)
        _, _, H, W = image.shape

        # Image -> Probability map
        logits = model(image)
        logits = torch.nn.functional.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
        probs = torch.nn.functional.softmax(logits, dim=1)[0]
        probs = probs.cpu().numpy()
        labelmap = np.argmax(probs, axis=0)
        labels = np.unique(labelmap)
        mask = np.where(labelmap == labels[1], 1.0, 0.0)
        # reshaping mask, only supports crop
        h, w = mask.shape
        cut_pixel = abs(w - h) // 2
        if w > h:
            mask = mask[:, cut_pixel:w - cut_pixel]
        else:
            mask = mask[cut_pixel:h - cut_pixel, :]

        mask = cv2.resize(mask, dsize=size, interpolation=cv2.INTER_CUBIC)

        mask = torch.from_numpy(np.float32(mask))
        mask = mask[None, None, :, :]
        ones = torch.ones(1, c - 3, mask.shape[2], mask.shape[3])  # .to(update_mask.get_device())
        mask = torch.cat((mask, mask, mask, ones), 1)
        print(mask.shape)
        return mask


def flow_to_mask(flow_list: Union[List[torch.Tensor], torch.Tensor], eps: float = 0.05, c: int = 12) -> torch.Tensor:
    """
    Creates a mask from a given list of flows

    Parameters
    ----------

    flow_list : Union[List[torch.Tensor], torch.Tensor]
        List of flow tensors
    eps : float
        how close to zero do we consider a vector to be zero in the mask
    c : int
        number of channels
    Returns
    -------
    Tensor
        The mask in shape [1, 12, flow.shape]
    """
    if isinstance(flow_list, list):
        total_flow = torch.zeros_like(flow_list[0])
        for flow in flow_list:
            total_flow += torch.abs(flow)
        total_flow /= len(flow_list)
        total_flow = total_flow.cpu()
    else:
        total_flow = flow_list
    # mask making
    update_mask = torch.where((torch.abs(total_flow[0, 0]) <= eps) & (torch.abs(total_flow[0, 1]) <= eps), 0.0, 1.0)[None, None, :, :]
    ones = torch.ones(1, c - 3, total_flow.shape[2], total_flow.shape[3])  # .to(update_mask.get_device())
    update_mask = torch.cat((update_mask, update_mask, update_mask, ones), 1)

    return update_mask


def smooth_mask(mask: torch.Tensor, smoothness: int = 15, sigma: int = 3) -> torch.Tensor:
    """
    Smooths a given tensor using gaussian blur
    Parameters
    ----------
    mask : torch.Tensor
        The tensor to smooth
    smoothness : int
        The size of the kernel used
    sigma : int
        sigma used to define the gaussian
    Returns
    -------
    torch.Tensor
        smoothed input
    """
    if smoothness > 0 and smoothness % 2 == 1:
        mask = torchvision.transforms.GaussianBlur(kernel_size=(smoothness, smoothness), sigma=sigma)(mask)
    return mask
