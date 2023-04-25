# Flow visualization code used from https://github.com/tomrunia/OpticalFlow_Visualization


# MIT License
#
# Copyright (c) 2018 Tom Runia
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Tom Runia
# Date Created: 2018-08-03

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import torch
import torchvision.transforms

from PIL import Image
import io

from typing import List

def plot_vec_field(vector_field, name="target", vmin=None, vmax=None):
    """
    Parameters
    ----------
    vector_field : numpy array with 
        the shape: 2 x H x W
    """

    _, H, W = vector_field.shape
    norm = np.sqrt(vector_field[0, ::-1] ** 2 + vector_field[1, ::-1] ** 2)

    if vmin is None:
        vmin = norm.min()
    if vmax is None:
        vmax = norm.max()

    normalize = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax, clip=False)

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection="rectilinear")
    title = f"{name} vector field."
    xs = np.linspace(-1.0, 1.0, W)
    ys = np.linspace(-1.0, 1.0, H)

    sp = plt.streamplot(
        xs,
        ys,
        vector_field[0, ::-1],
        -vector_field[1, ::-1],
        color=norm,
        linewidth=(norm + 0.05) / 1.25,
        norm=normalize,
        density=0.75,
        #         broken_streamlines=False,
        #         minlength=0.3,
        # vmin=2.0, vmax=2.0,
    )
    #     print(norm.min(), norm.max())
    #     ax.set_xlabel("X")
    #     ax.set_ylabel("Y")
    #     ax.set_title(title)
    # ax.axis('equal', adjustable='box')

    fig.colorbar(sp.lines)
    fig.canvas.draw()

    frame = plt.gca()
    frame.axes.xaxis.set_ticklabels([])
    frame.axes.yaxis.set_ticklabels([])
    frame.axes.get_xaxis().set_ticks([])
    frame.axes.get_yaxis().set_ticks([])
    frame.set_aspect('equal', adjustable='box')
    buf = io.BytesIO()
    fig.savefig(buf)
    plt.clf()
    plt.close()

    buf.seek(0)
    img = Image.open(buf)

    return img


def make_colorwheel():
    """
    Generates a color wheel for optical flow visualization as presented in:
        Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV, 2007)
        URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf
    Code follows the original C++ source code of Daniel Scharstein.
    Code follows the the Matlab source code of Deqing Sun.
    Returns:
        np.ndarray: Color wheel
    """

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))
    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255 * np.arange(0, RY) / RY)
    col = col + RY
    # YG
    colorwheel[col:col + YG, 0] = 255 - np.floor(255 * np.arange(0, YG) / YG)
    colorwheel[col:col + YG, 1] = 255
    col = col + YG
    # GC
    colorwheel[col:col + GC, 1] = 255
    colorwheel[col:col + GC, 2] = np.floor(255 * np.arange(0, GC) / GC)
    col = col + GC
    # CB
    colorwheel[col:col + CB, 1] = 255 - np.floor(255 * np.arange(CB) / CB)
    colorwheel[col:col + CB, 2] = 255
    col = col + CB
    # BM
    colorwheel[col:col + BM, 2] = 255
    colorwheel[col:col + BM, 0] = np.floor(255 * np.arange(0, BM) / BM)
    col = col + BM
    # MR
    colorwheel[col:col + MR, 2] = 255 - np.floor(255 * np.arange(MR) / MR)
    colorwheel[col:col + MR, 0] = 255
    return colorwheel


def flow_uv_to_colors(u, v, convert_to_bgr=False):
    """
    Applies the flow color wheel to (possibly clipped) flow components u and v.
    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun
    Args:
        u (np.ndarray): Input horizontal flow of shape [H,W]
        v (np.ndarray): Input vertical flow of shape [H,W]
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.
    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    flow_image = np.zeros((u.shape[0], u.shape[1], 3), np.uint8)
    colorwheel = make_colorwheel()  # shape [55x3]
    ncols = colorwheel.shape[0]
    rad = np.sqrt(np.square(u) + np.square(v))
    a = np.arctan2(-v, -u) / np.pi
    fk = (a + 1) / 2 * (ncols - 1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0
    for i in range(colorwheel.shape[1]):
        tmp = colorwheel[:, i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1 - f) * col0 + f * col1
        idx = (rad <= 1)
        col[idx] = 1 - rad[idx] * (1 - col[idx])
        col[~idx] = col[~idx] * 0.75  # out of range
        # Note the 2-i => BGR instead of RGB
        ch_idx = 2 - i if convert_to_bgr else i
        flow_image[:, :, ch_idx] = np.floor(255 * col)
    return flow_image


def flow_to_image(flow_uv, clip_flow=None, convert_to_bgr=False, rad_max=None):
    """
    Expects a two dimensional flow image of shape.
    Args:
        flow_uv (np.ndarray): Flow UV image of shape [H,W,2]
        clip_flow (float, optional): Clip maximum of flow values. Defaults to None.
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.
    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    assert flow_uv.ndim == 3, 'input flow must have three dimensions'
    assert flow_uv.shape[2] == 2, 'input flow must have shape [H,W,2]'
    if clip_flow is not None:
        flow_uv = np.clip(flow_uv, 0, clip_flow)
    u = flow_uv[:, :, 0]
    v = flow_uv[:, :, 1]
    rad = np.sqrt(np.square(u) + np.square(v))
    if rad_max is None:
        rad_max = np.max(rad)
    epsilon = 1e-5
    u = u / (rad_max + epsilon)
    v = v / (rad_max + epsilon)
    return flow_uv_to_colors(u, v, convert_to_bgr)


def flow_to_mask(flow_list: List[torch.Tensor], eps: float = 0.05, c: int = 12, smoothness: int = 0):
    """
    Creates a mask from a given list of flows

    Parameters
    ----------

    flow_list : List[torch.Tensor]
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
    total_flow = torch.zeros_like(flow_list[0])
    for flow in flow_list:
        total_flow += torch.abs(flow)
    total_flow /= len(flow_list)
    total_flow = total_flow.cpu()
    # mask making
    update_mask = torch.where((torch.abs(total_flow[0, 0]) < eps) & (torch.abs(total_flow[0, 1]) < eps), 0.0, 1.0)[None, None, :, :]
    if smoothness > 0:
        update_mask = torchvision.transforms.GaussianBlur(kernel_size=(smoothness, smoothness), sigma=3)(update_mask)
    print("Vector Mask: ")
    plt.imshow(update_mask[0].permute(1, 2, 0), vmin=0)
    plt.axis("off")
    plt.show()
    ones = torch.ones(1, c - 3, total_flow.shape[2], total_flow.shape[3])  # .to(update_mask.get_device())
    update_mask = torch.cat((update_mask, update_mask, update_mask, ones), 1)

    return update_mask
