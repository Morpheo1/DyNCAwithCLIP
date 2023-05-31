# CS413 Project at EPFL, supervised by E. Pajouheshgar - based on DyNCA

[![arXiv](https://img.shields.io/badge/arXiv-2108.00946-b31b1b.svg)](https://arxiv.org/abs/2211.11417)

[[Original Project Website](https://dynca.github.io/)]

This repository contains all the necessary code to reproduce our results.
Compared to the original project, we mainly modified the files `utils/loss/appearance_loss.py` and `models/dynca.py`, as well as adding slightly modifying the files in `utils/misc` where necessary to make them work with our changes.

## Run in Google Colab or locally

Our contributions include the possibility to train the model using CLIP loss instead of another appearance loss to use text as target instead of an image. We also implemented the ability to add motion to the part of an image containing water. Our third improvement relates to the size of texture elements, which can be an issue if we have limited training resources. All of these have at least one notebook that can be run to reproduce our main results.

| **Functionality** | **Colab or Jupyter Notebook**|
|:-----------------:|:------------------:|
| CLIP loss between 2 images | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ZbjToOcnUyVwtrsXRjgVqT5pKgIzOCgr) |
| CLIP loss between target text and generated image | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/17IPqQlnjPd5_4x_WLg5uIovtKNWketcA) |
| Animating the part of an image containing water | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1yA1pRRDnXzf2NY_E_FI6MtwjNdzhjip3) |
| Using a model to detect water and apply proper masking when training and adding the animation | [Jupyter notebook](notebooks/vector_field_motion_water_masked.ipynb) |
| Controlling the size of texture elements after training | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1MWuwxoUd-oLUP3k1hWe9EQ4QuUvguJjt) |
| Training with patches | [TODO](notebooks/vector_field_motion_water_masked.ipynb) |
| Combining animating the water in an image and scale control | [TODO](notebooks/vector_field_motion_water_masked.ipynb) |

## Explanation needed to understand and use/experiment with each notebook



## Installing Requirements

To be able to use the model that detects water areas, download it [here](https://drive.google.com/drive/folders/1q8W_CGnMSOsaB3TGTrSabuX-oZ89vusw?usp=sharing) and add it to `pretrained_models/deeplabv3`.

Check `requirements.txt` to know which packages to install to be able to run the jupyter notebooks

## Citation (copied from the original project's repository)

If you make use of our work, please cite our paper:

```
@InProceedings{pajouheshgar2022dynca,
  title     = {DyNCA: Real-Time Dynamic Texture Synthesis Using Neural Cellular Automata},
  author    = {Pajouheshgar, Ehsan and Xu, Yitao and Zhang, Tong and S{\"u}sstrunk, Sabine},
  booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2023},
}
```