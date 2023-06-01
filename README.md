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
| Using a model to detect water and apply proper masking | [Jupyter notebook](notebooks/water_segmentation.ipynb) |
| Controlling the size of texture elements after training | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1MWuwxoUd-oLUP3k1hWe9EQ4QuUvguJjt) |
| Training with patches | [Jupyter notebook](notebooks/patch_tests.ipynb) |
| Combining water masking, patches, multi-scale, to generate texture. FINAL IMAGE FROM HERE | [Jupyter notebook](notebooks/vector_field_motion_water_masked.ipynb) |

## Explanation needed to understand and use/experiment with each notebook

* CLIP loss between 2 images: nothing to do, this just shows that the loss works to compare 2 images
* CLIP loss between target text and generated image : in the cell "Setup training", you can modify the line `input_dict['target_text'] = ` to any text you see fit as a target for the model
* Animating the part of an image containing water : in the cell "Training configuration", there is a comment explaining how to modify the target appearance and vector field to those of an image that exhibits issues. The default target works properly.
* Using a model to detect water and apply proper masking : Simple notebook to showcase the masking model, no real options to modify apart from input image.
* Controlling the size of texture elements after training : in the cell "Generate videos" at the bottom, you can modify the function `save_video()` to get different results after training (no need to re-train after each of these modifications). This will allow controlling the size of the texture elements. More detailed explanations can be found in the notebook as comments.
* Combining water masking, patches, multi-scale, to generate texture. FINAL IMAGE FROM HERE : Contains many options that can all be modified in the args at the top of the notebook. Currently using a simple directional vector field, but by changing the corresponding option `self.motion_vector_field_name` to custom, will use the vector field imported corresponding to the image. 

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
