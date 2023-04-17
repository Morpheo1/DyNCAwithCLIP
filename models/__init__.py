import os
from functools import partial

import torch

import progressbar

pbar = None


def show_progress(block_num, block_size, total_size):
    global pbar
    if pbar is None:
        pbar = progressbar.ProgressBar(maxval=total_size)
        pbar.start()

    downloaded = block_num * block_size
    if downloaded < total_size:
        pbar.update(downloaded)
    else:
        pbar.finish()
        pbar = None


def _load_MSOEmultiscale_model(model_name, models_path, download=False):
    assert model_name == 'two_stream_dynamic_model'
    output_dir = os.path.join(models_path, "two_stream")
    output = os.path.join(output_dir, model_name+".pth")
    if not os.path.exists(output):
        download = True

    if download:
        import gdown
        url = 'https://drive.google.com/uc?id=10qoSx0P3TJzf17bUN42x1ZAFNjr-J69f'
        os.makedirs(output_dir, exist_ok=True)
        gdown.download(url, output, quiet=False)

    from models.MSOEmultiscale import MSOEmultiscale

    model = MSOEmultiscale()
    states_dict = torch.load(output)
    model.load_state_dict(states_dict)
    model = model.eval()

    return model


_model_factories = {}
_model_factories['two_stream_dynamic'] = partial(_load_MSOEmultiscale_model, model_name='two_stream_dynamic_model')


def get_available_models():
    return _model_factories.keys()


def get_model(name, *args, **kwargs):
    return _model_factories[name](*args, **kwargs)
