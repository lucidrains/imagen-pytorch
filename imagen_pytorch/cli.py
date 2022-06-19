import click
import torch
from functools import reduce
from pathlib import Path

from imagen_pytorch import Imagen
from imagen_pytorch.version import __version__

def safeget(dictionary, keys, default = None):
    return reduce(lambda d, key: d.get(key, default) if isinstance(d, dict) else default, keys.split('.'), dictionary)

def simple_slugify(text, max_length = 255):
    return text.replace("-", "_").replace(",", "").replace(" ", "_").replace("|", "--").strip('-_')[:max_length]

def main():
    pass

@click.command()
@click.option('--model', default = './imagen.pt', help = 'path to trained Imagen model')
@click.option('--cond_scale', default = 5, help = 'conditioning scale (classifier free guidance) in decoder')
@click.argument('text')
def imagen(
    model,
    cond_scale,
    text
):
    model_path = Path(model)
    full_model_path = str(model_path.resolve())
    assert model_path.exists(), f'model not found at {full_model_path}'
    loaded = torch.load(str(model_path))

    version = safeget(loaded, 'version')
    print(f'loading Imagen from {full_model_path}, saved at version {version} - current package version is {__version__}')

    init_params = safeget(loaded, 'init_params')
    model_params = safeget(loaded, 'model_params')

    imagen = Imagen(**init_params)
    imagen.load_state_dict(model_params)

    pil_image = imagen(text, cond_scale = cond_scale, return_pil_images = True)

    return pil_image.save(f'./{simple_slugify(text)}.png')
