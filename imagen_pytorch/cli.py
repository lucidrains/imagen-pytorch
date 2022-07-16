import click
import torch
from functools import reduce
from pathlib import Path

from imagen_pytorch import ImagenConfig, ElucidatedImagenConfig
from imagen_pytorch.version import __version__

def exists(val):
    return val is not None

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

    # get version

    version = safeget(loaded, 'version')
    print(f'loading Imagen from {full_model_path}, saved at version {version} - current package version is {__version__}')

    # get imagen parameters and type

    imagen_params = safeget(loaded, 'imagen_params')
    imagen_type = safeget(loaded, 'imagen_type')

    if imagen_type == 'original':
        imagen_klass = ImagenConfig
    elif imagen_type == 'elucidated':
        imagen_klass = ElucidatedImagenConfig
    else:
        raise ValueError(f'unknown imagen type {imagen_type}')

    assert exists(imagen_params) and exists(imagen_type), 'imagen type and configuration not saved in this checkpoint'

    imagen = imagen_klass(**imagen_params).create()

    imagen.cuda()

    # get trained model parameters

    model_params = safeget(loaded, 'model')

    # load model parameters

    imagen.load_state_dict(model_params)

    # generate image

    pil_image = imagen.sample(text, cond_scale = cond_scale, return_pil_images = True)

    image_path = f'./{simple_slugify(text)}.png'
    pil_image[0].save(image_path)

    print(f'image saved to {str(image_path)}')
    return
