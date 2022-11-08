import click
import torch
from pathlib import Path

from imagen_pytorch import load_imagen_from_checkpoint
from imagen_pytorch.version import __version__
from imagen_pytorch.utils import safeget
from imagen_pytorch import ImagenTrainer
from imagen_pytorch import t5

import accelerate
def exists(val):
    return val is not None

def simple_slugify(text, max_length = 255):
    return text.replace("-", "_").replace(",", "").replace(" ", "_").replace("|", "--").strip('-_')[:max_length]

def main():
    pass

def _convert_yes_no_to_bool(value):
    return {"yes": True, "no": False}[value.lower()]
def _convert_true_false_to_bool(value):
    return {"true": True, "false": False}[value.lower()]

def _convert_int_greater_zero(value):
    value = int(value)
    assert value > 0
    return value

def _convert_int_greater_zero_or_none(value):
    if value.lower() is not 'none':
        value = int(value)
        assert value > 0
        return value
    return None

def _convert_t5_name(value):
    t5.get_encoded_dim(value)
    return value

def _convert_tuple_int(value):
    assert len(value) > 0
    t =  map(int, value.split(' '))
    if len(t) is 1:
        return t[0]
    return t
    
def _convert_tuple_bool(value):
    assert len(value) > 0
    t = map(_convert_true_false_to_bool, value.split(' '))
    if len(t) is 1:
        return t[0]
    return t

def ask(input_text, convert_value=None, default=None, error_message=None):
    while True:
        result = input(input_text)
        try:
            if default is not None and len(result) == 0:
                return default
            return convert_value(result) if convert_value is not None else result
        except:
            if error_message is not None:
                print(error_message)
@click.group()
def imagen():
    pass

@imagen.command()
@click.option('--model', default = './imagen.pt', help = 'path to trained Imagen model')
@click.option('--cond_scale', default = 5, help = 'conditioning scale (classifier free guidance) in decoder')
@click.option('--load_ema', default = True, help = 'load EMA version of unets if available')
@click.argument('text')
def sample(
    model,
    cond_scale,
    load_ema,
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

    imagen = load_imagen_from_checkpoint(str(model_path), load_ema_if_available = load_ema)
    imagen.cuda()

    # generate image

    pil_image = imagen.sample(text, cond_scale = cond_scale, return_pil_images = True)

    image_path = f'./{simple_slugify(text)}.png'
    pil_image[0].save(image_path)

    print(f'image saved to {str(image_path)}')
    return

@imagen.command()
def config():
    use_elucidated = ask('Should the Elucidated DDPM be used? [yes/NO]: ', _convert_yes_no_to_bool, False, 'Please enter yes or no')
    use_video = ask('Should the model be used for video? [yes/NO]: ', _convert_yes_no_to_bool, False, 'Please enter yes or no')
    is_cond = ask('Should the model be conditional? [YES/no]: ', _convert_yes_no_to_bool, True, 'Please enter yes or no')
    num_unets = ask('How many unets? [3]: ', _convert_int_greater_zero, 3, 'Please enter a value greater than 0')

    t5_name = None
    if is_cond:
        t5_name = ask('Please enter the name of the t5 model [google/t5-v1_1-base]: ', _convert_t5_name, 'google/t5-v1_1-base', 'Please enter a valid t5 model name')

    for index in range(num_unets):
        dim = ask('Dimension [128]: ', _convert_int_greater_zero, 128, 'Please enter a value greater than 0')
        dim_mults = ask('Multiplicators for the dimension (numbers separated by space): ', _convert_tuple_int, error_message = 'Please enter a valid list of numbers greater than 1')
        num_resnet_blocks = ask('Number of Resnet blocks?: ', _convert_int_greater_zero, error_message = 'Please enter a value greater than 0')
        layer_attns = ask('Layer attentions (bools separated by space): ', _convert_tuple_bool, error_message = 'Please enter a valid list of bools')
        layer_cross_attns = ask('Layer cross attentions (bools separated by space): ', _convert_tuple_bool, error_message = 'Please enter a valid list of bools')
        attn_heads = ask('Number of attention heads? [8]: ', _convert_int_greater_zero, 8, 'Please enter a value greater than 0')
        res = ask('image size of this Unet?: ', _convert_int_greater_zero, error_message='Please enter a value greater than 0')
        random_crop = ask('Random crop size for the unet?: ', _convert_int_greater_zero_or_none, error_message='Please enter a value greater than 0 or None')
    

@imagen.command()
@click.option('--model', default = './imagen.pt', help = 'path to the Imagen model checkpoint')
@click.option('--config', default = './imagen.cfg', help = 'path to the Imagen model config')
@click.option('--unet', default = 1, help = 'unet to train')
def train(
    model,
    config,
    unet
):
    # check model path

    model_path = Path(model)
    full_model_path = str(model_path.resolve())
    assert model_path.exists(), f'model not found at {full_model_path}'
    loaded = torch.load(str(model_path))

    # check config path
    
    config_path = Path(config)
    full_config_path = str(config_path.resolve())
    assert config_path.exists(), f'config not found at {full_config_path}'
    loaded = torch.load(str(config_path))

    # get version

    version = safeget(loaded, 'version')
    print(f'loading Imagen from {full_model_path}, saved at version {version} - current package version is {__version__}')

    # get imagen parameters and type

    trainer = ImagenTrainer(imagen_checkpoint_path=model_path)
    trainer.cuda()
