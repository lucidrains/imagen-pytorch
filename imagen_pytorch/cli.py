import click
import torch
from pathlib import Path

from imagen_pytorch import load_imagen_from_checkpoint
from imagen_pytorch.version import __version__
from imagen_pytorch.utils import safeget
from imagen_pytorch import ImagenTrainer
from imagen_pytorch import t5

import json

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
def _convert_int_percent(value):
    value = int(value)
    assert value > 0
    return value / 100

def _convert_int_greater_zero_or_none(value):
    if value.lower() != 'none':
        value = int(value)
        assert value > 0
        return value
    return None

def _convert_t5_name(value):
    t5.get_encoded_dim(value)
    return value

def _convert_tuple_int(value):
    assert len(value) > 0
    t =  list(map(int, value.split(' ')))
    if len(t) == 1:
        return t[0]
    return t
    
def _convert_tuple_bool(value):
    assert len(value) > 0
    t = list(map(_convert_true_false_to_bool, value.split(' ')))
    if len(t) == 1:
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
    timesteps = ask('timesteps (numbers separated by space or one number): ', _convert_tuple_int, error_message = 'Please enter a value greater than 0 or a list of numbers')
    is_cond = ask('Should the model be conditional on text? [YES/no]: ', _convert_yes_no_to_bool, True, 'Please enter yes or no')
    
    num_unets = ask('How many unets? [3]: ', _convert_int_greater_zero, 3, 'Please enter a value greater than 0')
    t5_name = None
    cond_drop_prob = 0.1
    if is_cond:
        t5_name = ask('Name of the t5 model [google/t5-v1_1-base]?: ', _convert_t5_name, 'google/t5-v1_1-base', 'Please enter a valid t5 model name')
        cond_drop_prob = ask('How much percent of conditionals should be dropped? (1 - 100)?: ', _convert_int_percent, error_message= 'Please enter a value greater than 0')

    config = {
        'type': 'elucidated' if use_elucidated else 'original',
        'video': use_video,
        'timesteps': timesteps,
        't5': t5_name,
        'cond_drop_prob': cond_drop_prob,
        'unets': [],
        'image_sizes': [],
        'condition_on_text': is_cond
    }
    random_crops = []
    for index in range(num_unets):
        print(f'Unet {index+1}')
        dim = ask('Dimension [128]?: ', _convert_int_greater_zero, 128, 'Please enter a value greater than 0')
        dim_mults = ask('Multiplicators for the dimension (numbers separated by space or one number)?: ', _convert_tuple_int, error_message = 'Please enter a value greater than 0 or a list of numbers')
        num_resnet_blocks = ask('Number of Resnet blocks (numbers separated by space or one number)?: ', _convert_tuple_int, error_message = 'Please enter a value greater than 0 or a list of numbers')
        layer_attns = ask('Layer attentions (bools separated by space or one bool): ', _convert_tuple_bool, error_message = 'Please enter a valid list of bools or a bool')
        layer_cross_attns = ask('Layer cross attentions (bools separated by space or one bool)?: ', _convert_tuple_bool, error_message = 'Please enter a valid list of bools or a bool')
        attn_heads = ask('Number of attention heads? [8]: ', _convert_int_greater_zero, 8, 'Please enter a value greater than 0')
        resolution = ask('Image size of this Unet?: ', _convert_int_greater_zero, error_message='Please enter a value greater than 0')
        random_crop = ask('Random crop size for the unet?: ', _convert_int_greater_zero_or_none, error_message='Please enter a value greater than 0 or None')
        config['unets'].append({
            'dim': dim,
            'dim_mults': dim_mults,
            'num_resnet_blocks': num_resnet_blocks,
            'layer_attns': layer_attns,
            'layer_cross_attns': layer_cross_attns,
            'attn_heads': attn_heads,
        })
        config['image_sizes'].append(resolution)
        
        random_crops.append(random_crop)
    
    if any(random_crops):
        config['random_crop_sizes'] = random_crops
    
    with open('./imagen.cfg', 'w') as f:
        f.write(json.dumps(config, indent = 4))
    

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
