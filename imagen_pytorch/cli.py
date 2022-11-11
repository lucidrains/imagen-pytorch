import click
import torch
from pathlib import Path

from imagen_pytorch import load_imagen_from_checkpoint
from imagen_pytorch.version import __version__
from imagen_pytorch.data import Collator
from imagen_pytorch.utils import safeget
from imagen_pytorch import ImagenTrainer, ElucidatedImagenConfig, ImagenConfig
from datasets import load_dataset

import json

def exists(val):
    return val is not None

def simple_slugify(text, max_length = 255):
    return text.replace('-', '_').replace(',', '').replace(' ', '_').replace('|', '--').strip('-_')[:max_length]

def main():
    pass

@click.group()
def imagen():
    pass

@imagen.command(help = 'Sample from the Imagem model checkpoint')
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

@imagen.command(help = 'Generate a config for the Imagen model')
def config():
    config = {
        'type': 'original',
        'imagen': {
            'video': False,
            'timesteps': [1024, 512, 512],
            'image_sizes': [64, 256, 1024],
            'random_crop_sizes': [None, 64, 256],
            'condition_on_text': True,
            'cond_drop_prob': 0.1,
            'text_encoder_name': 'google/t5-v1_1-large',
            'unets': [
                {
                    'dim': 512,
                    'dim_mults': [1, 2, 3, 4],
                    'num_resnet_blocks': 3,
                    'layer_attns': [False, True, True, True],
                    'layer_cross_attns': [False, True, True, True],
                    'attn_heads': 8
                },
                {
                    'dim': 128,
                    'dim_mults': [1, 2, 3, 4],
                    'num_resnet_blocks': [2, 4, 8, 8],
                    'layer_attns': [False, False, False, True],
                    'layer_cross_attns': [False, False, False, True],
                    'attn_heads': 8
                },
                {
                    'dim': 128,
                    'dim_mults': [1, 2, 3, 4],
                    'num_resnet_blocks': [2, 4, 8, 8],
                    'layer_attns': False,
                    'layer_cross_attns': [False, False, False, True],
                    'attn_heads': 8
                }
            ],
        },
        'trainer': {
            'lr': 1e-4,

        },
        'dataset': {
            'name': 'laion/laion2B-en',
            'batch_size': 2048,
            'shuffle': True,
        },
        'image_label': None,
        'url_label': 'URL',
        'text_label': 'TEXT',
        'checkpoint_path': './imagen.pt',
        
    }
    with open('./imagen.cfg', 'w') as f:
        f.write(json.dumps(config, indent = 4))

@imagen.command(help = 'Train the Imagen model')
@click.option('--config', default = './imagen.cfg', help = 'path to the Imagen model config')
@click.option('--unet', default = 1, help = 'unet to train')
@click.option('--epoches', default = 1000, help = 'amount of epoches to train for')
@click.option('--text', required = False, help = 'text to sample with between epoches', type=str)
def train(
    config,
    unet,
    epoches,
    text
):
    # check config path
    
    config_path = Path(config)
    full_config_path = str(config_path.resolve())
    assert config_path.exists(), f'config not found at {full_config_path}'
    
    with open(config_path, 'r') as f:
        config_data = json.loads(f.read())

    assert 'checkpoint_path' in config_data, 'checkpoint path not found int config'
    
    model_path = Path(config_data['checkpoint_path'])
    full_model_path = str(model_path.resolve())
    
    if model_path.exists():
        loaded = torch.load(str(model_path))
        version = safeget(loaded, 'version')
        print(f'loading Imagen from {full_model_path}, saved at version {version} - current package version is {__version__}')
        trainer = ImagenTrainer(imagen_checkpoint_path=model_path)
    else:
        if config_data['type'] == 'elucidated':
            imagen = ElucidatedImagenConfig(
                **config_data['imagen']
            ).create()
        else:
            imagen = ImagenConfig(
                **config_data['imagen']
            ).create()
        trainer = ImagenTrainer(
            imagen=imagen,
            **config_data['trainer']
        )
    trainer.cuda()

    size = config_data['imagen']['image_sizes'][unet-1]

    max_batch_size = config_data['max_batch_size'] if 'max_batch_size' in config_data else 1

    trainer.add_train_dataset(
        ds = load_dataset(config_data['dataset_name'])['train'],
        collate_fn = Collator(
            image_size=size,image_label=config_data['image_label'],
            text_label=config_data['text_label'],
            url_label=config_data['url_label'],
            name=config_data['imagen']['text_encoder_name']
        ),
        **config_data['dataset']
    )

    for i in range(epoches):
        loss = trainer.train_step(unet_number=unet, max_batch_size = max_batch_size)
        print(f'loss: {loss}')

        #if not (i % 50):
        #    valid_loss = trainer.valid_step(unet_number = unet, max_batch_size = max_batch_size)
        #    print(f'valid loss: {valid_loss}')

        if not (i % 100) and trainer.is_main and text is not None: # is_main makes sure this can run in distributed
            images = trainer.sample(texts = [text], batch_size = 1, return_pil_images = True) # returns List[Image]
            images[0].save(f'./sample-{i // 100}.png')

    trainer.save(model_path)
