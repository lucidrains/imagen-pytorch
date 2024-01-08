import click
import torch
from pathlib import Path
import pkgutil

from imagen_pytorch import load_imagen_from_checkpoint
from imagen_pytorch.version import __version__
from imagen_pytorch.data import Collator
from imagen_pytorch.utils import safeget
from imagen_pytorch import ImagenTrainer, ElucidatedImagenConfig, ImagenConfig
from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm
import json

def exists(val):
    return val is not None

def simple_slugify(text: str, max_length = 255):
    return text.replace('-', '_').replace(',', '').replace(' ', '_').replace('|', '--').strip('-_./\\')[:max_length]

def main():
    pass

@click.group()
def imagen():
    pass

@imagen.command(help = 'Sample from the Imagen model checkpoint')
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

    pil_image = imagen.sample([text], cond_scale = cond_scale, return_pil_images = True)

    image_path = f'./{simple_slugify(text)}.png'
    pil_image[0].save(image_path)

    print(f'image saved to {str(image_path)}')
    return

@imagen.command(help = 'Generate a config for the Imagen model')
@click.option('--path', default = './imagen_config.json', help = 'Path to the Imagen model config')
def config(
    path
):
    data = pkgutil.get_data(__name__, 'default_config.json').decode("utf-8") 
    with open(path, 'w') as f:
        f.write(data)

@imagen.command(help = 'Train the Imagen model')
@click.option('--config', default = './imagen_config.json', help = 'Path to the Imagen model config')
@click.option('--unet', default = 1, help = 'Unet to train', type = click.IntRange(1, 3, False, True, True))
@click.option('--epoches', default = 50, help = 'Amount of epoches to train for')
def train(
    config,
    unet,
    epoches,
):
    # check config path

    config_path = Path(config)
    full_config_path = str(config_path.resolve())
    assert config_path.exists(), f'config not found at {full_config_path}'
    
    with open(config_path, 'r') as f:
        config_data = json.loads(f.read())

    assert 'checkpoint_path' in config_data, 'checkpoint path not found in config'
    
    model_path = Path(config_data['checkpoint_path'])
    full_model_path = str(model_path.resolve())
    
    # setup imagen config

    imagen_config_klass = ElucidatedImagenConfig if config_data['type'] == 'elucidated' else ImagenConfig
    imagen = imagen_config_klass(**config_data['imagen']).create()

    trainer = ImagenTrainer(
    imagen = imagen,
        **config_data['trainer']
    )

    # load pt
    if model_path.exists():
        loaded = torch.load(str(model_path))
        version = safeget(loaded, 'version')
        print(f'loading Imagen from {full_model_path}, saved at version {version} - current package version is {__version__}')
        trainer.load(model_path)
        
    if torch.cuda.is_available():
        trainer = trainer.cuda()

    size = config_data['imagen']['image_sizes'][unet-1]

    max_batch_size = config_data['max_batch_size'] if 'max_batch_size' in config_data else 1

    channels = 'RGB'
    if 'channels' in config_data['imagen']:
        assert config_data['imagen']['channels'] > 0 and config_data['imagen']['channels'] < 5, 'Imagen only support 1 to 4 channels L, LA, RGB, RGBA'
        if config_data['imagen']['channels'] == 4:
            channels = 'RGBA' # Color with alpha
        elif config_data['imagen']['channels'] == 2:
            channels == 'LA' # Luminance (Greyscale) with alpha
        elif config_data['imagen']['channels'] == 1:
            channels = 'L' # Luminance (Greyscale)


    assert 'batch_size' in config_data['dataset'], 'A batch_size is required in the config file'
    
    # load and add train dataset and valid dataset
    ds = load_dataset(config_data['dataset_name'])
    
    train_ds = None
    
    # if we have train and valid split we combine them into one dataset to let trainer handle the split
    if 'train' in ds and 'valid' in ds:
        train_ds = concatenate_datasets([ds['train'], ds['valid']])
    elif 'train' in ds:
        train_ds = ds['train']
    elif 'valid' in ds:
        train_ds = ds['valid']
    else:
        train_ds = ds
        
    assert train_ds is not None, 'No train dataset could be fetched from the dataset name provided'
    
    trainer.add_train_dataset(
        ds = train_ds,
        collate_fn = Collator(
            image_size = size,
            image_label = config_data['image_label'],
            text_label = config_data['text_label'],
            url_label = config_data['url_label'],
            name = imagen.text_encoder_name,
            channels = channels
        ),
        **config_data['dataset']
    )
    
    should_validate = trainer.split_valid_from_train and 'validate_at_every' in config_data
    should_sample = 'sample_texts' in config_data and 'sample_at_every' in config_data
    should_save = 'save_at_every' in config_data
    
    valid_at_every = config_data['validate_at_every'] if should_validate else 0
    assert isinstance(valid_at_every, int), 'validate_at_every must be an integer'
    sample_at_every = config_data['sample_at_every'] if should_sample else 0
    assert isinstance(sample_at_every, int), 'sample_at_every must be an integer'
    save_at_every = config_data['save_at_every'] if should_save else 0
    assert isinstance(save_at_every, int), 'save_at_every must be an integer'
    sample_texts = config_data['sample_texts'] if should_sample else []
    assert isinstance(sample_texts, list), 'sample_texts must be a list'
    
    # check if when should_sample is true, sample_texts is not empty
    assert not should_sample or len(sample_texts) > 0, 'sample_texts must not be empty when sample_at_every is set'
    
    for i in range(epoches):
        for _ in tqdm(range(len(trainer.train_dl))):
            loss = trainer.train_step(unet_number = unet, max_batch_size = max_batch_size)
            print(f'loss: {loss}')

        if not (i % valid_at_every) and i > 0 and trainer.is_main and should_validate:
            valid_loss = trainer.valid_step(unet_number = unet, max_batch_size = max_batch_size)
            print(f'valid loss: {valid_loss}')

        if not (i % save_at_every) and i > 0 and trainer.is_main and should_sample:
            images = trainer.sample(texts = [sample_texts], batch_size = 1, return_pil_images = True, stop_at_unet_number = unet)
            images[0].save(f'./sample-{i // 100}.png')
            
        if not (i % save_at_every) and i > 0 and trainer.is_main and should_save:
            trainer.save(model_path)

    trainer.save(model_path)
