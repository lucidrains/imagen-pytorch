import torch
from functools import reduce
from pathlib import Path
from imagen_pytorch.configs import ImagenConfig, ElucidatedImagenConfig

def exists(val):
    return val is not None

def safeget(dictionary, keys, default = None):
    return reduce(lambda d, key: d.get(key, default) if isinstance(d, dict) else default, keys.split('.'), dictionary)

def load_imagen_from_checkpoint(checkpoint_path):
    model_path = Path(checkpoint_path)
    full_model_path = str(model_path.resolve())
    assert model_path.exists(), f'checkpoint not found at {full_model_path}'
    loaded = torch.load(str(model_path))

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
    return imagen
