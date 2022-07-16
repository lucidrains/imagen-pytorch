import json
from pydantic import BaseModel, validator, root_validator
from typing import List, Iterable, Optional, Union, Tuple, Dict, Any
from enum import Enum

from imagen_pytorch.imagen_pytorch import Imagen, Unet
from imagen_pytorch.trainer import ImagenTrainer
from imagen_pytorch.elucidated_imagen import ElucidatedImagen

from imagen_pytorch.t5 import DEFAULT_T5_NAME, get_encoded_dim

# helper functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def ListOrTuple(inner_type):
    return Union[List[inner_type], Tuple[inner_type]]

def SingleOrList(inner_type):
    return Union[inner_type, ListOrTuple(inner_type)]

# noise schedule

class NoiseSchedule(Enum):
    cosine = 'cosine'
    linear = 'linear'

class AllowExtraBaseModel(BaseModel):
    class Config:
        extra = "allow"
        use_enum_values = True

# imagen pydantic classes

class UnetConfig(AllowExtraBaseModel):
    dim:                int
    dim_mults:          ListOrTuple(int)
    text_embed_dim:     int = get_encoded_dim(DEFAULT_T5_NAME)
    cond_dim:           int = None
    channels:           int = 3
    attn_dim_head:      int = 32
    attn_heads:         int = 16

    def create(self):
        return Unet(**self.dict())

class ImagenConfig(AllowExtraBaseModel):
    unets:                  ListOrTuple(UnetConfig)
    image_sizes:            ListOrTuple(int)
    timesteps:              SingleOrList(int) = 1000
    noise_schedules:        SingleOrList(NoiseSchedule) = 'cosine'
    text_encoder_name:      str = DEFAULT_T5_NAME
    channels:               int = 3
    loss_type:              str = 'l2'
    cond_drop_prob:         float = 0.5

    @validator('image_sizes')
    def check_image_sizes(cls, image_sizes, values):
        unets = values.get('unets')
        if len(image_sizes) != len(unets):
            raise ValueError(f'image sizes length {len(image_sizes)} must be equivalent to the number of unets {len(unets)}')
        return image_sizes

    def create(self):
        decoder_kwargs = self.dict()
        decoder_kwargs.pop('unets')

        unets = [unet.create() for unet in self.unets]
        imagen = Imagen(unets, **decoder_kwargs)

        imagen._config = self.dict().copy()
        return imagen

class ElucidatedImagenConfig(AllowExtraBaseModel):
    unets:                  ListOrTuple(UnetConfig)
    image_sizes:            ListOrTuple(int)
    text_encoder_name:      str = DEFAULT_T5_NAME
    channels:               int = 3
    cond_drop_prob:         float = 0.5
    num_sample_steps:       SingleOrList(int) = 32
    sigma_min:              SingleOrList(float) = 0.002
    sigma_max:              SingleOrList(int) = 80
    sigma_data:             SingleOrList(float) = 0.5
    rho:                    SingleOrList(int) = 7
    P_mean:                 SingleOrList(float) = -1.2
    P_std:                  SingleOrList(float) = 1.2
    S_churn:                SingleOrList(int) = 80
    S_tmin:                 SingleOrList(float) = 0.05
    S_tmax:                 SingleOrList(int) = 50
    S_noise:                SingleOrList(float) = 1.003

    @validator('image_sizes')
    def check_image_sizes(cls, image_sizes, values):
        unets = values.get('unets')
        if len(image_sizes) != len(unets):
            raise ValueError(f'image sizes length {len(image_sizes)} must be equivalent to the number of unets {len(unets)}')
        return image_sizes

    def create(self):
        decoder_kwargs = self.dict()
        decoder_kwargs.pop('unets')

        unets = [unet.create() for unet in self.unets]
        imagen = ElucidatedImagen(unets, **decoder_kwargs)

        imagen._config = self.dict().copy()
        return imagen

class ImagenTrainerConfig(AllowExtraBaseModel):
    imagen:                 dict
    elucidated:             bool = False
    use_ema:                bool = True
    lr:                     SingleOrList(float) = 1e-4
    eps:                    SingleOrList(float) = 1e-8
    beta1:                  float = 0.9
    beta2:                  float = 0.99
    max_grad_norm:          Optional[float] = None
    group_wd_params:        bool = True
    warmup_steps:           SingleOrList(Optional[int]) = None
    cosine_decay_max_steps: SingleOrList(Optional[int]) = None

    def create(self):
        trainer_kwargs = self.dict()

        imagen_config = trainer_kwargs.pop('imagen')
        elucidated = trainer_kwargs.pop('elucidated')

        imagen_config_klass = ElucidatedImagenConfig if elucidated else ImagenConfig
        imagen = imagen_config_klass(**imagen_config).create()

        return ImagenTrainer(imagen, **trainer_kwargs)
