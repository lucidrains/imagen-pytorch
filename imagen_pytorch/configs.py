import json
from pydantic import BaseModel, validator, root_validator
from typing import List, Iterable, Optional, Union, Tuple, Dict, Any
from enum import Enum

from imagen_pytorch.imagen_pytorch import Imagen, Unet
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

# imagen pydantic classes

class UnetConfig(AllowExtraBaseModel):
    dim: int
    dim_mults: ListOrTuple(int)
    text_embed_dim: int = get_encoded_dim(DEFAULT_T5_NAME)
    cond_dim: int = None
    channels: int = 3
    attn_dim_head: int = 32
    attn_heads: int = 16

class ImagenConfig(AllowExtraBaseModel):
    unets: ListOrTuple(UnetConfig)
    image_sizes: ListOrTuple(int)
    timesteps: SingleOrList(int) = 1000
    noise_schedules: SingleOrList(NoiseSchedule) = 'cosine'
    warmup_steps: SingleOrList(int) = None
    cosine_decay_max_steps: SingleOrList(int) = None
    text_encoder_name: str = DEFAULT_T5_NAME
    channels: int = 3
    loss_type: str = 'l2'
    learned_variance: bool = True
    cond_drop_prob: float = 0.5,
    accelerate: bool = False

    @validator('image_sizes')
    def check_image_sizes(cls, image_sizes, values):
        unets = values.get('unets')
        if len(image_sizes) != len(unets):
            raise ValueError(f'image sizes length {len(image_sizes)} must be equivalent to the number of unets {len(unets)}')
        return image_sizes

    def create(self):
        decoder_kwargs = self.dict()
        unet_configs = decoder_kwargs.pop('unets')
        unets = [Unet(**config) for config in unet_configs]
        return Imagen(unets, **decoder_kwargs)

    class Config:
        extra = "allow"
