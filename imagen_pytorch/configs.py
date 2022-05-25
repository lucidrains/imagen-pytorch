import json
from pydantic import BaseModel, validator, root_validator
from typing import List, Iterable, Optional, Union, Tuple, Dict, Any
from imagen_pytorch.imagen_pytorch import Imagen, Unet

# helper functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def ListOrTuple(inner_type):
    return Union[List[inner_type], Tuple[inner_type]]

# imagen pydantic classes

class UnetConfig(BaseModel):
    dim: int
    dim_mults: ListOrTuple(int)
    text_embed_dim: int = 1024
    cond_dim: int = None
    channels: int = 3
    attn_dim_head: int = 32
    attn_heads: int = 16

    class Config:
        extra = "allow"

class ImagenConfig(BaseModel):
    unets: ListOrTuple(UnetConfig)
    image_sizes: ListOrTuple(int)
    channels: int = 3
    timesteps: int = 1000
    loss_type: str = 'l2'
    beta_schedule: str = 'cosine'
    learned_variance: bool = True
    cond_drop_prob: float = 0.5

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
