from pydantic import BaseModel, model_validator
from typing import List, Optional, Union, Tuple
from enum import Enum

from imagen_pytorch.imagen_pytorch import Imagen, Unet, Unet3D, NullUnet
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

class NullUnetConfig(BaseModel):
    is_null:            bool

    def create(self):
        return NullUnet()

class UnetConfig(AllowExtraBaseModel):
    dim:                int
    dim_mults:          ListOrTuple(int)
    text_embed_dim:     int = get_encoded_dim(DEFAULT_T5_NAME)
    cond_dim:           Optional[int] = None
    channels:           int = 3
    attn_dim_head:      int = 32
    attn_heads:         int = 16

    def create(self):
        return Unet(**self.dict())

class Unet3DConfig(AllowExtraBaseModel):
    dim:                int
    dim_mults:          ListOrTuple(int)
    text_embed_dim:     int = get_encoded_dim(DEFAULT_T5_NAME)
    cond_dim:           Optional[int] = None
    channels:           int = 3
    attn_dim_head:      int = 32
    attn_heads:         int = 16

    def create(self):
        return Unet3D(**self.dict())

class ImagenConfig(AllowExtraBaseModel):
    unets:                  ListOrTuple(Union[UnetConfig, Unet3DConfig, NullUnetConfig])
    image_sizes:            ListOrTuple(int)
    video:                  bool = False
    timesteps:              SingleOrList(int) = 1000
    noise_schedules:        SingleOrList(NoiseSchedule) = 'cosine'
    text_encoder_name:      str = DEFAULT_T5_NAME
    channels:               int = 3
    loss_type:              str = 'l2'
    cond_drop_prob:         float = 0.5

    @model_validator(mode="after")
    def check_image_sizes(self):
        if len(self.image_sizes) != len(self.unets):
            raise ValueError(f'image sizes length {len(self.image_sizes)} must be equivalent to the number of unets {len(self.unets)}')
        return self

    def create(self):
        decoder_kwargs = self.dict()
        unets_kwargs = decoder_kwargs.pop('unets')
        is_video = decoder_kwargs.pop('video', False)

        unets = []

        for unet, unet_kwargs in zip(self.unets, unets_kwargs):
            if isinstance(unet, NullUnetConfig):
                unet_klass = NullUnet
            elif is_video:
                unet_klass = Unet3D
            else:
                unet_klass = Unet

            unets.append(unet_klass(**unet_kwargs))

        imagen = Imagen(unets, **decoder_kwargs)

        imagen._config = self.dict().copy()
        return imagen

class ElucidatedImagenConfig(AllowExtraBaseModel):
    unets:                  ListOrTuple(Union[UnetConfig, Unet3DConfig, NullUnetConfig])
    image_sizes:            ListOrTuple(int)
    video:                  bool = False
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

    @model_validator(mode="after")
    def check_image_sizes(self):
        if len(self.image_sizes) != len(self.unets):
            raise ValueError(f'image sizes length {len(self.image_sizes)} must be equivalent to the number of unets {len(self.unets)}')
        return self

    def create(self):
        decoder_kwargs = self.dict()
        unets_kwargs = decoder_kwargs.pop('unets')
        is_video = decoder_kwargs.pop('video', False)

        unet_klass = Unet3D if is_video else Unet

        unets = []

        for unet, unet_kwargs in zip(self.unets, unets_kwargs):
            if isinstance(unet, NullUnetConfig):
                unet_klass = NullUnet
            elif is_video:
                unet_klass = Unet3D
            else:
                unet_klass = Unet

            unets.append(unet_klass(**unet_kwargs))

        imagen = ElucidatedImagen(unets, **decoder_kwargs)

        imagen._config = self.dict().copy()
        return imagen

class ImagenTrainerConfig(AllowExtraBaseModel):
    imagen:                 dict
    elucidated:             bool = False
    video:                  bool = False
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
        imagen = imagen_config_klass(**{**imagen_config, 'video': video}).create()

        return ImagenTrainer(imagen, **trainer_kwargs)
