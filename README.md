<img src="./imagen.png" width="450px"></img>

## Imagen - Pytorch (wip)

Implementation of <a href="https://gweb-research-imagen.appspot.com/">Imagen</a>, Google's Text-to-Image Neural Network that beats DALL-E2, in Pytorch. It is the new SOTA for text-to-image synthesis.

Architecturally, it is actually much simpler than DALL-E2. It composes of a cascading DDPM conditioned on text embeddings from a large pretrained T5 model (attention network). It also contains dynamic clipping for improved classifier free guidance, noise level conditioning, and a memory efficient unet design.

It appears neither CLIP nor prior network is needed after all. And so research continues.

## Install

```bash
$ pip install imagen-pytorch
```

## Usage

```python
import torch
from imagen_pytorch import Unet, Imagen

# unet for imagen

unet1 = Unet(
    dim = 32,
    cond_dim = 128,
    channels = 3,
    dim_mults=(1, 2, 4, 8)
).cuda()

unet2 = Unet(
    dim = 32,
    cond_dim = 128,
    channels = 3,
    dim_mults=(1, 2, 4, 8)
).cuda()

# imagen, which contains the unets above (base unet and super resoluting ones)

imagen = Imagen(
    unets = (unet1, unet2),
    image_sizes = (64, 256),
    timesteps = 100,
    cond_drop_prob = 0.5
).cuda()

# mock images (get a lot of this) and text encodings from large T5

text_embeds = torch.randn(4, 256, 512).cuda()
images = torch.randn(4, 3, 256, 256).cuda()

# feed images into imagen, training each unet in the cascade

for i in (1, 2):
    loss = imagen(images, text_embeds = text_embeds, unet_number = i)
    loss.backward()

# do the above for many many many many steps
# now you can sample an image based on the text embeddings from the cascading ddpm

images = imagen.sample(texts = [
    'a whale breaching from afar',
    'young girl blowing out candles on her birthday cake',
    'fireworks with blue and green sparkles'
], cond_scale = 2.)

images.shape # (3, 3, 256, 256)
```

With the `ImagenTrainer` wrapper class, the exponential moving averages for all of the U-nets in the cascading DDPM will be automatically taken care of when calling `update`

```python
import torch
from imagen_pytorch import Unet, Imagen, ImagenTrainer

# unet for imagen

unet1 = Unet(
    dim = 32,
    cond_dim = 512,
    channels = 3,
    dim_mults=(1, 2, 4, 8)
).cuda()

unet2 = Unet(
    dim = 32,
    cond_dim = 512,
    channels = 3,
    dim_mults=(1, 2, 4, 8)
).cuda()

# imagen, which contains the unets above (base unet and super resoluting ones)

imagen = Imagen(
    unets = (unet1, unet2),
    text_encoder_name = 't5-large',
    image_sizes = (64, 256),
    timesteps = 100,
    cond_drop_prob = 0.5
).cuda()

# wrap imagen with the trainer class

trainer = ImagenTrainer(imagen)

# mock images (get a lot of this) and text encodings from large T5

text_embeds = torch.randn(4, 256, 1024).cuda()
images = torch.randn(4, 3, 256, 256).cuda()

# feed images into imagen, training each unet in the cascade

for i in (1, 2):
    loss = trainer(images, text_embeds = text_embeds, unet_number = i)
    trainer.update(unet_number = i)

# do the above for many many many many steps
# now you can sample an image based on the text embeddings from the cascading ddpm

images = trainer.sample(texts = [
    'a puppy looking anxiously at a giant donut on the table',
    'the milky way galaxy in the style of monet'
], cond_scale = 2.)

images.shape # (3, 3, 256, 256)
```

## Todo

- [x] use huggingface transformers for T5-small text embeddings
- [x] add dynamic thresholding
- [x] add dynamic thresholding DALLE2 and video-diffusion repository as well
- [x] allow for one to set T5-large (and perhaps small factory method to take in any huggingface transformer)
- [x] add the lowres noise level with the pseudocode in appendix, and figure out what is this sweep they do at inference time
- [x] port over some training code from DALLE2
- [ ] separate unet into base unet and SR3 unet
- [ ] build whatever efficient unet they came up with
- [ ] figure out if learned variance was used at all, and remove it if it was inconsequential
- [ ] switch to continuous timesteps instead of discretized, as it seems that is what they used for all stages
- [ ] need to be able to use a different noise schedule per unet (cosine was used for base, but linear for SR)

## Citations

```bibtex
@inproceedings{Saharia2022PhotorealisticTD,
    title   = {Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding},
    author  = {Chitwan Saharia and William Chan and Saurabh Saxena and Lala Li and Jay Whang and Emily L. Denton and Seyed Kamyar Seyed Ghasemipour and Burcu Karagol Ayan and Seyedeh Sara Mahdavi and Raphael Gontijo Lopes and Tim Salimans and Jonathan Ho and David Fleet and Mohammad Norouzi},
    year    = {2022}
}
```
