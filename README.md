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
    unet = (unet1, unet2),
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
])

images.shape # (3, 3, 256, 256)
```

## Todo

- [x] use huggingface transformers for T5-small text embeddings
- [x] add dynamic thresholding
- [ ] add dynamic thresholding DALLE2 and video-diffusion repository as well
- [ ] allow for one to set T5-large (and perhaps small factory method to take in any huggingface transformer)
- [ ] separate unet into base unet and SR3 unet
- [ ] build whatever efficient unet they came up with
- [ ] add the noise level conditioning with the pseudocode in appendix, and figure out what is this sweep they do at inference time
- [ ] port over some training code from DALLE2
- [ ] figure out if learned variance was used at all, and remove it if it was inconsequential

## Citations

```bibtex
@misc{Saharia2022,
    title   = {Imagen: unprecedented photorealism × deep level of language understanding}, 
    author  = {Chitwan Saharia*, William Chan*, Saurabh Saxena†, Lala Li†, Jay Whang†, Emily Denton, Seyed Kamyar Seyed Ghasemipour, Burcu Karagol Ayan, S. Sara Mahdavi, Rapha Gontijo Lopes, Tim Salimans, Jonathan Ho†, David Fleet†, Mohammad Norouzi*},
    year    = {2022}
}
```
