<img src="./imagen.png" width="450px"></img>

## Imagen - Pytorch

Implementation of <a href="https://gweb-research-imagen.appspot.com/">Imagen</a>, Google's Text-to-Image Neural Network that beats DALL-E2, in Pytorch. It is the new SOTA for text-to-image synthesis.

Architecturally, it is actually much simpler than DALL-E2. It consists of a cascading DDPM conditioned on text embeddings from a large pretrained T5 model (attention network). It also contains dynamic clipping for improved classifier free guidance, noise level conditioning, and a memory efficient unet design.

It appears neither CLIP nor prior network is needed after all. And so research continues.

<a href="https://www.youtube.com/watch?v=xqDeAz0U-R4">AI Coffee Break with Letitia</a> | <a href="https://www.assemblyai.com/blog/how-imagen-actually-works/">Assembly AI</a>

Please join <a href="https://discord.gg/xBPBXfcFHd"><img alt="Join us on Discord" src="https://img.shields.io/discord/823813159592001537?color=5865F2&logo=discord&logoColor=white"></a> if you are interested in helping out with the replication with the <a href="https://laion.ai/">LAION</a> community

## Shoutouts

- <a href="https://stability.ai/">StabilityAI</a> for the generous sponsorship, as well as my other sponsors out there

- <a href="https://huggingface.co/">🤗 Huggingface</a> for their amazing transformers library. The text encoder portion is pretty much taken care of because of them

- <a href="https://github.com/sgugger">Sylvain</a> and <a href="https://github.com/muellerzr">Zachary</a> for the <a href="https://github.com/huggingface/accelerate">Accelerate</a> library, which this repository uses for distributed training

- <a href="https://github.com/jorgemcgomes">Jorge Gomes</a> for helping out with the T5 loading code and advice on the correct T5 version

- <a href="https://github.com/crowsonkb">Katherine Crowson</a>, for her <a href="https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/utils.py">beautiful code</a>, which helped me understand the continuous time version of gaussian diffusion

- <a href="https://github.com/marunine">Marunine</a> and <a href="https://github.com/Netruk44">Netruk44</a>, for reviewing code, sharing experimental results, and help with debugging

- <a href="https://github.com/marunine">Marunine</a> for providing a <a href="https://github.com/lucidrains/imagen-pytorch/issues/72#issuecomment-1163275757">potential solution</a> for a color shifting issue in the memory efficient u-nets. Thanks to <a href="https://github.com/jacobwjs">Jacob</a> for sharing experimental comparisons between the base and memory-efficient unets

- <a href="https://github.com/marunine">Marunine</a> for finding numerous bugs, resolving an issue with resize right, and for sharing his experimental configurations and results

- <a href="https://github.com/MalumaDev">MalumaDev</a> for proposing the use of pixel shuffle upsampler to fix checkboard artifacts

- <a href="https://github.com/KhrulkovV">Valentin</a> for pointing out insufficient skip connections in the unet, as well as the specific method of attention conditioning in the base-unet in the appendix

- <a href="https://github.com/BIGJUN777">BIGJUN</a> for catching a big bug with continuous time gaussian diffusion noise level conditioning at inference time

- <a href="https://github.com/animebing">Bingbing</a> for identifying a bug with sampling and order of normalizing and noising with low resolution conditioning image

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
    cond_dim = 512,
    dim_mults = (1, 2, 4, 8),
    num_resnet_blocks = 3,
    layer_attns = (False, True, True, True),
    layer_cross_attns = (False, True, True, True)
)

unet2 = Unet(
    dim = 32,
    cond_dim = 512,
    dim_mults = (1, 2, 4, 8),
    num_resnet_blocks = (2, 4, 8, 8),
    layer_attns = (False, False, False, True),
    layer_cross_attns = (False, False, False, True)
)

# imagen, which contains the unets above (base unet and super resoluting ones)

imagen = Imagen(
    unets = (unet1, unet2),
    image_sizes = (64, 256),
    timesteps = 1000,
    cond_drop_prob = 0.1
).cuda()

# mock images (get a lot of this) and text encodings from large T5

text_embeds = torch.randn(4, 256, 768).cuda()
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
], cond_scale = 3.)

images.shape # (3, 3, 256, 256)
```

For simpler training, you can directly supply text strings instead of precomputing text encodings. (Although for scaling purposes, you will definitely want to precompute the textual embeddings + mask)

The number of textual captions must match the batch size of the images if you go this route.

```python
# mock images and text (get a lot of this)

texts = [
    'a child screaming at finding a worm within a half-eaten apple',
    'lizard running across the desert on two feet',
    'waking up to a psychedelic landscape',
    'seashells sparkling in the shallow waters'
]

images = torch.randn(4, 3, 256, 256).cuda()

# feed images into imagen, training each unet in the cascade

for i in (1, 2):
    loss = imagen(images, texts = texts, unet_number = i)
    loss.backward()
```

With the `ImagenTrainer` wrapper class, the exponential moving averages for all of the U-nets in the cascading DDPM will be automatically taken care of when calling `update`

```python
import torch
from imagen_pytorch import Unet, Imagen, ImagenTrainer

# unet for imagen

unet1 = Unet(
    dim = 32,
    cond_dim = 512,
    dim_mults = (1, 2, 4, 8),
    num_resnet_blocks = 3,
    layer_attns = (False, True, True, True),
)

unet2 = Unet(
    dim = 32,
    cond_dim = 512,
    dim_mults = (1, 2, 4, 8),
    num_resnet_blocks = (2, 4, 8, 8),
    layer_attns = (False, False, False, True),
    layer_cross_attns = (False, False, False, True)
)

# imagen, which contains the unets above (base unet and super resoluting ones)

imagen = Imagen(
    unets = (unet1, unet2),
    text_encoder_name = 't5-large',
    image_sizes = (64, 256),
    timesteps = 1000,
    cond_drop_prob = 0.1
).cuda()

# wrap imagen with the trainer class

trainer = ImagenTrainer(imagen)

# mock images (get a lot of this) and text encodings from large T5

text_embeds = torch.randn(64, 256, 1024).cuda()
images = torch.randn(64, 3, 256, 256).cuda()

# feed images into imagen, training each unet in the cascade

loss = trainer(
    images,
    text_embeds = text_embeds,
    unet_number = 1,            # training on unet number 1 in this example, but you will have to also save checkpoints and then reload and continue training on unet number 2
    max_batch_size = 4          # auto divide the batch of 64 up into batch size of 4 and accumulate gradients, so it all fits in memory
)

trainer.update(unet_number = 1)

# do the above for many many many many steps
# now you can sample an image based on the text embeddings from the cascading ddpm

images = trainer.sample(texts = [
    'a puppy looking anxiously at a giant donut on the table',
    'the milky way galaxy in the style of monet'
], cond_scale = 3.)

images.shape # (2, 3, 256, 256)
```

You can also train Imagen without text (unconditional image generation) as follows

```python
import torch
from imagen_pytorch import Unet, Imagen, SRUnet256, ImagenTrainer

# unets for unconditional imagen

unet1 = Unet(
    dim = 32,
    dim_mults = (1, 2, 4),
    num_resnet_blocks = 3,
    layer_attns = (False, True, True),
    layer_cross_attns = False,
    use_linear_attn = True
)

unet2 = SRUnet256(
    dim = 32,
    dim_mults = (1, 2, 4),
    num_resnet_blocks = (2, 4, 8),
    layer_attns = (False, False, True),
    layer_cross_attns = False
)

# imagen, which contains the unets above (base unet and super resoluting ones)

imagen = Imagen(
    condition_on_text = False,   # this must be set to False for unconditional Imagen
    unets = (unet1, unet2),
    image_sizes = (64, 128),
    timesteps = 1000
)

trainer = ImagenTrainer(imagen).cuda()

# now get a ton of images and feed it through the Imagen trainer

training_images = torch.randn(4, 3, 256, 256).cuda()

# train each unet separately
# in this example, only training on unet number 1

loss = trainer(training_images, unet_number = 1)
trainer.update(unet_number = 1)

# do the above for many many many many steps
# now you can sample images unconditionally from the cascading unet(s)

images = trainer.sample(batch_size = 16) # (16, 3, 128, 128)
```

At any time you can save and load the trainer and all associated states with the `save` and `load` methods. It is recommended you use these methods instead of manually saving with a `state_dict` call, as there are some device memory management being done underneath the hood within the trainer.

ex.

```python
trainer.save('./path/to/checkpoint.pt')

trainer.load('./path/to/checkpoint.pt')

trainer.steps # (2,) step number for each of the unets, in this case 2
```

## Dataloader

You can also rely on the `ImagenTrainer` to automatically train off `DataLoader` instances. You simply have to craft your `DataLoader` to return either `images` (for unconditional case), or of `('images', 'text_embeds')` for text-guided generation.

ex. unconditional training

```python
from imagen_pytorch import Unet, Imagen, ImagenTrainer
from imagen_pytorch.data import Dataset

# unets for unconditional imagen

unet = Unet(
    dim = 32,
    dim_mults = (1, 2, 4, 8),
    num_resnet_blocks = 1,
    layer_attns = (False, False, False, True),
    layer_cross_attns = False
)

# imagen, which contains the unet above

imagen = Imagen(
    condition_on_text = False,  # this must be set to False for unconditional Imagen
    unets = unet,
    image_sizes = 128,
    timesteps = 1000
)

trainer = ImagenTrainer(
    imagen = imagen,
    split_valid_from_train = True # whether to split the validation dataset from the training
).cuda()

# instantiate your dataloader, which returns the necessary inputs to the DDPM as tuple in the order of images, text embeddings, then text masks. in this case, only images is returned as it is unconditional training

dataset = Dataset('/path/to/training/images', image_size = 128)

trainer.add_train_dataset(dataset, batch_size = 16)

# working training loop

for i in range(200000):
    loss = trainer.train_step(unet_number = 1, max_batch_size = 4)
    print(f'loss: {loss}')

    if not (i % 50):
        valid_loss = trainer.valid_step(unet_number = 1, max_batch_size = 4)
        print(f'valid loss: {valid_loss}')

    if not (i % 100) and trainer.is_main: # is_main makes sure this can run in distributed
        images = trainer.sample(batch_size = 1, return_pil_images = True) # returns List[Image]
        images[0].save(f'./sample-{i // 100}.png')

```

## Multi GPU

Thanks to <a href="https://huggingface.co/docs/accelerate/index">🤗 Accelerate</a>, you can do multi GPU training easily with two steps.

First you need to invoke `accelerate config` in the same directory as your training script (say it is named `train.py`)

```bash
$ accelerate config
```

Next, instead of calling `python train.py` as you would for single GPU, you would use the accelerate CLI as so

```bash
$ accelerate launch train.py
```

That's it!

## Command-line

To further democratize the use of this machine imagination, I have built in the ability to generate an image with any text prompt using one command line as so

ex.

```bash
$ imagen --model ./path/to/model/checkpoint.pt "a squirrel raiding the birdfeeder"
# image is saved to ./a_squirrel_raiding_the_birdfeeder.png
```

In order to save checkpoints that can make use of this feature, you must instantiate your Imagen instance using the config classes, `ImagenConfig` and `ElucidatedImagenConfig`

For proper training, you'll likely want to setup config-driven training anyways.

ex.

```python
import torch
from imagen_pytorch import ImagenConfig, ElucidatedImagenConfig, ImagenTrainer

# in this example, using elucidated imagen

imagen = ElucidatedImagenConfig(
    unets = [
        dict(dim = 32, dim_mults = (1, 2, 4, 8)),
        dict(dim = 32, dim_mults = (1, 2, 4, 8))
    ],
    image_sizes = (64, 128),
    cond_drop_prob = 0.5,
    num_sample_steps = 32
).create()

trainer = ImagenTrainer(imagen)

# do your training ...

# then save it

trainer.save('./checkpoint.pt')

# you should see a message informing you that ./checkpoint.pt is commandable from the terminal
```

It really should be as simple as that

You can also pass this checkpoint file around, and anyone can continue finetune on their own data

```python
from imagen_pytorch import load_imagen_from_checkpoint, ImagenTrainer

imagen = load_imagen_from_checkpoint('./checkpoint.pt')

trainer = ImagenTrainer(imagen)

# continue training / fine-tuning
```

## Inpainting

Inpainting follows the formulation laid out by the recent <a href="https://arxiv.org/abs/2201.09865">Repaint paper</a>. Simply pass in `inpaint_images` and `inpaint_masks` to the `sample` function on either `Imagen` or `ElucidatedImagen`

```python

inpaint_images = torch.randn(4, 3, 512, 512).cuda()      # (batch, channels, height, width)
inpaint_masks = torch.ones((4, 512, 512)).bool().cuda()  # (batch, height, width)

inpainted_images = trainer.sample(texts = [
    'a whale breaching from afar',
    'young girl blowing out candles on her birthday cake',
    'fireworks with blue and green sparkles',
    'fireworks with blue and green sparkles'
], inpaint_images = inpaint_images, inpaint_masks = inpaint_masks, cond_scale = 5.)

inpainted_images # (4, 3, 512, 512)
```

## Experimental

<a href="https://research.nvidia.com/person/tero-karras">Tero Karras</a> of StyleGAN fame has written a <a href="https://arxiv.org/abs/2206.00364">new paper</a> with results that have been corroborated by a number of independent researchers as well as on my own machine. I have decided to create a version of `Imagen`, the `ElucidatedImagen`, so that one can use the new elucidated DDPM for text-guided cascading generation.

Simply import `ElucidatedImagen`, and then instantiate the instance as you did before. The hyperparameters are different than the usual ones for discrete and continuous time gaussian diffusion, and can be individualized for each unet in the cascade.

Ex.

```python
from imagen_pytorch import ElucidatedImagen

# instantiate your unets ...

imagen = ElucidatedImagen(
    unets = (unet1, unet2),
    image_sizes = (64, 128),
    cond_drop_prob = 0.1,
    num_sample_steps = (64, 32), # number of sample steps - 64 for base unet, 32 for upsampler (just an example, have no clue what the optimal values are)
    sigma_min = 0.002,           # min noise level
    sigma_max = (80, 160),       # max noise level, @crowsonkb recommends double the max noise level for upsampler
    sigma_data = 0.5,            # standard deviation of data distribution
    rho = 7,                     # controls the sampling schedule
    P_mean = -1.2,               # mean of log-normal distribution from which noise is drawn for training
    P_std = 1.2,                 # standard deviation of log-normal distribution from which noise is drawn for training
    S_churn = 80,                # parameters for stochastic sampling - depends on dataset, Table 5 in apper
    S_tmin = 0.05,
    S_tmax = 50,
    S_noise = 1.003,
).cuda()

# rest is the same as above

```

## FAQ

- Why are my generated images not aligning well with the text?

Imagen uses an algorithm called <a href="https://openreview.net/forum?id=qw8AKxfYbI">Classifier Free Guidance</a>. When sampling, you apply a scale to the conditioning (text in this case) of greater than `1.0`.

Researcher <a href="https://github.com/Netruk44 ">Netruk44</a> have reported `5-10` to be optimal, but anything greater than `10` to break.

```python
trainer.sample(texts = [
    'a cloud in the shape of a roman gladiator'
], cond_scale = 5.) # <-- cond_scale is the conditioning scale, needs to be greater than 1.0 to be better than average
```

- Are there any pretrained models yet?

Not at the moment but one will likely be trained and open sourced within the year, if not sooner. If you would like to participate, you can join the community of artificial neural network trainers at Laion (discord link is in the Readme above) and start collaborating.

- Will this technology take my job?

More the reason why you should start training your own model, starting today! The last thing we need is this technology being in the hands of an elite few. Hopefully this repository reduces the work to just finding the necessary compute, and augmenting with your own curated dataset.

## Related Works

- <a href="https://github.com/archinetai/audio-diffusion-pytorch">Audio diffusion</a> from <a href="https://github.com/flavioschneider">Flavio Schneider</a>

## Todo

- [x] use huggingface transformers for T5-small text embeddings
- [x] add dynamic thresholding
- [x] add dynamic thresholding DALLE2 and video-diffusion repository as well
- [x] allow for one to set T5-large (and perhaps small factory method to take in any huggingface transformer)
- [x] add the lowres noise level with the pseudocode in appendix, and figure out what is this sweep they do at inference time
- [x] port over some training code from DALLE2
- [x] need to be able to use a different noise schedule per unet (cosine was used for base, but linear for SR)
- [x] just make one master-configurable unet
- [x] complete resnet block (biggan inspired? but with groupnorm) - complete self attention
- [x] complete conditioning embedding block (and make it completely configurable, whether it be attention, film etc)
- [x] consider using perceiver-resampler from https://github.com/lucidrains/flamingo-pytorch in place of attention pooling
- [x] add attention pooling option, in addition to cross attention and film
- [x] add optional cosine decay schedule with warmup, for each unet, to trainer
- [x] switch to continuous timesteps instead of discretized, as it seems that is what they used for all stages - first figure out the linear noise schedule case from the variational ddpm paper https://openreview.net/forum?id=2LdBqxc1Yv
- [x] figure out log(snr) for alpha cosine noise schedule.
- [x] suppress the transformers warning because only T5encoder is used
- [x] allow setting for using linear attention on layers where full attention cannot be used
- [x] force unets in continuous time case to use non-fouriered conditions (just pass the log(snr) through an MLP with optional layernorms), as that is what i have working locally
- [x] removed learned variance
- [x] add p2 loss weighting for continuous time
- [x] make sure cascading ddpm can be trained without text condition, and make sure both continuous and discrete time gaussian diffusion works
- [x] use primer's depthwise convs on the qkv projections in linear attention (or use token shifting before projections) - also use new dropout proposed by bayesformer, as it seems to work well with linear attention
- [x] explore skip layer excitation in unet decoder
- [x] accelerate integration
- [x] build out CLI tool and one-line generation of image
- [x] knock out any issues that arised from accelerate
- [x] add inpainting ability using resampler from repaint paper https://arxiv.org/abs/2201.09865
- [x] build a simple checkpointing system, backed by a folder
- [x] add skip connection from outputs of all upsample blocks, used in unet squared paper and some previous unet works
- [x] add fsspec, recommended by Romain @rom1504, for cloud / local file system agnostic persistence of checkpoints
- [ ] test out persistence in gcs with https://github.com/fsspec/gcsfs
- [ ] build out CLI tool for training, resuming training off config file
- [ ] preencoding of text to memmapped embeddings
- [ ] extend to video generation, using axial time attention as in Ho's video ddpm paper + https://github.com/lucidrains/flexible-diffusion-modeling-videos-pytorch for up to 25 minute video
- [ ] be able to create dataloader iterators based on the old epoch style, also configure shuffling etc
- [ ] be able to also pass in arguments (instead of requiring forward to be all keyword args on model)

## Citations

```bibtex
@inproceedings{Saharia2022PhotorealisticTD,
    title   = {Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding},
    author  = {Chitwan Saharia and William Chan and Saurabh Saxena and Lala Li and Jay Whang and Emily L. Denton and Seyed Kamyar Seyed Ghasemipour and Burcu Karagol Ayan and Seyedeh Sara Mahdavi and Raphael Gontijo Lopes and Tim Salimans and Jonathan Ho and David Fleet and Mohammad Norouzi},
    year    = {2022}
}
```

```bibtex
@article{Alayrac2022Flamingo,
    title   = {Flamingo: a Visual Language Model for Few-Shot Learning},
    author  = {Jean-Baptiste Alayrac et al},
    year    = {2022}
}
```

```bibtex
@article{Choi2022PerceptionPT,
    title   = {Perception Prioritized Training of Diffusion Models},
    author  = {Jooyoung Choi and Jungbeom Lee and Chaehun Shin and Sungwon Kim and Hyunwoo J. Kim and Sung-Hoon Yoon},
    journal = {ArXiv},
    year    = {2022},
    volume  = {abs/2204.00227}
}
```

```bibtex
@inproceedings{Sankararaman2022BayesFormerTW,
    title   = {BayesFormer: Transformer with Uncertainty Estimation},
    author  = {Karthik Abinav Sankararaman and Sinong Wang and Han Fang},
    year    = {2022}
}
```

```bibtex
@article{So2021PrimerSF,
    title   = {Primer: Searching for Efficient Transformers for Language Modeling},
    author  = {David R. So and Wojciech Ma'nke and Hanxiao Liu and Zihang Dai and Noam M. Shazeer and Quoc V. Le},
    journal = {ArXiv},
    year    = {2021},
    volume  = {abs/2109.08668}
}
```

```bibtex
@misc{cao2020global,
    title   = {Global Context Networks},
    author  = {Yue Cao and Jiarui Xu and Stephen Lin and Fangyun Wei and Han Hu},
    year    = {2020},
    eprint  = {2012.13375},
    archivePrefix = {arXiv},
    primaryClass = {cs.CV}
}
```

```bibtex
@article{Karras2022ElucidatingTD,
    title   = {Elucidating the Design Space of Diffusion-Based Generative Models},
    author  = {Tero Karras and Miika Aittala and Timo Aila and Samuli Laine},
    journal = {ArXiv},
    year    = {2022},
    volume  = {abs/2206.00364}
}
```

```bibtex
@inproceedings{NEURIPS2020_4c5bcfec,
    author      = {Ho, Jonathan and Jain, Ajay and Abbeel, Pieter},
    booktitle   = {Advances in Neural Information Processing Systems},
    editor      = {H. Larochelle and M. Ranzato and R. Hadsell and M.F. Balcan and H. Lin},
    pages       = {6840--6851},
    publisher   = {Curran Associates, Inc.},
    title       = {Denoising Diffusion Probabilistic Models},
    url         = {https://proceedings.neurips.cc/paper/2020/file/4c5bcfec8584af0d967f1ab10179ca4b-Paper.pdf},
    volume      = {33},
    year        = {2020}
}
```

```bibtex
@article{Lugmayr2022RePaintIU,
    title   = {RePaint: Inpainting using Denoising Diffusion Probabilistic Models},
    author  = {Andreas Lugmayr and Martin Danelljan and Andr{\'e}s Romero and Fisher Yu and Radu Timofte and Luc Van Gool},
    journal = {ArXiv},
    year    = {2022},
    volume  = {abs/2201.09865}
}
```
