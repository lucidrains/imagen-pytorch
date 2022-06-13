import os
import torch
import torchvision.transforms as T
import matplotlib

from shortuuid import uuid
from matplotlib import pyplot
from imagen_pytorch import Unet, Imagen, ImagenTrainer

use_cpu = False
device = torch.device('cpu') if use_cpu else torch.device('cuda:0')

save_dir = "<MODEL_DIRECTORY>"
checkpoint_name = "checkpoint_latest.pth"
batch_size = 10

# unet for imagen

unet1 = Unet(
    dim = 32,
    cond_dim = 512,
    dim_mults = (1, 2, 4, 8),
    num_resnet_blocks = 3,
    layer_attns = (False, True, True, True),
).to(device)

unet2 = Unet(
    dim = 32,
    cond_dim = 512,
    dim_mults = (1, 2, 4, 8),
    num_resnet_blocks = (2, 4, 8, 8),
    layer_attns = (False, False, False, True),
    layer_cross_attns = (False, False, False, True)
).to(device)



# imagen, which contains the unets above (base unet and super resoluting ones)

imagen = Imagen(
    unets = (unet1, unet2),
    text_encoder_name = 't5-3b',
    image_sizes = (64, 256),
    beta_schedules = ('cosine', 'linear'),
    timesteps = 1000,
    cond_drop_prob = 0.5
).to(device)


# now you can sample an image based on the text embeddings from the cascading ddpm
trainer = ImagenTrainer(imagen)

# Load checkpoint

print("Loading model...")


try:
    _ = trainer.load(os.path.join(save_dir, checkpoint_name), only_model = False)
except:
    pass

images = trainer.sample(
    batch_size = batch_size,
    return_all_unet_outputs = False,
    return_pil_images = True,
    cond_scale = 2.)

for image in images:
    image.save(uuid()+".png")

