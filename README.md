🖼️ Imagen - Pytorch
Text-to-Image Generation with Imagen (Google Research) - PyTorch Implementation

A faithful and minimal PyTorch implementation of Google’s Imagen — a powerful text-to-image diffusion model combining T5 for text encoding and cascading super-resolution diffusion models.

🚀 Features
✅ Fully modular & minimal design

🧠 Uses T5 encoder for rich text embeddings

🧊 Cascading Diffusion Models:

Base: 64×64 image generation

SR-1: 64→256 super-resolution

SR-2: 256→1024 final upscale

🎨 Trainer included with exponential moving average (EMA)

🔌 Easily plug your own UNet, text encoder, or scheduler

🧪 Installation
bash
Copy
pip install imagen-pytorch
Make sure you have PyTorch >= 1.10 and HuggingFace Transformers installed.

📦 Getting Started
python
Copy
from imagen_pytorch import Unet, Imagen, ImagenTrainer

# Define U-Nets for each stage
unet1 = Unet(dim=32, dim_mults=(1, 2, 4), ...)
unet2 = Unet(dim=32, dim_mults=(1, 2, 4), ...)
unet3 = Unet(dim=32, dim_mults=(1, 2, 4), ...)

imagen = Imagen(
    unets = (unet1, unet2, unet3),
    text_encoder_name = 't5-base',
    image_sizes = (64, 256, 1024),
    timesteps = 1000,
    ...
)

trainer = ImagenTrainer(imagen)

# Train
for i in range(100000):
    loss = trainer(
        text = ['a red car driving through mountains'],
        images = your_images_tensor
    )
📸 Sample Output


📚 Citation
Original paper:
Imagen: Scaling Up Denoising Diffusion Models for High Fidelity Image Generation
Saharia et al., Google Research, 2022
[arXiv]

🤝 Contributing
Pull requests are welcome — whether it’s adding new features, improving code clarity, or fixing bugs. Feel free to open issues for discussion or questions.

