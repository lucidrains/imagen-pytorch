from imagen_pytorch.trainer import ImagenTrainer
from imagen_pytorch.configs import ImagenConfig
from imagen_pytorch.t5 import t5_encode_text
from torch.utils.data import Dataset
import torch

def test_trainer_instantiation():
    unet1 = dict(
        dim = 8,
        dim_mults = (1, 1, 1, 1),
        num_resnet_blocks = 1,
        layer_attns = False,
        layer_cross_attns = False,
        attn_heads = 2
    )

    imagen = ImagenConfig(
        unets=(unet1,),
        image_sizes=(64,),
    ).create()

    trainer = ImagenTrainer(
        imagen=imagen
    )

def test_trainer_step():
    class TestDataset(Dataset):
        def __init__(self):
            super().__init__()
        def __len__(self):
            return 16
        def __getitem__(self, index):
            return (torch.zeros(3, 64, 64), torch.zeros(6, 768))
    unet1 = dict(
        dim = 8,
        dim_mults = (1, 1, 1, 1),
        num_resnet_blocks = 1,
        layer_attns = False,
        layer_cross_attns = False,
        attn_heads = 2
    )

    imagen = ImagenConfig(
        unets=(unet1,),
        image_sizes=(64,),
    ).create()

    trainer = ImagenTrainer(
        imagen=imagen
    )
    ds = TestDataset()
    trainer.add_train_dataset(ds, batch_size=8)
    trainer.train_step(1)
    assert trainer.num_steps_taken(1) == 1