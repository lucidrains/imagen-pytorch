from math import sqrt
from functools import partial
from contextlib import contextmanager, nullcontext
from typing import List
from collections import namedtuple
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch import nn, einsum
from torch.cuda.amp import autocast
import torchvision.transforms as T

import kornia.augmentation as K

from einops import rearrange, repeat, reduce

from imagen_pytorch.imagen_pytorch import (
    GaussianDiffusionContinuousTimes,
    Unet,
    first,
    exists,
    identity,
    maybe,
    default,
    cast_tuple,
    cast_uint8_images_to_float,
    eval_decorator,
    check_shape,
    pad_tuple_to_length,
    resize_image_to,
    right_pad_dims_to,
    module_device,
    normalize_neg_one_to_one,
    unnormalize_zero_to_one,
)

from imagen_pytorch.t5 import t5_encode_text, get_encoded_dim, DEFAULT_T5_NAME

# constants

Hparams_fields = [
    'num_sample_steps',
    'sigma_min',
    'sigma_max',
    'sigma_data',
    'rho',
    'P_mean',
    'P_std',
    'S_churn',
    'S_tmin',
    'S_tmax',
    'S_noise'
]

Hparams = namedtuple('Hparams', Hparams_fields)

# helper functions

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

# main class

class ElucidatedImagen(nn.Module):
    def __init__(
        self,
        unets,
        *,
        image_sizes,                                # for cascading ddpm, image size at each stage
        text_encoder_name = DEFAULT_T5_NAME,
        text_embed_dim = None,
        channels = 3,
        cond_drop_prob = 0.1,
        random_crop_sizes = None,
        lowres_sample_noise_level = 0.2,            # in the paper, they present a new trick where they noise the lowres conditioning image, and at sample time, fix it to a certain level (0.1 or 0.3) - the unets are also made to be conditioned on this noise level
        per_sample_random_aug_noise_level = False,  # unclear when conditioning on augmentation noise level, whether each batch element receives a random aug noise value - turning off due to @marunine's find
        condition_on_text = True,
        auto_normalize_img = True,                  # whether to take care of normalizing the image from [0, 1] to [-1, 1] and back automatically - you can turn this off if you want to pass in the [-1, 1] ranged image yourself from the dataloader
        dynamic_thresholding = True,
        dynamic_thresholding_percentile = 0.95,     # unsure what this was based on perusal of paper
        only_train_unet_number = None,
        lowres_noise_schedule = 'linear',
        num_sample_steps = 32,                      # number of sampling steps
        sigma_min = 0.002,                          # min noise level
        sigma_max = 80,                             # max noise level
        sigma_data = 0.5,                           # standard deviation of data distribution
        rho = 7,                                    # controls the sampling schedule
        P_mean = -1.2,                              # mean of log-normal distribution from which noise is drawn for training
        P_std = 1.2,                                # standard deviation of log-normal distribution from which noise is drawn for training
        S_churn = 80,                               # parameters for stochastic sampling - depends on dataset, Table 5 in apper
        S_tmin = 0.05,
        S_tmax = 50,
        S_noise = 1.003,
    ):
        super().__init__()

        # conditioning hparams

        self.condition_on_text = condition_on_text
        self.unconditional = not condition_on_text

        # channels

        self.channels = channels

        # automatically take care of ensuring that first unet is unconditional
        # while the rest of the unets are conditioned on the low resolution image produced by previous unet

        unets = cast_tuple(unets)
        num_unets = len(unets)

        # randomly cropping for upsampler training

        self.random_crop_sizes = cast_tuple(random_crop_sizes, num_unets)
        assert not exists(first(self.random_crop_sizes)), 'you should not need to randomly crop image during training for base unet, only for upsamplers - so pass in `random_crop_sizes = (None, 128, 256)` as example'

        # lowres augmentation noise schedule

        self.lowres_noise_schedule = GaussianDiffusionContinuousTimes(noise_schedule = lowres_noise_schedule)

        # get text encoder

        self.text_encoder_name = text_encoder_name
        self.text_embed_dim = default(text_embed_dim, lambda: get_encoded_dim(text_encoder_name))

        self.encode_text = partial(t5_encode_text, name = text_encoder_name)

        # construct unets

        self.unets = nn.ModuleList([])
        self.unet_being_trained_index = -1 # keeps track of which unet is being trained at the moment

        for ind, one_unet in enumerate(unets):
            assert isinstance(one_unet, Unet)
            is_first = ind == 0

            one_unet = one_unet.cast_model_parameters(
                lowres_cond = not is_first,
                cond_on_text = self.condition_on_text,
                text_embed_dim = self.text_embed_dim if self.condition_on_text else None,
                channels = self.channels,
                channels_out = self.channels
            )

            self.unets.append(one_unet)

        # unet image sizes

        self.image_sizes = cast_tuple(image_sizes)
        assert num_unets == len(image_sizes), f'you did not supply the correct number of u-nets ({len(self.unets)}) for resolutions {image_sizes}'

        self.sample_channels = cast_tuple(self.channels, num_unets)

        # cascading ddpm related stuff

        lowres_conditions = tuple(map(lambda t: t.lowres_cond, self.unets))
        assert lowres_conditions == (False, *((True,) * (num_unets - 1))), 'the first unet must be unconditioned (by low resolution image), and the rest of the unets must have `lowres_cond` set to True'

        self.lowres_sample_noise_level = lowres_sample_noise_level
        self.per_sample_random_aug_noise_level = per_sample_random_aug_noise_level

        # classifier free guidance

        self.cond_drop_prob = cond_drop_prob
        self.can_classifier_guidance = cond_drop_prob > 0.

        # normalize and unnormalize image functions

        self.normalize_img = normalize_neg_one_to_one if auto_normalize_img else identity
        self.unnormalize_img = unnormalize_zero_to_one if auto_normalize_img else identity
        self.input_image_range = (0. if auto_normalize_img else -1., 1.)

        # dynamic thresholding

        self.dynamic_thresholding = cast_tuple(dynamic_thresholding, num_unets)
        self.dynamic_thresholding_percentile = dynamic_thresholding_percentile

        # elucidating parameters

        hparams = [
            num_sample_steps,
            sigma_min,
            sigma_max,
            sigma_data,
            rho,
            P_mean,
            P_std,
            S_churn,
            S_tmin,
            S_tmax,
            S_noise,
        ]

        hparams = [cast_tuple(hp, num_unets) for hp in hparams]
        self.hparams = [Hparams(*unet_hp) for unet_hp in zip(*hparams)]

        # one temp parameter for keeping track of device

        self.register_buffer('_temp', torch.tensor([0.]), persistent = False)

        # default to device of unets passed in

        self.to(next(self.unets.parameters()).device)

    @property
    def device(self):
        return self._temp.device

    def get_unet(self, unet_number):
        assert 0 < unet_number <= len(self.unets)
        index = unet_number - 1

        if isinstance(self.unets, nn.ModuleList):
            unets_list = [unet for unet in self.unets]
            delattr(self, 'unets')
            self.unets = unets_list

        if index != self.unet_being_trained_index:
            for unet_index, unet in enumerate(self.unets):
                unet.to(self.device if unet_index == index else 'cpu')

        self.unet_being_trained_index = index
        return self.unets[index]

    def reset_unets_all_one_device(self, device = None):
        device = default(device, self.device)
        self.unets = nn.ModuleList([*self.unets])
        self.unets.to(device)

        self.unet_being_trained_index = -1

    @contextmanager
    def one_unet_in_gpu(self, unet_number = None, unet = None):
        assert exists(unet_number) ^ exists(unet)

        if exists(unet_number):
            unet = self.unets[unet_number - 1]

        devices = [module_device(unet) for unet in self.unets]
        self.unets.cpu()
        unet.to(self.device)

        yield

        for unet, device in zip(self.unets, devices):
            unet.to(device)

    # overriding state dict functions

    def state_dict(self, *args, **kwargs):
        self.reset_unets_all_one_device()
        return super().state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        self.reset_unets_all_one_device()
        return super().load_state_dict(*args, **kwargs)

    # dynamic thresholding

    def threshold_x_start(self, x_start, dynamic_threshold = True):
        if not dynamic_threshold:
            return x_start.clamp(-1., 1.)

        s = torch.quantile(
            rearrange(x_start, 'b ... -> b (...)').abs(),
            self.dynamic_thresholding_percentile,
            dim = -1
        )

        s.clamp_(min = 1.)
        s = right_pad_dims_to(x_start, s)
        return x_start.clamp(-s, s) / s

    # derived preconditioning params - Table 1

    def c_skip(self, sigma_data, sigma):
        return (sigma_data ** 2) / (sigma ** 2 + sigma_data ** 2)

    def c_out(self, sigma_data, sigma):
        return sigma * sigma_data * (sigma_data ** 2 + sigma ** 2) ** -0.5

    def c_in(self, sigma_data, sigma):
        return 1 * (sigma ** 2 + sigma_data ** 2) ** -0.5

    def c_noise(self, sigma):
        return log(sigma) * 0.25

   # preconditioned network output
    # equation (7) in the paper

    def preconditioned_network_forward(
        self,
        unet_forward,
        noised_images,
        sigma,
        *,
        sigma_data,
        clamp = False,
        dynamic_threshold = True,
        **kwargs
    ):
        batch, device = noised_images.shape[0], noised_images.device

        if isinstance(sigma, float):
            sigma = torch.full((batch,), sigma, device = device)

        padded_sigma = rearrange(sigma, 'b -> b 1 1 1')

        net_out = unet_forward(
            self.c_in(sigma_data, padded_sigma) * noised_images,
            self.c_noise(sigma),
            **kwargs
        )

        out = self.c_skip(sigma_data, padded_sigma) * noised_images +  self.c_out(sigma_data, padded_sigma) * net_out

        if not clamp:
            return out

        return self.threshold_x_start(out, dynamic_threshold)

    # sampling

    # sample schedule
    # equation (5) in the paper

    def sample_schedule(
        self,
        num_sample_steps,
        rho,
        sigma_min,
        sigma_max
    ):
        N = num_sample_steps
        inv_rho = 1 / rho

        steps = torch.arange(num_sample_steps, device = self.device, dtype = torch.float32)
        sigmas = (sigma_max ** inv_rho + steps / (N - 1) * (sigma_min ** inv_rho - sigma_max ** inv_rho)) ** rho

        sigmas = F.pad(sigmas, (0, 1), value = 0.) # last step is sigma value of 0.
        return sigmas

    @torch.no_grad()
    def one_unet_sample(
        self,
        unet,
        shape,
        *,
        unet_number,
        clamp = True,
        dynamic_threshold = True,
        cond_scale = 1.,
        use_tqdm = True,
        inpaint_images = None,
        inpaint_masks = None,
        inpaint_resample_times = 5,
        **kwargs
    ):
        # get specific sampling hyperparameters for unet

        hp = self.hparams[unet_number - 1]

        # get the schedule, which is returned as (sigma, gamma) tuple, and pair up with the next sigma and gamma

        sigmas = self.sample_schedule(hp.num_sample_steps, hp.rho, hp.sigma_min, hp.sigma_max)

        gammas = torch.where(
            (sigmas >= hp.S_tmin) & (sigmas <= hp.S_tmax),
            min(hp.S_churn / hp.num_sample_steps, sqrt(2) - 1),
            0.
        )

        sigmas_and_gammas = list(zip(sigmas[:-1], sigmas[1:], gammas[:-1]))

        # images is noise at the beginning

        init_sigma = sigmas[0]

        images = init_sigma * torch.randn(shape, device = self.device)

        # prepare inpainting images and mask

        has_inpainting = exists(inpaint_images) and exists(inpaint_masks)
        resample_times = inpaint_resample_times if has_inpainting else 1

        if has_inpainting:
            inpaint_images = self.normalize_img(inpaint_images)
            inpaint_images = resize_image_to(inpaint_images, shape[-1])
            inpaint_masks = resize_image_to(rearrange(inpaint_masks, 'b ... -> b 1 ...').float(), shape[-1]).bool()

        # unet kwargs

        unet_kwargs = dict(
            sigma_data = hp.sigma_data,
            clamp = clamp,
            dynamic_threshold = dynamic_threshold,
            cond_scale = cond_scale,
            **kwargs
        )

        # gradually denoise

        total_steps = len(sigmas_and_gammas)

        for ind, (sigma, sigma_next, gamma) in tqdm(enumerate(sigmas_and_gammas), total = total_steps, desc = 'sampling time step', disable = not use_tqdm):
            is_last_timestep = ind == (total_steps - 1)

            sigma, sigma_next, gamma = map(lambda t: t.item(), (sigma, sigma_next, gamma))

            for r in reversed(range(resample_times)):
                is_last_resample_step = r == 0

                eps = hp.S_noise * torch.randn(shape, device = self.device) # stochastic sampling

                sigma_hat = sigma + gamma * sigma
                added_noise = sqrt(sigma_hat ** 2 - sigma ** 2) * eps

                images_hat = images + added_noise

                if has_inpainting:
                    images_hat = images_hat * ~inpaint_masks + (inpaint_images + added_noise) * inpaint_masks

                model_output = self.preconditioned_network_forward(
                    unet.forward_with_cond_scale,
                    images_hat,
                    sigma_hat,
                    **unet_kwargs
                )

                denoised_over_sigma = (images_hat - model_output) / sigma_hat

                images_next = images_hat + (sigma_next - sigma_hat) * denoised_over_sigma

                # second order correction, if not the last timestep

                if sigma_next != 0:
                    model_output_next = self.preconditioned_network_forward(
                        unet.forward_with_cond_scale,
                        images_next,
                        sigma_next,
                        **unet_kwargs
                    )

                    denoised_prime_over_sigma = (images_next - model_output_next) / sigma_next
                    images_next = images_hat + 0.5 * (sigma_next - sigma_hat) * (denoised_over_sigma + denoised_prime_over_sigma)

                images = images_next

                if has_inpainting and not (is_last_resample_step or is_last_timestep):
                    # renoise in repaint and then resample
                    repaint_noise = torch.randn(shape, device = self.device)
                    images = images + (sigma - sigma_next) * repaint_noise

        images = images.clamp(-1., 1.)

        if has_inpainting:
            images = images * ~inpaint_masks + inpaint_images * inpaint_masks

        return self.unnormalize_img(images)

    @torch.no_grad()
    @eval_decorator
    def sample(
        self,
        texts: List[str] = None,
        text_masks = None,
        text_embeds = None,
        cond_images = None,
        inpaint_images = None,
        inpaint_masks = None,
        inpaint_resample_times = 5,
        batch_size = 1,
        cond_scale = 1.,
        lowres_sample_noise_level = None,
        stop_at_unet_number = None,
        return_all_unet_outputs = False,
        return_pil_images = False,
        use_tqdm = True,
        device = None,
    ):
        device = default(device, self.device)
        self.reset_unets_all_one_device(device = device)

        cond_images = maybe(cast_uint8_images_to_float)(cond_images)

        if exists(texts) and not exists(text_embeds) and not self.unconditional:
            with autocast(enabled = False):
                text_embeds, text_masks = self.encode_text(texts, return_attn_mask = True)

            text_embeds, text_masks = map(lambda t: t.to(device), (text_embeds, text_masks))

        if not self.unconditional:
            text_masks = default(text_masks, lambda: torch.any(text_embeds != 0., dim = -1))

        if not self.unconditional:
            batch_size = text_embeds.shape[0]

        assert not (self.condition_on_text and not exists(text_embeds)), 'text or text encodings must be passed into imagen if specified'
        assert not (not self.condition_on_text and exists(text_embeds)), 'imagen specified not to be conditioned on text, yet it is presented'
        assert not (exists(text_embeds) and text_embeds.shape[-1] != self.text_embed_dim), f'invalid text embedding dimension being passed in (should be {self.text_embed_dim})'

        assert not (exists(inpaint_images) ^ exists(inpaint_masks)),  'inpaint images and masks must be both passed in to do inpainting'

        outputs = []

        is_cuda = next(self.parameters()).is_cuda
        device = next(self.parameters()).device

        lowres_sample_noise_level = default(lowres_sample_noise_level, self.lowres_sample_noise_level)

        num_unets = len(self.unets)
        cond_scale = cast_tuple(cond_scale, num_unets)

        for unet_number, unet, channel, image_size, unet_hparam, dynamic_threshold, unet_cond_scale in tqdm(zip(range(1, num_unets + 1), self.unets, self.sample_channels, self.image_sizes, self.hparams, self.dynamic_thresholding, cond_scale), disable = not use_tqdm):

            context = self.one_unet_in_gpu(unet = unet) if is_cuda else nullcontext()

            with context:
                lowres_cond_img = lowres_noise_times = None
                shape = (batch_size, channel, image_size, image_size)

                if unet.lowres_cond:
                    lowres_noise_times = self.lowres_noise_schedule.get_times(batch_size, lowres_sample_noise_level, device = device)

                    lowres_cond_img = resize_image_to(img, image_size)
                    lowres_cond_img = self.normalize_img(lowres_cond_img)

                    lowres_cond_img, _ = self.lowres_noise_schedule.q_sample(x_start = lowres_cond_img, t = lowres_noise_times, noise = torch.randn_like(lowres_cond_img))

                shape = (batch_size, self.channels, image_size, image_size)

                img = self.one_unet_sample(
                    unet,
                    shape,
                    unet_number = unet_number,
                    text_embeds = text_embeds,
                    text_mask = text_masks,
                    cond_images = cond_images,
                    inpaint_images = inpaint_images,
                    inpaint_masks = inpaint_masks,
                    inpaint_resample_times = inpaint_resample_times,
                    cond_scale = unet_cond_scale,
                    lowres_cond_img = lowres_cond_img,
                    lowres_noise_times = lowres_noise_times,
                    dynamic_threshold = dynamic_threshold,
                    use_tqdm = use_tqdm
                )

                outputs.append(img)

            if exists(stop_at_unet_number) and stop_at_unet_number == unet_number:
                break

        output_index = -1 if not return_all_unet_outputs else slice(None) # either return last unet output or all unet outputs

        if not return_pil_images:
            return outputs[output_index]

        if not return_all_unet_outputs:
            outputs = outputs[-1:]

        pil_images = list(map(lambda img: list(map(T.ToPILImage(), img.unbind(dim = 0))), outputs))

        return pil_images[output_index] # now you have a bunch of pillow images you can just .save(/where/ever/you/want.png)

    # training

    def loss_weight(self, sigma_data, sigma):
        return (sigma ** 2 + sigma_data ** 2) * (sigma * sigma_data) ** -2

    def noise_distribution(self, P_mean, P_std, batch_size):
        return (P_mean + P_std * torch.randn((batch_size,), device = self.device)).exp()

    def forward(
        self,
        images,
        unet: Unet = None,
        texts: List[str] = None,
        text_embeds = None,
        text_masks = None,
        unet_number = None,
        cond_images = None
    ):
        assert images.shape[-1] == images.shape[-2], f'the images you pass in must be a square, but received dimensions of {images.shape[2]}, {images.shape[-1]}'
        assert not (len(self.unets) > 1 and not exists(unet_number)), f'you must specify which unet you want trained, from a range of 1 to {len(self.unets)}, if you are training cascading DDPM (multiple unets)'
        unet_number = default(unet_number, 1)
        assert not exists(self.only_train_unet_number) or self.only_train_unet_number == unet_number, 'you can only train on unet #{self.only_train_unet_number}'

        images = cast_uint8_images_to_float(images)
        cond_images = maybe(cast_uint8_images_to_float)(cond_images)

        unet_index = unet_number - 1
        
        unet = default(unet, lambda: self.get_unet(unet_number))

        target_image_size    = self.image_sizes[unet_index]
        random_crop_size     = self.random_crop_sizes[unet_index]
        prev_image_size      = self.image_sizes[unet_index - 1] if unet_index > 0 else None
        hp                   = self.hparams[unet_index]

        batch_size, c, h, w, device,  = *images.shape, images.device

        check_shape(images, 'b c h w', c = self.channels)
        assert h >= target_image_size and w >= target_image_size

        if exists(texts) and not exists(text_embeds) and not self.unconditional:
            assert len(texts) == len(images), 'number of text captions does not match up with the number of images given'

            with autocast(enabled = False):
                text_embeds, text_masks = self.encode_text(texts, return_attn_mask = True)

            text_embeds, text_masks = map(lambda t: t.to(images.device), (text_embeds, text_masks))

        if not self.unconditional:
            text_masks = default(text_masks, lambda: torch.any(text_embeds != 0., dim = -1))

        assert not (self.condition_on_text and not exists(text_embeds)), 'text or text encodings must be passed into decoder if specified'
        assert not (not self.condition_on_text and exists(text_embeds)), 'decoder specified not to be conditioned on text, yet it is presented'

        assert not (exists(text_embeds) and text_embeds.shape[-1] != self.text_embed_dim), f'invalid text embedding dimension being passed in (should be {self.text_embed_dim})'

        lowres_cond_img = lowres_aug_times = None
        if exists(prev_image_size):
            lowres_cond_img = resize_image_to(images, prev_image_size, clamp_range = self.input_image_range)
            lowres_cond_img = resize_image_to(lowres_cond_img, target_image_size, clamp_range = self.input_image_range)

            if self.per_sample_random_aug_noise_level:
                lowres_aug_times = self.lowres_noise_schedule.sample_random_times(batch_size, device = device)
            else:
                lowres_aug_time = self.lowres_noise_schedule.sample_random_times(1, device = device)
                lowres_aug_times = repeat(lowres_aug_time, '1 -> b', b = batch_size)

        images = resize_image_to(images, target_image_size)

        # normalize to [-1, 1]

        images = self.normalize_img(images)
        lowres_cond_img = maybe(self.normalize_img)(lowres_cond_img)

        # random cropping during training
        # for upsamplers

        if exists(random_crop_size):
            aug = K.RandomCrop((random_crop_size, random_crop_size), p = 1.)

            # make sure low res conditioner and image both get augmented the same way
            # detailed https://kornia.readthedocs.io/en/latest/augmentation.module.html?highlight=randomcrop#kornia.augmentation.RandomCrop
            images = aug(images)
            lowres_cond_img = aug(lowres_cond_img, params = aug._params)

        # noise the lowres conditioning image
        # at sample time, they then fix the noise level of 0.1 - 0.3

        lowres_cond_img_noisy = None
        if exists(lowres_cond_img):
            lowres_cond_img_noisy, _ = self.lowres_noise_schedule.q_sample(x_start = lowres_cond_img, t = lowres_aug_times, noise = torch.randn_like(lowres_cond_img))

        # get the sigmas

        sigmas = self.noise_distribution(hp.P_mean, hp.P_std, batch_size)
        padded_sigmas = rearrange(sigmas, 'b -> b 1 1 1')

        # noise

        noise = torch.randn_like(images)
        noised_images = images + padded_sigmas * noise  # alphas are 1. in the paper

        # get prediction

        denoised_images = self.preconditioned_network_forward(
            unet.forward,
            noised_images,
            sigmas,
            sigma_data = hp.sigma_data,
            text_embeds = text_embeds,
            text_mask = text_masks,
            cond_images = cond_images,
            lowres_noise_times = self.lowres_noise_schedule.get_condition(lowres_aug_times),
            lowres_cond_img = lowres_cond_img_noisy,
            cond_drop_prob = self.cond_drop_prob,
        )

        # losses

        losses = F.mse_loss(denoised_images, images, reduction = 'none')
        losses = reduce(losses, 'b ... -> b', 'mean')

        # loss weighting

        losses = losses * self.loss_weight(hp.sigma_data, sigmas)

        # return average loss

        return losses.mean()
