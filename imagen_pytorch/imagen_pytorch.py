import math
from typing import List
from tqdm import tqdm
from inspect import isfunction
from functools import partial, wraps
from contextlib import contextmanager
from collections import namedtuple
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn, einsum
from torch.special import expm1
import torchvision.transforms as T

from einops import rearrange, repeat
from einops.layers.torch import Rearrange, Reduce
from einops_exts import rearrange_many, repeat_many, check_shape
from einops_exts.torch import EinopsToAndFrom

from resize_right import resize

from imagen_pytorch.t5 import t5_encode_text, get_encoded_dim, DEFAULT_T5_NAME

# constants

NAT = 1. / math.log(2.)

# helper functions

def exists(val):
    return val is not None

def identity(t, *args, **kwargs):
    return t

def maybe(fn):
    @wraps(fn)
    def inner(x):
        if not exists(x):
            return x
        return fn(x)
    return inner

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def cast_tuple(val, length = 1):
    if isinstance(val, list):
        val = tuple(val)

    return val if isinstance(val, tuple) else ((val,) * length)

def module_device(module):
    return next(module.parameters()).device

@contextmanager
def null_context(*args, **kwargs):
    yield

def eval_decorator(fn):
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out
    return inner

def pad_tuple_to_length(t, length, fillvalue = None):
    remain_length = length - len(t)
    if remain_length <= 0:
        return t
    return (*t, *((fillvalue,) * remain_length))

# tensor helpers

def log(t, eps = 1e-12):
    return torch.log(t.clamp(min = eps))

def l2norm(t):
    return F.normalize(t, dim = -1)

def right_pad_dims_to(x, t):
    padding_dims = x.ndim - t.ndim
    if padding_dims <= 0:
        return t
    return t.view(*t.shape, *((1,) * padding_dims))

def masked_mean(t, *, dim, mask = None):
    if not exists(mask):
        return t.mean(dim = dim)

    denom = mask.sum(dim = dim, keepdim = True)
    mask = rearrange(mask, 'b n -> b n 1')
    masked_t = t.masked_fill(~mask, 0.)

    return masked_t.sum(dim = dim) / denom.clamp(min = 1e-5)

def resize_image_to(image, target_image_size):
    orig_image_size = image.shape[-1]

    if orig_image_size == target_image_size:
        return image

    scale_factors = target_image_size / orig_image_size
    return resize(image, scale_factors = scale_factors)

# image normalization functions
# ddpms expect images to be in the range of -1 to 1

def normalize_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_zero_to_one(normed_img):
    return (normed_img + 1) * 0.5

# classifier free guidance functions

def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device = device, dtype = torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device = device, dtype = torch.bool)
    else:
        return torch.zeros(shape, device = device).float().uniform_(0, 1) < prob

# gaussian diffusion helper functions

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def noise_like(shape, device, repeat=False):
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()

def meanflat(x):
    return x.mean(dim = tuple(range(1, len(x.shape))))

def normal_kl(mean1, logvar1, mean2, logvar2):
    return 0.5 * (-1.0 + logvar2 - logvar1 + torch.exp(logvar1 - logvar2) + ((mean1 - mean2) ** 2) * torch.exp(-logvar2))

def approx_standard_normal_cdf(x):
    return 0.5 * (1.0 + torch.tanh(((2.0 / math.pi) ** 0.5) * (x + 0.044715 * (x ** 3))))

def discretized_gaussian_log_likelihood(x, *, means, log_scales, thres = 0.999):
    assert x.shape == means.shape == log_scales.shape

    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1. / 255.)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1. / 255.)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_cdf_plus = log(cdf_plus)
    log_one_minus_cdf_min = log(1. - cdf_min)
    cdf_delta = cdf_plus - cdf_min

    log_probs = torch.where(x < -thres,
        log_cdf_plus,
        torch.where(x > thres,
            log_one_minus_cdf_min,
            log(cdf_delta)))

    return log_probs

def cosine_beta_schedule(timesteps, s = 0.008, thres = 0.999):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype = torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, thres)


def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)


class GaussianDiffusion(nn.Module):
    def __init__(self, *, beta_schedule, timesteps):
        super().__init__()

        if beta_schedule == "cosine":
            betas = cosine_beta_schedule(timesteps)
        elif beta_schedule == "linear":
            betas = linear_beta_schedule(timesteps)
        else:
            raise NotImplementedError()

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis = 0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        # register buffer helper function to cast double back to float

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32), persistent = False)

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', log(posterior_variance, eps = 1e-20))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

    def get_times(self, batch_size, noise_level):
        device = self.betas.device
        return torch.full((batch_size,), int(self.num_timesteps * noise_level), device = device, dtype = torch.long)

    def sample_random_times(self, batch_size):
        device = self.betas.device
        return torch.randint(0, self.num_timesteps, (batch_size,), device = device, dtype = torch.long)

    def get_learned_posterior_log_variance(self, var_interp_frac_unnormalized, x_t, t):
        # if learned variance, posterior variance and posterior log variance are predicted by the network
        # by an interpolation of the max and min log beta values
        # eq 15 - https://arxiv.org/abs/2102.09672
        min_log = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        max_log = extract(torch.log(self.betas), t, x_t.shape)
        var_interp_frac = unnormalize_zero_to_one(var_interp_frac_unnormalized)

        posterior_log_variance = var_interp_frac * max_log + (1 - var_interp_frac) * min_log
        return posterior_log_variance

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

# gaussian diffusion with continuos time helper functions and classes
# large part of this was thanks to @crowsonkb at https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/utils.py

@torch.jit.script
def beta_linear_log_snr(t):
    return -torch.log(expm1(1e-4 + 10 * (t ** 2)))

def alpha_cosine_log_snr(t):
    raise NotImplementedError

def log_snr_to_alpha_sigma(log_snr):
    return torch.sqrt(torch.sigmoid(log_snr)), torch.sqrt(torch.sigmoid(-log_snr))

class GaussianDiffusionContinuousTimes(GaussianDiffusion):
    def __init__(self, *, beta_schedule, **kwargs):
        super().__init__()
        if beta_schedule == 'linear':
            self.log_snr = beta_linear_log_snr
        elif beta_schedule == "cosine":
            raise NotImplementedError
        else:
            raise ValueError(f'invalid noise schedule {beta_schedule}')

    def get_times(self, batch_size, noise_level):
        device = self.betas.device
        return torch.full((batch_size,), noise_level, device = device, dtype = torch.long)

    def sample_random_times(self, batch_size, max_thres = 0.999):
        device = self.betas.device
        return torch.zeros((batch_size,), device = device).float().uniform_(0, max_thres)

    def get_learned_posterior_log_variance(self, var_interp_frac_unnormalized, x_t, t):
        raise NotImplementedError

    def q_posterior(self, pred_noise, x_t, t, t_next):
        """ https://openreview.net/attachment?id=2LdBqxc1Yv&name=supplementary_material """
        # pred_noise needs to be replaced by x_start, which comes from predict_start_from_noise, intermediate is dynamically threshold (i think?)
        log_snr = self.log_snr(t)
        log_snr_next = self.log_snr(t_next)
        log_snr, log_snr_next = map(partial(right_pad_dims_to, x_t), (log_snr, log_snr_next))

        alpha, sigma = log_snr_to_alpha_sigma(log_snr)
        alpha_next, sigma_next = log_snr_to_alpha_sigma(log_snr_next)

        # c - as defined near eq 33
        c = -expm1(log_snr_next - log_snr)

        # (eq. 31) - still need to derive x_start, dynamic threshold, before calculating posterior mean
        # as noted in the first comment of openreview, the equation in the paper is incorrect, and missing a sigma
        posterior_mean = (alpha_next / alpha) * (x_t - c * sigma * pred_noise)

        # following (eq. 33)
        posterior_variance = (sigma_next ** 2) * c
        posterior_log_variance_clipped = log(posterior_variance, eps = 1e-20)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def q_sample(self, x_start, t, noise = None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        log_snr = self.log_snr(x_start, t)
        log_snr = right_pad_dims_to(x_start, log_snr)
        alpha, sigma =  log_snr_to_alpha_sigma(log_snr)
        return alpha * x_start + sigma * noise

    def predict_start_from_noise(self, x_t, t, noise):
        log_snr = self.log_snr(t)
        log_snr = right_pad_dims_to(x_t, log_snr)
        alpha, sigma = log_snr_to_alpha_sigma(log_snr)
        return (x_t - sigma * noise) / alpha.clamp(min = 1e-5)

# norms and residuals

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)


class ChanLayerNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) / (var + self.eps).sqrt() * self.g


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

# attention pooling

class PerceiverAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head = 64,
        heads = 8
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm = nn.LayerNorm(dim)
        self.norm_latents = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias = False),
            nn.LayerNorm(dim)
        )

    def forward(self, x, latents, mask = None):
        x = self.norm(x)
        latents = self.norm_latents(latents)

        b, h = x.shape[0], self.heads

        q = self.to_q(latents)

        # the paper differs from Perceiver in which they also concat the key / values derived from the latents to be attended to
        kv_input = torch.cat((x, latents), dim = -2)
        k, v = self.to_kv(kv_input).chunk(2, dim = -1)

        q, k, v = rearrange_many((q, k, v), 'b n (h d) -> b h n d', h = h)

        q = q * self.scale

        # attention

        sim = einsum('... i d, ... j d  -> ... i j', q, k)

        if exists(mask):
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = F.pad(mask, (0, latents.shape[-2]), value = True)
            mask = rearrange(mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~mask, max_neg_value)

        attn = sim.softmax(dim = -1)

        out = einsum('... i j, ... j d -> ... i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)', h = h)
        return self.to_out(out)

class PerceiverResampler(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        dim_head = 64,
        heads = 8,
        num_latents = 64,
        num_latents_mean_pooled = 4, # number of latents derived from mean pooled representation of the sequence
        max_seq_len = 512,
        ff_mult = 4,
    ):
        super().__init__()
        self.pos_emb = nn.Embedding(max_seq_len, dim)

        self.latents = nn.Parameter(torch.randn(num_latents, dim))

        self.to_latents_from_mean_pooled_seq = None

        if num_latents_mean_pooled > 0:
            self.to_latents_from_mean_pooled_seq = nn.Sequential(
                LayerNorm(dim),
                nn.Linear(dim, dim * num_latents_mean_pooled),
                Rearrange('b (n d) -> b n d', n = num_latents_mean_pooled)
            )

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PerceiverAttention(dim = dim, dim_head = dim_head, heads = heads),
                FeedForward(dim = dim, mult = ff_mult)
            ]))

    def forward(self, x, mask = None):
        n, device = x.shape[1], x.device
        pos_emb = self.pos_emb(torch.arange(n, device = device))

        x_with_pos = x + pos_emb

        latents = repeat(self.latents, 'n d -> b n d', b = x.shape[0])

        if exists(self.to_latents_from_mean_pooled_seq):
            meanpooled_seq = masked_mean(x, dim = 1, mask = torch.ones(x.shape[:2], device = x.device, dtype = torch.bool))
            meanpooled_latents = self.to_latents_from_mean_pooled_seq(meanpooled_seq)
            latents = torch.cat((meanpooled_latents, latents), dim = -2)

        for attn, ff in self.layers:
            latents = attn(x_with_pos, latents, mask = mask) + latents
            latents = ff(latents) + latents

        return latents

# attention

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        *,
        dim_head = 64,
        heads = 8,
        causal = False,
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.causal = causal
        self.norm = LayerNorm(dim)

        self.null_kv = nn.Parameter(torch.randn(2, dim_head))
        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, dim_head * 2, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias = False),
            LayerNorm(dim)
        )

    def forward(self, x, mask = None, attn_bias = None):
        b, n, device = *x.shape[:2], x.device

        x = self.norm(x)
        q, k, v = (self.to_q(x), *self.to_kv(x).chunk(2, dim = -1))

        q = rearrange(q, 'b n (h d) -> b h n d', h = self.heads)
        q = q * self.scale

        # add null key / value for classifier free guidance in prior net

        nk, nv = repeat_many(self.null_kv.unbind(dim = -2), 'd -> b 1 d', b = b)
        k = torch.cat((nk, k), dim = -2)
        v = torch.cat((nv, v), dim = -2)

        # calculate query / key similarities

        sim = einsum('b h i d, b j d -> b h i j', q, k)

        # relative positional encoding (T5 style)

        if exists(attn_bias):
            sim = sim + attn_bias

        # masking

        max_neg_value = -torch.finfo(sim.dtype).max

        if exists(mask):
            mask = F.pad(mask, (1, 0), value = True)
            mask = rearrange(mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~mask, max_neg_value)

        # attention

        attn = sim.softmax(dim = -1)

        # aggregate values

        out = einsum('b h i j, b j d -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

# decoder

def Upsample(dim):
    return nn.ConvTranspose2d(dim, dim, 4, 2, 1)

def Downsample(dim):
    return nn.Conv2d(dim, dim, 4, 2, 1)

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device = x.device) * -emb)
        emb = rearrange(x, 'i -> i 1') * rearrange(emb, 'j -> 1 j')
        return torch.cat((emb.sin(), emb.cos()), dim = -1)

class Block(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
        groups = 8,
        norm = True
    ):
        super().__init__()
        self.groupnorm = nn.GroupNorm(groups, dim) if norm else nn.Identity()
        self.activation = nn.SiLU()
        self.project = nn.Conv2d(dim, dim_out, 3, padding = 1)

    def forward(self, x, scale_shift = None):
        x = self.groupnorm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.activation(x)
        return self.project(x)

class ResnetBlock(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
        *,
        cond_dim = None,
        time_cond_dim = None,
        groups = 8
    ):
        super().__init__()

        self.time_mlp = None

        if exists(time_cond_dim):
            self.time_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_cond_dim, dim_out * 2)
            )

        self.cross_attn = None

        if exists(cond_dim):
            self.cross_attn = EinopsToAndFrom(
                'b c h w',
                'b (h w) c',
                CrossAttention(
                    dim = dim_out,
                    context_dim = cond_dim
                )
            )

        self.block1 = Block(dim, dim_out, groups = groups)
        self.block2 = Block(dim_out, dim_out, groups = groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, cond = None, time_emb = None):

        scale_shift = None
        if exists(self.time_mlp) and exists(time_emb):
            time_emb = self.time_mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim = 1)

        h = self.block1(x)

        if exists(self.cross_attn):
            assert exists(cond)
            h = self.cross_attn(h, context = cond) + h

        h = self.block2(h, scale_shift = scale_shift)

        return h + self.res_conv(x)

class CrossAttention(nn.Module):
    def __init__(
        self,
        dim,
        *,
        context_dim = None,
        dim_head = 64,
        heads = 8,
        norm_context = False
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads

        context_dim = default(context_dim, dim)

        self.norm = LayerNorm(dim)
        self.norm_context = LayerNorm(context_dim) if norm_context else nn.Identity()

        self.null_kv = nn.Parameter(torch.randn(2, dim_head))
        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias = False),
            LayerNorm(dim)
        )

    def forward(self, x, context, mask = None):
        b, n, device = *x.shape[:2], x.device

        x = self.norm(x)
        context = self.norm_context(context)

        q, k, v = (self.to_q(x), *self.to_kv(context).chunk(2, dim = -1))

        q, k, v = rearrange_many((q, k, v), 'b n (h d) -> b h n d', h = self.heads)

        # add null key / value for classifier free guidance in prior net

        nk, nv = repeat_many(self.null_kv.unbind(dim = -2), 'd -> b h 1 d', h = self.heads,  b = b)

        k = torch.cat((nk, k), dim = -2)
        v = torch.cat((nv, v), dim = -2)

        q = q * self.scale

        sim = einsum('b h i d, b h j d -> b h i j', q, k)
        max_neg_value = -torch.finfo(sim.dtype).max

        if exists(mask):
            mask = F.pad(mask, (1, 0), value = True)
            mask = rearrange(mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~mask, max_neg_value)

        attn = sim.softmax(dim = -1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class LinearAttention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 32,
        heads = 8
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads
        self.norm = ChanLayerNorm(dim)

        self.nonlin = nn.SiLU()
        self.to_qkv = nn.Conv2d(dim, inner_dim * 3, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv2d(inner_dim, dim, 1, bias = False),
            ChanLayerNorm(dim)
        )

    def forward(self, fmap):
        h, x, y = self.heads, *fmap.shape[-2:]

        fmap = self.norm(fmap)
        q, k, v = self.to_qkv(fmap).chunk(3, dim = 1)
        q, k, v = rearrange_many((q, k, v), 'b (h c) x y -> (b h) (x y) c', h = h)

        q = q.softmax(dim = -1)
        k = k.softmax(dim = -2)

        q = q * self.scale

        context = einsum('b n d, b n e -> b d e', k, v)
        out = einsum('b n d, b d e -> b n e', q, context)
        out = rearrange(out, '(b h) (x y) d -> b (h d) x y', h = h, x = x, y = y)

        out = self.nonlin(out)
        return self.to_out(out)

def FeedForward(dim, mult = 2):
    hidden_dim = int(dim * mult)
    return nn.Sequential(
        LayerNorm(dim),
        nn.Linear(dim, hidden_dim, bias = False),
        nn.GELU(),
        LayerNorm(hidden_dim),
        nn.Linear(hidden_dim, dim, bias = False)
    )

def ChanFeedForward(dim, mult = 2):  # in paper, it seems for self attention layers they did feedforwards with twice channel width
    hidden_dim = int(dim * mult)
    return nn.Sequential(
        ChanLayerNorm(dim),
        nn.Conv2d(dim, hidden_dim, 1, bias = False),
        nn.GELU(),
        ChanLayerNorm(hidden_dim),
        nn.Conv2d(hidden_dim, dim, 1, bias = False)
    )

class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        *,
        heads = 8,
        dim_head = 32,
        ff_mult = 2
    ):
        super().__init__()
        self.attn = EinopsToAndFrom('b c h w', 'b (h w) c', Attention(dim = dim, heads = heads, dim_head = dim_head))
        self.ff = ChanFeedForward(dim = dim, mult = ff_mult)

    def forward(self, x):
        x = self.attn(x) + x
        x = self.ff(x) + x
        return x

class CrossEmbedLayer(nn.Module):
    def __init__(
        self,
        dim_in,
        kernel_sizes,
        dim_out = None,
        stride = 2
    ):
        super().__init__()
        assert all([*map(lambda t: (t % 2) == (stride % 2), kernel_sizes)])
        dim_out = default(dim_out, dim_in)

        kernel_sizes = sorted(kernel_sizes)
        num_scales = len(kernel_sizes)

        # calculate the dimension at each scale
        dim_scales = [int(dim_out / (2 ** i)) for i in range(1, num_scales)]
        dim_scales = [*dim_scales, dim_out - sum(dim_scales)]

        self.convs = nn.ModuleList([])
        for kernel, dim_scale in zip(kernel_sizes, dim_scales):
            self.convs.append(nn.Conv2d(dim_in, dim_scale, kernel, stride = stride, padding = (kernel - stride) // 2))

    def forward(self, x):
        fmaps = tuple(map(lambda conv: conv(x), self.convs))
        return torch.cat(fmaps, dim = 1)

class Unet(nn.Module):
    def __init__(
        self,
        *,
        dim,
        image_embed_dim = 1024,
        text_embed_dim = get_encoded_dim(DEFAULT_T5_NAME),
        num_resnet_blocks = 1,
        cond_dim = None,
        num_image_tokens = 4,
        num_time_tokens = 2,
        out_dim = None,
        dim_mults=(1, 2, 4, 8),
        channels = 3,
        channels_out = None,
        attn_dim_head = 64,
        attn_heads = 8,
        ff_mult = 2.,
        lowres_cond = False, # for cascading diffusion - https://cascaded-diffusion.github.io/
        layer_attns = True,
        attend_at_middle = True, # whether to have a layer of attention at the bottleneck (can turn off for higher resolution in cascading DDPM, before bringing in efficient attention)
        layer_cross_attns = True,
        cond_on_text = True,
        max_text_len = 256,
        init_dim = None,
        init_conv_kernel_size = 7,
        resnet_groups = 8,
        init_cross_embed_kernel_sizes = (3, 7, 15),
        cross_embed_downsample = False,
        cross_embed_downsample_kernel_sizes = (2, 4),
        attn_pool_text = True,
        attn_pool_num_latents = 32
    ):
        super().__init__()
        # save locals to take care of some hyperparameters for cascading DDPM

        self._locals = locals()
        self._locals.pop('self', None)
        self._locals.pop('__class__', None)

        # for eventual cascading diffusion

        self.lowres_cond = lowres_cond

        # determine dimensions

        self.channels = channels
        self.channels_out = default(channels_out, channels)

        init_channels = channels if not lowres_cond else channels * 2 # in cascading diffusion, one concats the low resolution image, blurred, for conditioning the higher resolution synthesis
        init_dim = default(init_dim, dim)

        self.init_conv = CrossEmbedLayer(init_channels, dim_out = init_dim, kernel_sizes = init_cross_embed_kernel_sizes, stride = 1)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # time, image embeddings, and optional text encoding

        cond_dim = default(cond_dim, dim)
        time_cond_dim = dim * 4

        self.to_time_hiddens = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, time_cond_dim),
            nn.SiLU()
        )

        self.to_lowres_time_hiddens = None
        if lowres_cond:
            self.to_lowres_time_hiddens = nn.Sequential(
                SinusoidalPosEmb(dim),
                nn.Linear(dim, time_cond_dim),
                nn.SiLU()
            )
            time_cond_dim *= 2

        # project to time tokens as well as time hiddens

        self.to_time_tokens = nn.Sequential(
            nn.Linear(time_cond_dim, cond_dim * num_time_tokens),
            Rearrange('b (r d) -> b r d', r = num_time_tokens)
        )

        self.to_time_cond = nn.Sequential(
            nn.Linear(time_cond_dim, time_cond_dim)
        )

        self.norm_cond = nn.LayerNorm(cond_dim)
        self.norm_mid_cond = nn.LayerNorm(cond_dim)

        # text encoding conditioning (optional)

        self.text_to_cond = None

        if cond_on_text:
            assert exists(text_embed_dim), 'text_embed_dim must be given to the unet if cond_on_text is True'
            self.text_to_cond = nn.Linear(text_embed_dim, cond_dim)

        # finer control over whether to condition on text encodings

        self.cond_on_text = cond_on_text

        # attention pooling

        self.attn_pool = PerceiverResampler(dim = cond_dim, depth = 2, dim_head = attn_dim_head, heads = attn_heads, num_latents = attn_pool_num_latents) if attn_pool_text else None

        # for classifier free guidance

        self.null_image_embed = nn.Parameter(torch.randn(1, num_image_tokens, cond_dim))

        self.max_text_len = max_text_len
        self.null_text_embed = nn.Parameter(torch.randn(1, max_text_len, cond_dim))

        # attention related params

        attn_kwargs = dict(heads = attn_heads, dim_head = attn_dim_head)

        num_layers = len(in_out)

        # resnet block klass

        num_resnet_blocks = cast_tuple(num_resnet_blocks, num_layers)
        resnet_groups = cast_tuple(resnet_groups, num_layers)

        layer_attns = cast_tuple(layer_attns, num_layers)
        layer_cross_attns = cast_tuple(layer_cross_attns, num_layers)

        assert all([layers == num_layers for layers in list(map(len, (resnet_groups, layer_attns, layer_cross_attns)))])

        # downsample klass

        downsample_klass = Downsample
        if cross_embed_downsample:
            downsample_klass = partial(CrossEmbedLayer, kernel_sizes = cross_embed_downsample_kernel_sizes)

        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        layer_params = [num_resnet_blocks, resnet_groups, layer_attns, layer_cross_attns]
        reversed_layer_params = list(map(reversed, layer_params))

        for ind, ((dim_in, dim_out), layer_num_resnet_blocks, groups, layer_attn, layer_cross_attn) in enumerate(zip(in_out, *layer_params)):
            is_last = ind >= (num_resolutions - 1)
            layer_cond_dim = cond_dim if layer_cross_attn else None

            self.downs.append(nn.ModuleList([
                ResnetBlock(dim_in, dim_out, cond_dim = layer_cond_dim, time_cond_dim = time_cond_dim, groups = groups),
                nn.ModuleList([ResnetBlock(dim_out, dim_out, groups = groups) for _ in range(layer_num_resnet_blocks)]),
                TransformerBlock(dim = dim_out, heads = attn_heads, dim_head = attn_dim_head, ff_mult = ff_mult) if layer_attn else nn.Identity(),
                downsample_klass(dim_out) if not is_last else nn.Identity()
            ]))

        mid_dim = dims[-1]

        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, cond_dim = cond_dim, time_cond_dim = time_cond_dim, groups = resnet_groups[-1])
        self.mid_attn = EinopsToAndFrom('b c h w', 'b (h w) c', Residual(Attention(mid_dim, **attn_kwargs))) if attend_at_middle else None
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, cond_dim = cond_dim, time_cond_dim = time_cond_dim, groups = resnet_groups[-1])

        for ind, ((dim_in, dim_out), layer_num_resnet_blocks, groups, layer_attn, layer_cross_attn) in enumerate(zip(reversed(in_out[1:]), *reversed_layer_params)):
            layer_cond_dim = cond_dim if layer_cross_attn else None

            self.ups.append(nn.ModuleList([
                ResnetBlock(dim_out * 2, dim_in, cond_dim = layer_cond_dim, time_cond_dim = time_cond_dim, groups = groups),
                nn.ModuleList([ResnetBlock(dim_in, dim_in, groups = groups) for _ in range(layer_num_resnet_blocks)]),
                TransformerBlock(dim = dim_in, heads = attn_heads, dim_head = attn_dim_head, ff_mult = ff_mult) if layer_attn else nn.Identity(),
                Upsample(dim_in)
            ]))

        self.final_conv = nn.Sequential(
            ResnetBlock(dim, dim, groups = resnet_groups[0]),
            nn.Conv2d(dim, self.channels_out, 1)
        )

    # if the current settings for the unet are not correct
    # for cascading DDPM, then reinit the unet with the right settings
    def cast_model_parameters(
        self,
        *,
        lowres_cond,
        text_embed_dim,
        channels,
        channels_out,
        cond_on_text
    ):
        if lowres_cond == self.lowres_cond and \
            channels == self.channels and \
            cond_on_text == self.cond_on_text and \
            text_embed_dim == self._locals['text_embed_dim'] and \
            channels_out == self.channels_out:
            return self

        updated_kwargs = dict(
            lowres_cond = lowres_cond,
            text_embed_dim = text_embed_dim,
            channels = channels,
            channels_out = channels_out,
            cond_on_text = cond_on_text
        )

        return self.__class__(**{**self._locals, **updated_kwargs})

    def forward_with_cond_scale(
        self,
        *args,
        cond_scale = 1.,
        **kwargs
    ):
        logits = self.forward(*args, **kwargs)

        if cond_scale == 1:
            return logits

        null_logits = self.forward(*args, cond_drop_prob = 1., **kwargs)
        return null_logits + (logits - null_logits) * cond_scale

    def forward(
        self,
        x,
        time,
        *,
        lowres_cond_img = None,
        lowres_noise_times = None,
        text_embeds = None,
        text_mask = None,
        cond_drop_prob = 0.
    ):
        batch_size, device = x.shape[0], x.device

        # add low resolution conditioning, if present

        assert not (self.lowres_cond and not exists(lowres_cond_img)), 'low resolution conditioning image must be present'
        assert not (self.lowres_cond and not exists(lowres_noise_times)), 'low resolution conditioning noise time must be present'

        if exists(lowres_cond_img):
            x = torch.cat((x, lowres_cond_img), dim = 1)

        # initial convolution

        x = self.init_conv(x)

        # time conditioning

        time_hiddens = self.to_time_hiddens(time)

        # add the time conditioning for the noised lowres conditioning, if needed

        if exists(self.to_lowres_time_hiddens):
            lowres_time_hiddens = self.to_lowres_time_hiddens(lowres_noise_times)
            time_hiddens = torch.cat((time_hiddens, lowres_time_hiddens), dim = -1)

        # derive time tokens

        time_tokens = self.to_time_tokens(time_hiddens)
        t = self.to_time_cond(time_hiddens)

        # conditional dropout

        text_keep_mask = prob_mask_like((batch_size,), 1 - cond_drop_prob, device = device)

        text_keep_mask = rearrange(text_keep_mask, 'b -> b 1 1')

        # take care of text encodings (optional)

        text_tokens = None

        if exists(text_embeds) and self.cond_on_text:
            text_tokens = self.text_to_cond(text_embeds)

            text_tokens = text_tokens[:, :self.max_text_len]

            text_tokens_len = text_tokens.shape[1]
            remainder = self.max_text_len - text_tokens_len

            if remainder > 0:
                text_tokens = F.pad(text_tokens, (0, 0, 0, remainder))

            if exists(text_mask):
                if remainder > 0:
                    text_mask = F.pad(text_mask, (0, remainder), value = False)

                text_mask = rearrange(text_mask, 'b n -> b n 1')
                text_keep_mask = text_mask & text_keep_mask

            null_text_embed = self.null_text_embed.to(text_tokens.dtype) # for some reason pytorch AMP not working

            text_tokens = torch.where(
                text_keep_mask,
                text_tokens,
                null_text_embed
            )

            if exists(self.attn_pool):
                text_tokens = self.attn_pool(text_tokens)

        # main conditioning tokens (c)

        c = time_tokens if not exists(text_tokens) else torch.cat((time_tokens, text_tokens), dim = -2)

        # normalize conditioning tokens

        c = self.norm_cond(c)

        # go through the layers of the unet, down and up

        hiddens = []

        for init_block, resnet_blocks, attn_block, downsample in self.downs:
            x = init_block(x, c, t)

            for resnet_block in resnet_blocks:
                x = resnet_block(x)

            x = attn_block(x)

            hiddens.append(x)
            x = downsample(x)

        x = self.mid_block1(x, c, t)

        if exists(self.mid_attn):
            x = self.mid_attn(x)

        x = self.mid_block2(x, c, t)

        for init_block, resnet_blocks, attn_block, upsample in self.ups:
            x = torch.cat((x, hiddens.pop()), dim=1)
            x = init_block(x, c, t)

            for resnet_block in resnet_blocks:
                x = resnet_block(x)

            x = attn_block(x)
            x = upsample(x)

        return self.final_conv(x)

# predefined unets, with configs lining up with hyperparameters in appendix of paper

class BaseUnet64(Unet):
    def __init__(self, *args, **kwargs):
        kwargs.update(dict(
            dim = 512,
            dim_mults = (1, 2, 3, 4),
            num_resnet_blocks = 3,
            layer_attns = (False, True, True, True),
            layer_cross_attns = (False, True, True, True),
            attn_heads = 8,
            ff_mult = 2.,
        ))
        super().__init__(*args, **kwargs)

class SRUnet256(Unet):
    def __init__(self, *args, **kwargs):
        kwargs.update(dict(
            dim = 128,
            dim_mults = (1, 2, 4, 8),
            num_resnet_blocks = (2, 4, 8, 8),
            layer_attns = (False, False, False, True),
            layer_cross_attns = (False, False, False, True),
            attn_heads = 8,
            ff_mult = 2.,
        ))
        super().__init__(*args, **kwargs)

class SRUnet1024(Unet):
    def __init__(self, *args, **kwargs):
        kwargs.update(dict(
            dim = 128,
            dim_mults = (1, 2, 4, 8),
            num_resnet_blocks = (2, 4, 8, 8),
            layer_attns = False,
            layer_cross_attns = (False, False, False, True),
            attn_heads = 8,
            ff_mult = 2.,
        ))
        super().__init__(*args, **kwargs)

# main imagen ddpm class, which is a cascading DDPM from Ho et al.

class Imagen(nn.Module):
    def __init__(
        self,
        unets,
        *,
        image_sizes,                                # for cascading ddpm, image size at each stage
        text_encoder_name = DEFAULT_T5_NAME,
        text_embed_dim = None,
        channels = 3,
        timesteps = 1000,
        cond_drop_prob = 0.1,
        loss_type = 'l2',
        beta_schedules = 'cosine',
        lowres_sample_noise_level = 0.2,            # in the paper, they present a new trick where they noise the lowres conditioning image, and at sample time, fix it to a certain level (0.1 or 0.3) - the unets are also made to be conditioned on this noise level
        condition_on_text = True,
        learned_variance = True,
        vb_loss_weight = 0.001,
        auto_normalize_img = True,                  # whether to take care of normalizing the image from [0, 1] to [-1, 1] and back automatically - you can turn this off if you want to pass in the [-1, 1] ranged image yourself from the dataloader
        continuous_times = False,
        dynamic_thresholding_percentile = 0.9       # unsure what this was based on perusal of paper
    ):
        super().__init__()

        # loss

        if loss_type == 'l1':
            loss_fn = F.l1_loss
        elif loss_type == 'l2':
            loss_fn = F.mse_loss
        elif loss_type == 'huber':
            loss_fn = F.smooth_l1_loss
        else:
            raise NotImplementedError()

        self.loss_type = loss_type
        self.loss_fn = loss_fn

        # conditioning hparams

        self.condition_on_text = condition_on_text
        self.unconditional = not condition_on_text

        # channels

        self.channels = channels

        # automatically take care of ensuring that first unet is unconditional
        # while the rest of the unets are conditioned on the low resolution image produced by previous unet

        unets = cast_tuple(unets)
        num_unets = len(unets)

        # determine noise schedules per unet

        timesteps = cast_tuple(timesteps, num_unets)
        beta_schedules = cast_tuple(beta_schedules, num_unets)

        noise_scheduler_klass = GaussianDiffusion if not continuous_times else GaussianDiffusionContinuousTimes
        self.noise_schedulers = nn.ModuleList([])

        for timestep, beta_schedule in zip(timesteps, beta_schedules):
            noise_scheduler = noise_scheduler_klass(beta_schedule = beta_schedule, timesteps = timestep)
            self.noise_schedulers.append(noise_scheduler)

        # whether to use learned variance, defaults to True for the first unet in the cascade, as in paper

        learned_variance = pad_tuple_to_length(cast_tuple(learned_variance), num_unets, fillvalue = False)
        self.learned_variance = learned_variance
        self.vb_loss_weight = vb_loss_weight

        # get text encoder

        self.text_encoder_name = text_encoder_name
        #self.text_embed_dim = get_encoded_dim(text_encoder_name)
        self.text_embed_dim = text_embed_dim if text_embed_dim is not None else get_encoded_dim(self.text_encoder_name)
        # construct unets

        self.unets = nn.ModuleList([])

        for ind, (one_unet, one_unet_learned_var) in enumerate(zip(unets, learned_variance)):
            assert isinstance(one_unet, Unet)
            is_first = ind == 0

            unet_channels_out = self.channels * (1 if not one_unet_learned_var else 2)

            one_unet = one_unet.cast_model_parameters(
                lowres_cond = not is_first,
                cond_on_text = self.condition_on_text,
                text_embed_dim = self.text_embed_dim if self.condition_on_text else None,
                channels = self.channels,
                channels_out = unet_channels_out
            )

            self.unets.append(one_unet)

        # unet image sizes

        assert num_unets == len(image_sizes), f'you did not supply the correct number of u-nets ({len(self.unets)}) for resolutions {image_sizes}'
        self.image_sizes = cast_tuple(image_sizes)
        self.sample_channels = cast_tuple(self.channels, num_unets)

        # cascading ddpm related stuff

        lowres_conditions = tuple(map(lambda t: t.lowres_cond, self.unets))
        assert lowres_conditions == (False, *((True,) * (num_unets - 1))), 'the first unet must be unconditioned (by low resolution image), and the rest of the unets must have `lowres_cond` set to True'

        self.lowres_sample_noise_level = lowres_sample_noise_level

        # classifier free guidance

        self.cond_drop_prob = cond_drop_prob
        self.can_classifier_guidance = cond_drop_prob > 0.

        # normalize and unnormalize image functions

        self.normalize_img = normalize_neg_one_to_one if auto_normalize_img else identity
        self.unnormalize_img = unnormalize_zero_to_one if auto_normalize_img else identity

        # dynamic thresholding

        self.dynamic_thresholding_percentile = dynamic_thresholding_percentile

    def get_unet(self, unet_number):
        assert 0 < unet_number <= len(self.unets)
        index = unet_number - 1
        return self.unets[index]

    @contextmanager
    def one_unet_in_gpu(self, unet_number = None, unet = None):
        assert exists(unet_number) ^ exists(unet)

        if exists(unet_number):
            unet = self.get_unet(unet_number)

        self.cuda()

        devices = [module_device(unet) for unet in self.unets]
        self.unets.cpu()
        unet.cuda()

        yield

        for unet, device in zip(self.unets, devices):
            unet.to(device)

    def p_mean_variance(self, unet, x, t, *, noise_scheduler, text_embeds = None, text_mask = None, lowres_cond_img = None, lowres_noise_times = None, clip_denoised = True, learned_variance = False, cond_scale = 1., model_output = None):
        assert not (cond_scale != 1. and not self.can_classifier_guidance), 'imagen was not trained with conditional dropout, and thus one cannot use classifier free guidance (cond_scale anything other than 1)'

        pred = default(model_output, lambda: unet.forward_with_cond_scale(x, t, text_embeds = text_embeds, text_mask = text_mask, cond_scale = cond_scale, lowres_cond_img = lowres_cond_img, lowres_noise_times = lowres_noise_times))

        if learned_variance:
            pred, var_interp_frac_unnormalized = pred.chunk(2, dim = 1)

        x_recon = noise_scheduler.predict_start_from_noise(x, t = t, noise = pred)

        if clip_denoised:
            # following pseudocode in appendix
            # s is the dynamic threshold, determined by percentile of absolute values of reconstructed sample per batch element

            s = torch.quantile(
                rearrange(x_recon, 'b ... -> b (...)').abs(),
                self.dynamic_thresholding_percentile,
                dim = -1
            )

            s.clamp_(min = 1.)
            s = s.view(-1, *((1,) * (x_recon.ndim - 1)))
            x_recon = x_recon.clamp(-s, s) / s

        model_mean, posterior_variance, posterior_log_variance = noise_scheduler.q_posterior(x_start=x_recon, x_t=x, t=t)

        if learned_variance:
            posterior_log_variance = noise_scheduler.get_learned_posterior_log_variance(var_interp_frac_unnormalized, x_t = x, t = t)
            posterior_variance = posterior_log_variance.exp()

        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, unet, x, t, *, noise_scheduler, text_embeds = None, text_mask = None, cond_scale = 1., lowres_cond_img = None, lowres_noise_times = None,  learned_variance = False, clip_denoised = True, repeat_noise = False):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(unet, x = x, t = t, noise_scheduler = noise_scheduler, text_embeds = text_embeds, text_mask = text_mask, cond_scale = cond_scale, lowres_cond_img = lowres_cond_img, lowres_noise_times = lowres_noise_times, clip_denoised = clip_denoised, learned_variance = learned_variance)
        noise = noise_like(x.shape, device, repeat_noise)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, unet, shape, *, noise_scheduler, learned_variance = False, clip_denoised = True, lowres_cond_img = None, lowres_noise_times = None, text_embeds = None, text_mask = None, cond_scale = 1):
        device = noise_scheduler.betas.device

        batch = shape[0]
        img = torch.randn(shape, device = device)
        lowres_cond_img = maybe(self.normalize_img)(lowres_cond_img)

        num_timesteps = noise_scheduler.num_timesteps

        for i in tqdm(reversed(range(0, num_timesteps)), desc = 'sampling loop time step', total = num_timesteps):
            img = self.p_sample(
                unet,
                img,
                torch.full((batch,), i, device = device, dtype = torch.long),
                text_embeds = text_embeds,
                text_mask = text_mask,
                cond_scale = cond_scale,
                lowres_cond_img = lowres_cond_img,
                lowres_noise_times = lowres_noise_times,
                noise_scheduler = noise_scheduler,
                learned_variance = learned_variance,
                clip_denoised = clip_denoised
            )

        unnormalize_img = self.unnormalize_img(img)
        return unnormalize_img

    @torch.no_grad()
    @eval_decorator
    def sample(
        self,
        texts: List[str] = None,
        text_masks = None,
        text_embeds = None,
        batch_size = 1,
        cond_scale = 1.,
        lowres_sample_noise_level = None,
        stop_at_unet_number = None,
        return_pil_images = False,
        device = None
    ):
        device = default(device, lambda: next(self.parameters()).device)

        if exists(texts) and not exists(text_embeds) and not self.unconditional:
            text_embeds, text_masks = t5_encode_text(texts, name = self.text_encoder_name)
            text_embeds, text_masks = map(lambda t: t.to(device), (text_embeds, text_masks))

        if not self.unconditional:
            batch_size = text_embeds.shape[0]

        assert not (self.condition_on_text and not exists(text_embeds)), 'text or text encodings must be passed into imagen if specified'
        assert not (not self.condition_on_text and exists(text_embeds)), 'imagen specified not to be conditioned on text, yet it is presented'
        assert not (exists(text_embeds) and text_embeds.shape[-1] != self.text_embed_dim), f'invalid text embedding dimension being passed in (should be {self.text_embed_dim})'

        img = None
        is_cuda = next(self.parameters()).is_cuda
        device = next(self.parameters()).device

        lowres_sample_noise_level = default(lowres_sample_noise_level, self.lowres_sample_noise_level)

        for unet_number, unet, channel, image_size, learned_variance, noise_scheduler in tqdm(zip(range(1, len(self.unets) + 1), self.unets, self.sample_channels, self.image_sizes, self.learned_variance, self.noise_schedulers)):

            context = self.one_unet_in_gpu(unet = unet) if is_cuda else null_context()

            with context:
                lowres_cond_img = lowres_noise_times = None
                shape = (batch_size, channel, image_size, image_size)

                if unet.lowres_cond:
                    lowres_noise_times = noise_scheduler.get_times(batch_size, lowres_sample_noise_level)

                    lowres_cond_img = resize_image_to(img, image_size)
                    lowres_cond_img = noise_scheduler.q_sample(x_start = lowres_cond_img, t = lowres_noise_times, noise = torch.randn_like(lowres_cond_img))

                shape = (batch_size, self.channels, image_size, image_size)

                img = self.p_sample_loop(
                    unet,
                    shape,
                    text_embeds = text_embeds,
                    text_mask = text_masks,
                    cond_scale = cond_scale,
                    learned_variance = learned_variance,
                    clip_denoised = True,
                    lowres_cond_img = lowres_cond_img,
                    lowres_noise_times = lowres_noise_times,
                    noise_scheduler = noise_scheduler
                )

            if exists(stop_at_unet_number) and stop_at_unet_number == unet_number:
                break

        if not return_pil_images:
            return img

        pil_images = list(map(T.ToPILImage(), img.unbind(dim = 0)))
        return pil_images # now you have a bunch of pillow images you can just .save(/where/ever/you/want.png)

    def p_losses(self, unet, x_start, times, *, noise_scheduler, lowres_cond_img = None, lowres_aug_times = None, text_embeds = None, text_mask = None, noise = None, learned_variance = False, clip_denoised = False):
        noise = default(noise, lambda: torch.randn_like(x_start))

        # normalize to [-1, 1]

        x_start = self.normalize_img(x_start)
        lowres_cond_img = maybe(self.normalize_img)(lowres_cond_img)

        # get x_t

        x_noisy = noise_scheduler.q_sample(x_start = x_start, t = times, noise = noise)

        # also noise the lowres conditioning image
        # at sample time, they then fix the noise level of 0.1 - 0.3

        lowres_cond_img_noisy = None
        if exists(lowres_cond_img):
            lowres_aug_times = default(lowres_aug_times, times)
            lowres_cond_img_noisy = noise_scheduler.q_sample(x_start = lowres_cond_img, t = lowres_aug_times, noise = torch.randn_like(lowres_cond_img))

        # get prediction

        model_output = unet(
            x_noisy,
            times,
            text_embeds = text_embeds,
            text_mask = text_mask,
            lowres_noise_times = lowres_aug_times,
            lowres_cond_img = lowres_cond_img_noisy,
            cond_drop_prob = self.cond_drop_prob,
        )

        if learned_variance:
            pred, _ = model_output.chunk(2, dim = 1)
        else:
            pred = model_output

        loss = self.loss_fn(pred, noise)

        if not learned_variance:
            # return simple loss if not using learned variance
            return loss

        # most of the code below is transcribed from
        # https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/diffusion_utils_2.py
        # the Improved DDPM paper then further modified it so that the mean is detached (shown a couple lines before), and weighted to be smaller than the l1 or l2 "simple" loss
        # it is questionable whether this is really needed, looking at some of the figures in the paper, but may as well stay faithful to their implementation

        # if learning the variance, also include the extra weight kl loss

        true_mean, _, true_log_variance_clipped = noise_scheduler.q_posterior(x_start = x_start, x_t = x_noisy, t = times)
        model_mean, _, model_log_variance = self.p_mean_variance(unet, x = x_noisy, t = times, noise_scheduler = noise_scheduler, clip_denoised = clip_denoised, learned_variance = True, model_output = model_output)

        # kl loss with detached model predicted mean, for stability reasons as in paper

        detached_model_mean = model_mean.detach()

        kl = normal_kl(true_mean, true_log_variance_clipped, detached_model_mean, model_log_variance)
        kl = meanflat(kl) * NAT

        decoder_nll = -discretized_gaussian_log_likelihood(x_start, means = detached_model_mean, log_scales = 0.5 * model_log_variance)
        decoder_nll = meanflat(decoder_nll) * NAT

        # at the first timestep return the decoder NLL, otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))

        vb_losses = torch.where(times == 0, decoder_nll, kl)

        # weight the vb loss smaller, for stability, as in the paper (recommended 0.001)

        vb_loss = vb_losses.mean() * self.vb_loss_weight

        return loss + vb_loss

    def forward(
        self,
        image,
        texts: List[str] = None,
        text_embeds = None,
        text_masks = None,
        unet_number = None
    ):
        assert not (len(self.unets) > 1 and not exists(unet_number)), f'you must specify which unet you want trained, from a range of 1 to {len(self.unets)}, if you are training cascading DDPM (multiple unets)'
        unet_number = default(unet_number, 1)
        unet_index = unet_number - 1
        
        unet = self.get_unet(unet_number)

        noise_scheduler     = self.noise_schedulers[unet_index]
        target_image_size   = self.image_sizes[unet_index]
        prev_image_size     = self.image_sizes[unet_index - 1] if unet_index > 0 else None
        learned_variance    = self.learned_variance[unet_index]
        b, c, h, w, device, = *image.shape, image.device

        check_shape(image, 'b c h w', c = self.channels)
        assert h >= target_image_size and w >= target_image_size

        times = noise_scheduler.sample_random_times(b)

        if exists(texts) and not exists(text_embeds) and not self.unconditional:
            text_embeds, text_masks = t5_encode_text(texts, name = self.text_encoder_name)
            text_embeds, text_masks = map(lambda t: t.to(image.device), (text_embeds, text_masks))

        assert not (self.condition_on_text and not exists(text_embeds)), 'text or text encodings must be passed into decoder if specified'
        assert not (not self.condition_on_text and exists(text_embeds)), 'decoder specified not to be conditioned on text, yet it is presented'

        assert not (exists(text_embeds) and text_embeds.shape[-1] != self.text_embed_dim), f'invalid text embedding dimension being passed in (should be {self.text_embed_dim})'

        lowres_cond_img = lowres_aug_times = None
        if exists(prev_image_size):
            lowres_cond_img = resize_image_to(image, prev_image_size)
            lowres_cond_img = resize_image_to(lowres_cond_img, target_image_size)
            lowres_aug_times = noise_scheduler.sample_random_times(b)

        image = resize_image_to(image, target_image_size)

        return self.p_losses(unet, image, times, text_embeds = text_embeds, text_mask = text_masks, noise_scheduler = noise_scheduler, lowres_cond_img = lowres_cond_img, lowres_aug_times = lowres_aug_times, learned_variance = learned_variance)
