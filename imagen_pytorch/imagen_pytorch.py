import math
import copy
from typing import List
from tqdm import tqdm
from functools import partial, wraps
from contextlib import contextmanager
from collections import namedtuple
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn, einsum
from torch.special import expm1
import torchvision.transforms as T

from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange, Reduce
from einops_exts import rearrange_many, repeat_many, check_shape
from einops_exts.torch import EinopsToAndFrom

from resize_right import resize

from imagen_pytorch.t5 import t5_encode_text, get_encoded_dim, DEFAULT_T5_NAME

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
    return d() if callable(d) else d

def cast_tuple(val, length = None):
    if isinstance(val, list):
        val = tuple(val)

    output = val if isinstance(val, tuple) else ((val,) * default(length, 1))

    if exists(length):
        assert len(output) == length

    return output

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

# helper classes

class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

# tensor helpers

def log(t, eps: float = 1e-12):
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
    def __init__(
        self,
        *,
        noise_schedule,
        timesteps
    ):
        super().__init__()

        if noise_schedule == "cosine":
            betas = cosine_beta_schedule(timesteps)
        elif noise_schedule == "linear":
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

    def get_times(self, batch_size, noise_level, *, device):
        return torch.full((batch_size,), int(self.num_timesteps * noise_level), device = device, dtype = torch.long)

    def sample_random_times(self, batch_size, *, device):
        return torch.randint(0, self.num_timesteps, (batch_size,), device = device, dtype = torch.long)

    def get_condition(self, times):
        return times

    def get_sampling_timesteps(self, batch, *, device):
        time_transitions = []

        for i in reversed(range(self.num_timesteps)):
            time_transitions.append((torch.full((batch,), i, device = device, dtype = torch.long), None))

        return time_transitions

    def q_posterior(self, x_start, x_t, t, **kwargs):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def q_sample(self, x_start, t, noise = None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        noised = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        alphas_cumprod = extract(self.alphas_cumprod, t, t.shape)
        log_snr = -log(1. / alphas_cumprod - 1)

        return noised, log_snr

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

# gaussian diffusion with continuous time helper functions and classes
# large part of this was thanks to @crowsonkb at https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/utils.py

@torch.jit.script
def beta_linear_log_snr(t):
    return -torch.log(expm1(1e-4 + 10 * (t ** 2)))

@torch.jit.script
def alpha_cosine_log_snr(t, s: float = 0.008):
    return -log((torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** -2) - 1, eps = 1e-5) # not sure if this accounts for beta being clipped to 0.999 in discrete version

def log_snr_to_alpha_sigma(log_snr):
    return torch.sqrt(torch.sigmoid(log_snr)), torch.sqrt(torch.sigmoid(-log_snr))

class GaussianDiffusionContinuousTimes(nn.Module):
    def __init__(self, *, noise_schedule, timesteps = 1000):
        super().__init__()
        if noise_schedule == 'linear':
            self.log_snr = beta_linear_log_snr
        elif noise_schedule == "cosine":
            self.log_snr = alpha_cosine_log_snr
        else:
            raise ValueError(f'invalid noise schedule {noise_schedule}')

        self.num_timesteps = timesteps

    def get_times(self, batch_size, noise_level, *, device):
        return torch.full((batch_size,), noise_level, device = device, dtype = torch.long)

    def sample_random_times(self, batch_size, max_thres = 0.999, *, device):
        return torch.zeros((batch_size,), device = device).float().uniform_(0, max_thres)

    def get_condition(self, times):
        return maybe(self.log_snr)(times)

    def get_sampling_timesteps(self, batch, *, device):
        times = torch.linspace(1., 0., self.num_timesteps + 1, device = device)
        times = repeat(times, 't -> b t', b = batch)
        times = torch.stack((times[:, :-1], times[:, 1:]), dim = 0)
        times = times.unbind(dim = -1)
        return times

    def q_posterior(self, x_start, x_t, t, *, t_next = None):
        t_next = default(t_next, lambda: (t - 1. / self.num_timesteps).clamp(min = 0.))

        """ https://openreview.net/attachment?id=2LdBqxc1Yv&name=supplementary_material """
        log_snr = self.log_snr(t)
        log_snr_next = self.log_snr(t_next)
        log_snr, log_snr_next = map(partial(right_pad_dims_to, x_t), (log_snr, log_snr_next))

        alpha, sigma = log_snr_to_alpha_sigma(log_snr)
        alpha_next, sigma_next = log_snr_to_alpha_sigma(log_snr_next)

        # c - as defined near eq 33
        c = -expm1(log_snr - log_snr_next)
        posterior_mean = alpha_next * (x_t * (1 - c) / alpha + c * x_start)

        # following (eq. 33)
        posterior_variance = (sigma_next ** 2) * c
        posterior_log_variance_clipped = log(posterior_variance, eps = 1e-20)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def q_sample(self, x_start, t, noise = None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        log_snr = self.log_snr(t)
        log_snr_padded_dim = right_pad_dims_to(x_start, log_snr)
        alpha, sigma =  log_snr_to_alpha_sigma(log_snr_padded_dim)
        return alpha * x_start + sigma * noise, log_snr

    def predict_start_from_noise(self, x_t, t, noise):
        log_snr = self.log_snr(t)
        log_snr = right_pad_dims_to(x_t, log_snr)
        alpha, sigma = log_snr_to_alpha_sigma(log_snr)
        return (x_t - sigma * noise) / alpha.clamp(min = 1e-8)

# norms and residuals

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer('beta', torch.zeros(dim))

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

class Always():
    def __init__(self, val):
        self.val = val

    def __call__(self, *args, **kwargs):
        return self.val

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class Parallel(nn.Module):
    def __init__(self, *fns):
        super().__init__()
        self.fns = nn.ModuleList(fns)

    def forward(self, x):
        outputs = [fn(x) for fn in self.fns]
        return sum(outputs)

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

        attn = sim.softmax(dim = -1, dtype = torch.float32)

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
        context_dim = None
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm = LayerNorm(dim)

        self.null_kv = nn.Parameter(torch.randn(2, dim_head))
        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, dim_head * 2, bias = False)

        self.to_context = nn.Sequential(nn.LayerNorm(context_dim), nn.Linear(context_dim, dim_head * 2)) if exists(context_dim) else None

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias = False),
            LayerNorm(dim)
        )

    def forward(self, x, context = None, mask = None, attn_bias = None):
        b, n, device = *x.shape[:2], x.device

        x = self.norm(x)
        q, k, v = (self.to_q(x), *self.to_kv(x).chunk(2, dim = -1))

        q = rearrange(q, 'b n (h d) -> b h n d', h = self.heads)
        q = q * self.scale

        # add null key / value for classifier free guidance in prior net

        nk, nv = repeat_many(self.null_kv.unbind(dim = -2), 'd -> b 1 d', b = b)
        k = torch.cat((nk, k), dim = -2)
        v = torch.cat((nv, v), dim = -2)

        # add text conditioning, if present

        if exists(context):
            assert exists(self.to_context)
            ck, cv = self.to_context(context).chunk(2, dim = -1)
            k = torch.cat((ck, k), dim = -2)
            v = torch.cat((cv, v), dim = -2)

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

        attn = sim.softmax(dim = -1, dtype = torch.float32)

        # aggregate values

        out = einsum('b h i j, b j d -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

# decoder

def Upsample(dim, dim_out = None):
    dim_out = default(dim_out, dim)

    return nn.Sequential(
        nn.Upsample(scale_factor = 2, mode = 'nearest'),
        nn.Conv2d(dim, dim_out, 3, padding = 1)
    )

def Downsample(dim, dim_out = None):
    dim_out = default(dim_out, dim)
    return nn.Conv2d(dim, dim_out, 4, 2, 1)

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

class LearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with learned sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered

class Block(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
        groups = 8,
        norm = True
    ):
        super().__init__()
        self.groupnorm = nn.GroupNorm(groups, dim) if norm else Identity()
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
        groups = 8,
        linear_attn = False,
        use_gca = False,
        squeeze_excite = False
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
            attn_klass = CrossAttention if not linear_attn else LinearCrossAttention

            self.cross_attn = EinopsToAndFrom(
                'b c h w',
                'b (h w) c',
                attn_klass(
                    dim = dim_out,
                    context_dim = cond_dim
                )
            )

        self.block1 = Block(dim, dim_out, groups = groups)
        self.block2 = Block(dim_out, dim_out, groups = groups)

        self.gca = GlobalContext(dim_in = dim_out, dim_out = dim_out) if use_gca else Always(1)

        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else Identity()


    def forward(self, x, time_emb = None, cond = None):

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

        h = h * self.gca(h)

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
        self.norm_context = LayerNorm(context_dim) if norm_context else Identity()

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

        attn = sim.softmax(dim = -1, dtype = torch.float32)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class LinearCrossAttention(CrossAttention):
    def forward(self, x, context, mask = None):
        b, n, device = *x.shape[:2], x.device

        x = self.norm(x)
        context = self.norm_context(context)

        q, k, v = (self.to_q(x), *self.to_kv(context).chunk(2, dim = -1))

        q, k, v = rearrange_many((q, k, v), 'b n (h d) -> (b h) n d', h = self.heads)

        # add null key / value for classifier free guidance in prior net

        nk, nv = repeat_many(self.null_kv.unbind(dim = -2), 'd -> (b h) 1 d', h = self.heads,  b = b)

        k = torch.cat((nk, k), dim = -2)
        v = torch.cat((nv, v), dim = -2)

        # masking

        max_neg_value = -torch.finfo(x.dtype).max

        if exists(mask):
            mask = F.pad(mask, (1, 0), value = True)
            mask = rearrange(mask, 'b n -> b n 1')
            k = k.masked_fill(~mask, max_neg_value)
            v = v.masked_fill(~mask, 0.)

        # linear attention

        q = q.softmax(dim = -1)
        k = k.softmax(dim = -2)

        q = q * self.scale

        context = einsum('b n d, b n e -> b d e', k, v)
        out = einsum('b n d, b d e -> b n e', q, context)
        out = rearrange(out, '(b h) n d -> b n (h d)', h = self.heads)
        return self.to_out(out)

class LinearAttention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 32,
        heads = 8,
        dropout = 0.05,
        context_dim = None
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads
        self.norm = ChanLayerNorm(dim)

        self.nonlin = nn.SiLU()

        self.to_q = nn.Sequential(
            nn.Dropout(dropout),
            nn.Conv2d(dim, inner_dim, 1, bias = False),
            nn.Conv2d(inner_dim, inner_dim, 3, bias = False, padding = 1, groups = inner_dim)
        )

        self.to_k = nn.Sequential(
            nn.Dropout(dropout),
            nn.Conv2d(dim, inner_dim, 1, bias = False),
            nn.Conv2d(inner_dim, inner_dim, 3, bias = False, padding = 1, groups = inner_dim)
        )

        self.to_v = nn.Sequential(
            nn.Dropout(dropout),
            nn.Conv2d(dim, inner_dim, 1, bias = False),
            nn.Conv2d(inner_dim, inner_dim, 3, bias = False, padding = 1, groups = inner_dim)
        )

        self.to_context = nn.Sequential(nn.LayerNorm(context_dim), nn.Linear(context_dim, inner_dim * 2, bias = False)) if exists(context_dim) else None

        self.to_out = nn.Sequential(
            nn.Conv2d(inner_dim, dim, 1, bias = False),
            ChanLayerNorm(dim)
        )

    def forward(self, fmap, context = None):
        h, x, y = self.heads, *fmap.shape[-2:]

        fmap = self.norm(fmap)
        q, k, v = map(lambda fn: fn(fmap), (self.to_q, self.to_k, self.to_v))
        q, k, v = rearrange_many((q, k, v), 'b (h c) x y -> (b h) (x y) c', h = h)

        if exists(context):
            assert exists(self.to_context)
            ck, cv = self.to_context(context).chunk(2, dim = -1)
            ck, cv = rearrange_many((ck, cv), 'b n (h d) -> (b h) n d', h = h)
            k = torch.cat((k, ck), dim = -2)
            v = torch.cat((v, cv), dim = -2)

        q = q.softmax(dim = -1)
        k = k.softmax(dim = -2)

        q = q * self.scale

        context = einsum('b n d, b n e -> b d e', k, v)
        out = einsum('b n d, b d e -> b n e', q, context)
        out = rearrange(out, '(b h) (x y) d -> b (h d) x y', h = h, x = x, y = y)

        out = self.nonlin(out)
        return self.to_out(out)

class GlobalContext(nn.Module):
    """ basically a superior form of squeeze-excitation that is attention-esque """

    def __init__(
        self,
        *,
        dim_in,
        dim_out
    ):
        super().__init__()
        self.to_k = nn.Conv2d(dim_in, 1, 1)
        hidden_dim = max(3, dim_out // 2)

        self.net = nn.Sequential(
            nn.Conv2d(dim_in, hidden_dim, 1),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, dim_out, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        context = self.to_k(x)
        x, context = rearrange_many((x, context), 'b n ... -> b n (...)')
        out = einsum('b i n, b c n -> b c i', context.softmax(dim = -1), x)
        out = rearrange(out, '... -> ... 1')
        return self.net(out)

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
        ff_mult = 2,
        context_dim = None
    ):
        super().__init__()
        self.attn = EinopsToAndFrom('b c h w', 'b (h w) c', Attention(dim = dim, heads = heads, dim_head = dim_head, context_dim = context_dim))
        self.ff = ChanFeedForward(dim = dim, mult = ff_mult)

    def forward(self, x, context = None):
        x = self.attn(x, context = context) + x
        x = self.ff(x) + x
        return x

class LinearAttentionTransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        *,
        heads = 8,
        dim_head = 32,
        ff_mult = 2,
        context_dim = None
    ):
        super().__init__()
        self.attn = LinearAttention(dim = dim, heads = heads, dim_head = dim_head, context_dim = context_dim)
        self.ff = ChanFeedForward(dim = dim, mult = ff_mult)

    def forward(self, x, context = None):
        x = self.attn(x, context = context) + x
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
        learned_sinu_pos_emb = True,
        learned_sinu_pos_emb_dim = 16,
        out_dim = None,
        dim_mults=(1, 2, 4, 8),
        cond_images_channels = 0,
        channels = 3,
        channels_out = None,
        attn_dim_head = 64,
        attn_heads = 8,
        ff_mult = 2.,
        lowres_cond = False,                # for cascading diffusion - https://cascaded-diffusion.github.io/
        layer_attns = True,
        layer_attns_add_text_cond = True,   # whether to condition the self-attention blocks with the text embeddings, as described in Appendix D.3.1
        attend_at_middle = True,            # whether to have a layer of attention at the bottleneck (can turn off for higher resolution in cascading DDPM, before bringing in efficient attention)
        layer_cross_attns = True,
        use_linear_attn = False,
        use_linear_cross_attn = False,
        cond_on_text = True,
        max_text_len = 256,
        init_dim = None,
        init_conv_kernel_size = 7,
        resnet_groups = 8,
        init_cross_embed_kernel_sizes = (3, 7, 15),
        cross_embed_downsample = False,
        cross_embed_downsample_kernel_sizes = (2, 4),
        attn_pool_text = True,
        attn_pool_num_latents = 32,
        dropout = 0.,
        memory_efficient = False,
        init_conv_to_final_conv_residual = False,
        use_global_context_attn = True,
        scale_skip_connection = True,
        final_resnet_block = True,
        final_conv_kernel_size = 3
    ):
        super().__init__()

        # guide researchers

        assert attn_heads > 1, 'you need to have more than 1 attention head, ideally at least 4 or 8'

        if dim < 128:
            print('The base dimension of your u-net should ideally be no smaller than 128, as recommended by a professional DDPM trainer https://nonint.com/2022/05/04/friends-dont-let-friends-train-small-diffusion-models/')

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

        # optional image conditioning

        self.has_cond_image = cond_images_channels > 0
        self.cond_images_channels = cond_images_channels

        init_channels += cond_images_channels

        # initial convolution

        self.init_conv = CrossEmbedLayer(init_channels, dim_out = init_dim, kernel_sizes = init_cross_embed_kernel_sizes, stride = 1)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # time conditioning

        cond_dim = default(cond_dim, dim)
        time_cond_dim = dim * 4 * (2 if lowres_cond else 1)

        # embedding time for discrete gaussian diffusion or log(snr) noise for continuous version

        self.learned_sinu_pos_emb = learned_sinu_pos_emb

        if learned_sinu_pos_emb:
            sinu_pos_emb = LearnedSinusoidalPosEmb(learned_sinu_pos_emb_dim)
            sinu_pos_emb_input_dim = learned_sinu_pos_emb_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim)
            sinu_pos_emb_input_dim = dim

        self.to_time_hiddens = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(sinu_pos_emb_input_dim, time_cond_dim),
            nn.SiLU()
        )

        self.to_time_cond = nn.Sequential(
            nn.Linear(time_cond_dim, time_cond_dim)
        )

        # project to time tokens as well as time hiddens

        self.to_time_tokens = nn.Sequential(
            nn.Linear(time_cond_dim, cond_dim * num_time_tokens),
            Rearrange('b (r d) -> b r d', r = num_time_tokens)
        )

        # low res aug noise conditioning

        self.lowres_cond = lowres_cond

        if lowres_cond:
            self.to_lowres_time_hiddens = nn.Sequential(
                LearnedSinusoidalPosEmb(learned_sinu_pos_emb_dim),
                nn.Linear(learned_sinu_pos_emb_dim + 1, time_cond_dim),
                nn.SiLU()
            )

            self.to_lowres_time_cond = nn.Sequential(
                nn.Linear(time_cond_dim, time_cond_dim)
            )

            self.to_lowres_time_tokens = nn.Sequential(
                nn.Linear(time_cond_dim, cond_dim * num_time_tokens),
                Rearrange('b (r d) -> b r d', r = num_time_tokens)
            )

        # normalizations

        self.norm_cond = nn.LayerNorm(cond_dim)

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

        self.max_text_len = max_text_len

        self.null_text_embed = nn.Parameter(torch.randn(1, max_text_len, cond_dim))
        self.null_text_hidden = nn.Parameter(torch.randn(1, time_cond_dim))

        # for non-attention based text conditioning at all points in the network where time is also conditioned

        self.to_text_non_attn_cond = None

        if cond_on_text:
            self.to_text_non_attn_cond = nn.Sequential(
                nn.LayerNorm(cond_dim),
                nn.Linear(cond_dim, time_cond_dim),
                nn.SiLU(),
                nn.Linear(time_cond_dim, time_cond_dim)
            )

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

        # initial resnet block (for memory efficient unet)

        self.init_resnet_block = ResnetBlock(init_dim, init_dim, time_cond_dim = time_cond_dim, groups = resnet_groups[0], use_gca = use_global_context_attn) if memory_efficient else None

        # scale for resnet skip connections

        self.skip_connect_scale = 1. if not scale_skip_connection else (2 ** -0.5)

        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        layer_params = [num_resnet_blocks, resnet_groups, layer_attns, layer_cross_attns]
        reversed_layer_params = list(map(reversed, layer_params))

        # downsampling layers

        skip_connect_dims = [] # keep track of skip connection dimensions

        for ind, ((dim_in, dim_out), layer_num_resnet_blocks, groups, layer_attn, layer_cross_attn) in enumerate(zip(in_out, *layer_params)):
            is_last = ind >= (num_resolutions - 1)

            layer_use_linear_cross_attn = not layer_cross_attn and use_linear_cross_attn
            layer_cond_dim = cond_dim if layer_cross_attn or layer_use_linear_cross_attn else None

            transformer_block_klass = TransformerBlock if layer_attn else (LinearAttentionTransformerBlock if use_linear_attn else Identity)

            current_dim = dim_in

            # whether to pre-downsample, from memory efficient unet

            pre_downsample = None

            if memory_efficient:
                pre_downsample = downsample_klass(dim_in, dim_out)
                current_dim = dim_out

            skip_connect_dims.append(current_dim)

            # whether to do post-downsample, for non-memory efficient unet

            post_downsample = None
            if not memory_efficient:
                post_downsample = downsample_klass(current_dim, dim_out) if not is_last else Parallel(nn.Conv2d(dim_in, dim_out, 3, padding = 1), nn.Conv2d(dim_in, dim_out, 1))

            self.downs.append(nn.ModuleList([
                pre_downsample,
                ResnetBlock(current_dim, current_dim, cond_dim = layer_cond_dim, linear_attn = layer_use_linear_cross_attn, time_cond_dim = time_cond_dim, groups = groups),
                nn.ModuleList([ResnetBlock(current_dim, current_dim, time_cond_dim = time_cond_dim, groups = groups, use_gca = use_global_context_attn) for _ in range(layer_num_resnet_blocks)]),
                transformer_block_klass(dim = current_dim, heads = attn_heads, dim_head = attn_dim_head, ff_mult = ff_mult, context_dim = cond_dim),
                post_downsample
            ]))

        # middle layers

        mid_dim = dims[-1]

        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, cond_dim = cond_dim, time_cond_dim = time_cond_dim, groups = resnet_groups[-1])
        self.mid_attn = EinopsToAndFrom('b c h w', 'b (h w) c', Residual(Attention(mid_dim, **attn_kwargs))) if attend_at_middle else None
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, cond_dim = cond_dim, time_cond_dim = time_cond_dim, groups = resnet_groups[-1])

        # upsampling layers

        for ind, ((dim_in, dim_out), layer_num_resnet_blocks, groups, layer_attn, layer_cross_attn) in enumerate(zip(reversed(in_out), *reversed_layer_params)):
            is_last = ind == (len(in_out) - 1)
            layer_use_linear_cross_attn = not layer_cross_attn and use_linear_cross_attn
            layer_cond_dim = cond_dim if layer_cross_attn or layer_use_linear_cross_attn else None
            transformer_block_klass = TransformerBlock if layer_attn else (LinearAttentionTransformerBlock if use_linear_attn else Identity)

            skip_connect_dim = skip_connect_dims.pop()

            self.ups.append(nn.ModuleList([
                ResnetBlock(dim_out + skip_connect_dim, dim_out, cond_dim = layer_cond_dim, linear_attn = layer_use_linear_cross_attn, time_cond_dim = time_cond_dim, groups = groups),
                nn.ModuleList([ResnetBlock(dim_out + skip_connect_dim, dim_out, time_cond_dim = time_cond_dim, groups = groups, use_gca = use_global_context_attn) for _ in range(layer_num_resnet_blocks)]),
                transformer_block_klass(dim = dim_out, heads = attn_heads, dim_head = attn_dim_head, ff_mult = ff_mult, context_dim = cond_dim),
                Upsample(dim_out, dim_in) if not is_last or memory_efficient else Identity()
            ]))

        # whether to do a final residual from initial conv to the final resnet block out

        self.init_conv_to_final_conv_residual = init_conv_to_final_conv_residual
        final_conv_dim = dim * (2 if init_conv_to_final_conv_residual else 1)

        # final optional resnet block and convolution out

        self.final_res_block = ResnetBlock(final_conv_dim, dim, time_cond_dim = time_cond_dim, groups = resnet_groups[0], use_gca = True) if final_resnet_block else None

        final_conv_dim_in = dim if final_resnet_block else final_conv_dim
        self.final_conv = nn.Conv2d(final_conv_dim_in, self.channels_out, final_conv_kernel_size, padding = final_conv_kernel_size // 2)

    # if the current settings for the unet are not correct
    # for cascading DDPM, then reinit the unet with the right settings
    def cast_model_parameters(
        self,
        *,
        lowres_cond,
        text_embed_dim,
        channels,
        channels_out,
        cond_on_text,
        learned_sinu_pos_emb
    ):
        if lowres_cond == self.lowres_cond and \
            channels == self.channels and \
            cond_on_text == self.cond_on_text and \
            text_embed_dim == self._locals['text_embed_dim'] and \
            learned_sinu_pos_emb == self.learned_sinu_pos_emb and \
            channels_out == self.channels_out:
            return self

        updated_kwargs = dict(
            lowres_cond = lowres_cond,
            text_embed_dim = text_embed_dim,
            channels = channels,
            channels_out = channels_out,
            cond_on_text = cond_on_text,
            learned_sinu_pos_emb = learned_sinu_pos_emb
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
        cond_images = None,
        cond_drop_prob = 0.
    ):
        batch_size, device = x.shape[0], x.device

        # add low resolution conditioning, if present

        assert not (self.lowres_cond and not exists(lowres_cond_img)), 'low resolution conditioning image must be present'
        assert not (self.lowres_cond and not exists(lowres_noise_times)), 'low resolution conditioning noise time must be present'

        if exists(lowres_cond_img):
            x = torch.cat((x, lowres_cond_img), dim = 1)

        # condition on input image

        assert not (self.has_cond_image ^ exists(cond_images)), 'you either requested to condition on an image on the unet, but the conditioning image is not supplied, or vice versa'

        if exists(cond_images):
            assert cond_images.shape[1] == self.cond_images_channels, 'the number of channels on the conditioning image you are passing in does not match what you specified on initialiation of the unet'
            cond_images = resize_image_to(cond_images, x.shape[-1])
            x = torch.cat((cond_images, x), dim = 1)

        # initial convolution

        x = self.init_conv(x)

        # init conv residual

        if self.init_conv_to_final_conv_residual:
            init_conv_residual = x.clone()

        # time conditioning

        time_hiddens = self.to_time_hiddens(time)

        # derive time tokens

        time_tokens = self.to_time_tokens(time_hiddens)
        t = self.to_time_cond(time_hiddens)

        # add lowres time conditioning to time hiddens
        # and add lowres time tokens along sequence dimension for attention

        if self.lowres_cond:
            lowres_time_hiddens = self.to_lowres_time_hiddens(lowres_noise_times)
            lowres_time_tokens = self.to_lowres_time_tokens(lowres_time_hiddens)
            lowres_t = self.to_lowres_time_cond(lowres_time_hiddens)

            t = t + lowres_t
            time_tokens = torch.cat((time_tokens, lowres_time_tokens), dim = -2)

        # text conditioning

        text_tokens = None

        if exists(text_embeds) and self.cond_on_text:

            # conditional dropout

            text_keep_mask = prob_mask_like((batch_size,), 1 - cond_drop_prob, device = device)

            text_keep_mask_embed = rearrange(text_keep_mask, 'b -> b 1 1')
            text_keep_mask_hidden = rearrange(text_keep_mask, 'b -> b 1')

            # calculate text embeds

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
                text_keep_mask_embed = text_mask & text_keep_mask_embed

            null_text_embed = self.null_text_embed.to(text_tokens.dtype) # for some reason pytorch AMP not working

            text_tokens = torch.where(
                text_keep_mask_embed,
                text_tokens,
                null_text_embed
            )

            if exists(self.attn_pool):
                text_tokens = self.attn_pool(text_tokens)

            # extra non-attention conditioning by projecting and then summing text embeddings to time
            # termed as text hiddens

            mean_pooled_text_tokens = text_tokens.mean(dim = -2)

            text_hiddens = self.to_text_non_attn_cond(mean_pooled_text_tokens)

            null_text_hidden = self.null_text_hidden.to(t.dtype)

            text_hiddens = torch.where(
                text_keep_mask_hidden,
                text_hiddens,
                null_text_hidden
            )

            t = t + text_hiddens

        # main conditioning tokens (c)

        c = time_tokens if not exists(text_tokens) else torch.cat((time_tokens, text_tokens), dim = -2)

        # normalize conditioning tokens

        c = self.norm_cond(c)

        # initial resnet block (for memory efficient unet)

        if exists(self.init_resnet_block):
            x = self.init_resnet_block(x, t)

        # go through the layers of the unet, down and up

        hiddens = []

        for pre_downsample, init_block, resnet_blocks, attn_block, post_downsample in self.downs:
            if exists(pre_downsample):
                x = pre_downsample(x)

            x = init_block(x, t, c)

            for resnet_block in resnet_blocks:
                x = resnet_block(x, t)
                hiddens.append(x)

            x = attn_block(x, c)
            hiddens.append(x)

            if exists(post_downsample):
                x = post_downsample(x)

        x = self.mid_block1(x, t, c)

        if exists(self.mid_attn):
            x = self.mid_attn(x)

        x = self.mid_block2(x, t, c)

        add_skip_connection = lambda x: torch.cat((x, hiddens.pop() * self.skip_connect_scale), dim = 1)

        for init_block, resnet_blocks, attn_block, upsample in self.ups:
            x = add_skip_connection(x)
            x = init_block(x, t, c)

            for resnet_block in resnet_blocks:
                x = add_skip_connection(x)
                x = resnet_block(x, t)

            x = attn_block(x, c)
            x = upsample(x)

        # final top-most residual if needed

        if self.init_conv_to_final_conv_residual:
            x = torch.cat((x, init_conv_residual), dim = 1)

        if exists(self.final_res_block):
            x = self.final_res_block(x, t)

        return self.final_conv(x)

# predefined unets, with configs lining up with hyperparameters in appendix of paper

class BaseUnet64(Unet):
    def __init__(self, *args, **kwargs):
        default_kwargs = dict(
            dim = 512,
            dim_mults = (1, 2, 3, 4),
            num_resnet_blocks = 3,
            layer_attns = (False, True, True, True),
            layer_cross_attns = (False, True, True, True),
            attn_heads = 8,
            ff_mult = 2.,
            memory_efficient = False
        )
        super().__init__(*args, **{**default_kwargs, **kwargs})

class SRUnet256(Unet):
    def __init__(self, *args, **kwargs):
        default_kwargs = dict(
            dim = 128,
            dim_mults = (1, 2, 4, 8),
            num_resnet_blocks = (2, 4, 8, 8),
            layer_attns = (False, False, False, True),
            layer_cross_attns = (False, False, False, True),
            attn_heads = 8,
            ff_mult = 2.,
            memory_efficient = True
        )
        super().__init__(*args, **{**default_kwargs, **kwargs})

class SRUnet1024(Unet):
    def __init__(self, *args, **kwargs):
        default_kwargs = dict(
            dim = 128,
            dim_mults = (1, 2, 4, 8),
            num_resnet_blocks = (2, 4, 8, 8),
            layer_attns = False,
            layer_cross_attns = (False, False, False, True),
            attn_heads = 8,
            ff_mult = 2.,
            memory_efficient = True
        )
        super().__init__(*args, **{**default_kwargs, **kwargs})

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
        noise_schedules = 'cosine',
        pred_objectives = 'noise',
        lowres_noise_schedule = 'linear',
        lowres_sample_noise_level = 0.2,            # in the paper, they present a new trick where they noise the lowres conditioning image, and at sample time, fix it to a certain level (0.1 or 0.3) - the unets are also made to be conditioned on this noise level
        per_sample_random_aug_noise_level = False,  # unclear when conditioning on augmentation noise level, whether each batch element receives a random aug noise value - turning off due to @marunine's find
        condition_on_text = True,
        auto_normalize_img = True,                  # whether to take care of normalizing the image from [0, 1] to [-1, 1] and back automatically - you can turn this off if you want to pass in the [-1, 1] ranged image yourself from the dataloader
        continuous_times = True,
        p2_loss_weight_gamma = 0.5,                 # p2 loss weight, from https://arxiv.org/abs/2204.00227 - 0 is equivalent to weight of 1 across time
        p2_loss_weight_k = 1,
        dynamic_thresholding = True,
        dynamic_thresholding_percentile = 0.9,      # unsure what this was based on perusal of paper
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

        # make sure noise schedule defaults to 'cosine', 'cosine', and then 'linear' for rest of super-resoluting unets

        noise_schedules = cast_tuple(noise_schedules)
        noise_schedules = pad_tuple_to_length(noise_schedules, 2, 'cosine')
        noise_schedules = pad_tuple_to_length(noise_schedules, num_unets, 'linear')

        # construct noise schedulers

        noise_scheduler_klass = GaussianDiffusion if not continuous_times else GaussianDiffusionContinuousTimes
        self.noise_schedulers = nn.ModuleList([])

        for timestep, noise_schedule in zip(timesteps, noise_schedules):
            noise_scheduler = noise_scheduler_klass(noise_schedule = noise_schedule, timesteps = timestep)
            self.noise_schedulers.append(noise_scheduler)

        # lowres augmentation noise schedule

        self.lowres_noise_schedule = GaussianDiffusionContinuousTimes(noise_schedule = lowres_noise_schedule)

        # ddpm objectives - predicting noise by default

        self.pred_objectives = cast_tuple(pred_objectives, num_unets)

        # get text encoder

        self.text_encoder_name = text_encoder_name
        self.text_embed_dim = default(text_embed_dim, lambda: get_encoded_dim(text_encoder_name))

        # construct unets

        self.unets = nn.ModuleList([])

        for ind, one_unet in enumerate(unets):
            assert isinstance(one_unet, Unet)
            is_first = ind == 0

            one_unet = one_unet.cast_model_parameters(
                lowres_cond = not is_first,
                cond_on_text = self.condition_on_text,
                text_embed_dim = self.text_embed_dim if self.condition_on_text else None,
                channels = self.channels,
                channels_out = self.channels,
                learned_sinu_pos_emb = continuous_times
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

        # dynamic thresholding

        self.dynamic_thresholding = cast_tuple(dynamic_thresholding, num_unets)
        self.dynamic_thresholding_percentile = dynamic_thresholding_percentile

        # p2 loss weight

        self.p2_loss_weight_k = p2_loss_weight_k
        self.p2_loss_weight_gamma = cast_tuple(p2_loss_weight_gamma, num_unets)

        assert all([(gamma_value <= 2) for gamma_value in self.p2_loss_weight_gamma]), 'in paper, they noticed any gamma greater than 2 is harmful'

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

    def p_mean_variance(self, unet, x, t, *, noise_scheduler, text_embeds = None, text_mask = None, cond_images = None, lowres_cond_img = None, lowres_noise_times = None, cond_scale = 1., model_output = None, t_next = None, pred_objective = 'noise', dynamic_threshold = True):
        assert not (cond_scale != 1. and not self.can_classifier_guidance), 'imagen was not trained with conditional dropout, and thus one cannot use classifier free guidance (cond_scale anything other than 1)'

        pred = default(model_output, lambda: unet.forward_with_cond_scale(x, noise_scheduler.get_condition(t), text_embeds = text_embeds, text_mask = text_mask, cond_images = cond_images, cond_scale = cond_scale, lowres_cond_img = lowres_cond_img, lowres_noise_times = self.lowres_noise_schedule.get_condition(lowres_noise_times)))

        if pred_objective == 'noise':
            x_start = noise_scheduler.predict_start_from_noise(x, t = t, noise = pred)
        elif pred_objective == 'x_start':
            x_start = pred
        else:
            raise ValueError(f'unknown objective {pred_objective}')

        if dynamic_threshold:
            # following pseudocode in appendix
            # s is the dynamic threshold, determined by percentile of absolute values of reconstructed sample per batch element
            s = torch.quantile(
                rearrange(x_start, 'b ... -> b (...)').abs(),
                self.dynamic_thresholding_percentile,
                dim = -1
            )

            s.clamp_(min = 1.)
            s = right_pad_dims_to(x_start, s)
            x_start = x_start.clamp(-s, s) / s
        else:
            x_start.clamp_(-1., 1.)

        return noise_scheduler.q_posterior(x_start = x_start, x_t = x, t = t, t_next = t_next)

    @torch.no_grad()
    def p_sample(self, unet, x, t, *, noise_scheduler, t_next = None, text_embeds = None, text_mask = None, cond_images = None, cond_scale = 1., lowres_cond_img = None, lowres_noise_times = None, pred_objective = 'noise', dynamic_threshold = True):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(unet, x = x, t = t, t_next = t_next, noise_scheduler = noise_scheduler, text_embeds = text_embeds, text_mask = text_mask, cond_images = cond_images, cond_scale = cond_scale, lowres_cond_img = lowres_cond_img, lowres_noise_times = lowres_noise_times, pred_objective = pred_objective, dynamic_threshold = dynamic_threshold)
        noise = torch.randn_like(x)
        # no noise when t == 0
        is_last_sampling_timestep = (t_next == 0) if isinstance(noise_scheduler, GaussianDiffusionContinuousTimes) else (t == 0)
        nonzero_mask = (1 - is_last_sampling_timestep.float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, unet, shape, *, noise_scheduler, lowres_cond_img = None, lowres_noise_times = None, text_embeds = None, text_mask = None, cond_images = None, cond_scale = 1, pred_objective = 'noise', dynamic_threshold = True):
        device = self.device

        batch = shape[0]
        img = torch.randn(shape, device = device)
        lowres_cond_img = maybe(self.normalize_img)(lowres_cond_img)

        timesteps = noise_scheduler.get_sampling_timesteps(batch, device = device)

        for times, times_next in tqdm(timesteps, desc = 'sampling loop time step', total = len(timesteps)):
            img = self.p_sample(
                unet,
                img,
                times,
                t_next = times_next,
                text_embeds = text_embeds,
                text_mask = text_mask,
                cond_images = cond_images,
                cond_scale = cond_scale,
                lowres_cond_img = lowres_cond_img,
                lowres_noise_times = lowres_noise_times,
                noise_scheduler = noise_scheduler,
                pred_objective = pred_objective,
                dynamic_threshold = dynamic_threshold
            )

        img.clamp_(-1., 1.)
        unnormalize_img = self.unnormalize_img(img)
        return unnormalize_img

    @torch.no_grad()
    @eval_decorator
    def sample(
        self,
        texts: List[str] = None,
        text_masks = None,
        text_embeds = None,
        cond_images = None,
        batch_size = 1,
        cond_scale = 1.,
        lowres_sample_noise_level = None,
        stop_at_unet_number = None,
        return_all_unet_outputs = False,
        return_pil_images = False,
        device = None,
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

        outputs = []

        is_cuda = next(self.parameters()).is_cuda
        device = next(self.parameters()).device

        lowres_sample_noise_level = default(lowres_sample_noise_level, self.lowres_sample_noise_level)

        for unet_number, unet, channel, image_size, noise_scheduler, pred_objective, dynamic_threshold in tqdm(zip(range(1, len(self.unets) + 1), self.unets, self.sample_channels, self.image_sizes, self.noise_schedulers, self.pred_objectives, self.dynamic_thresholding)):

            context = self.one_unet_in_gpu(unet = unet) if is_cuda else null_context()

            with context:
                lowres_cond_img = lowres_noise_times = None
                shape = (batch_size, channel, image_size, image_size)

                if unet.lowres_cond:
                    lowres_noise_times = self.lowres_noise_schedule.get_times(batch_size, lowres_sample_noise_level, device = device)

                    lowres_cond_img = resize_image_to(img, image_size)
                    lowres_cond_img, _ = self.lowres_noise_schedule.q_sample(x_start = lowres_cond_img, t = lowres_noise_times, noise = torch.randn_like(lowres_cond_img))

                shape = (batch_size, self.channels, image_size, image_size)

                img = self.p_sample_loop(
                    unet,
                    shape,
                    text_embeds = text_embeds,
                    text_mask = text_masks,
                    cond_images = cond_images,
                    cond_scale = cond_scale,
                    lowres_cond_img = lowres_cond_img,
                    lowres_noise_times = lowres_noise_times,
                    noise_scheduler = noise_scheduler,
                    pred_objective = pred_objective,
                    dynamic_threshold = dynamic_threshold
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

    def p_losses(self, unet, x_start, times, *, noise_scheduler, lowres_cond_img = None, lowres_aug_times = None, text_embeds = None, text_mask = None, cond_images = None, noise = None, times_next = None, pred_objective = 'noise', p2_loss_weight_gamma = 0.):
        noise = default(noise, lambda: torch.randn_like(x_start))

        # normalize to [-1, 1]

        x_start = self.normalize_img(x_start)
        lowres_cond_img = maybe(self.normalize_img)(lowres_cond_img)

        # get x_t

        x_noisy, log_snr = noise_scheduler.q_sample(x_start = x_start, t = times, noise = noise)

        # also noise the lowres conditioning image
        # at sample time, they then fix the noise level of 0.1 - 0.3

        lowres_cond_img_noisy = None
        if exists(lowres_cond_img):
            lowres_aug_times = default(lowres_aug_times, times)
            lowres_cond_img_noisy, _ = self.lowres_noise_schedule.q_sample(x_start = lowres_cond_img, t = lowres_aug_times, noise = torch.randn_like(lowres_cond_img))

        # get prediction

        pred = unet.forward(
            x_noisy,
            noise_scheduler.get_condition(times),
            text_embeds = text_embeds,
            text_mask = text_mask,
            cond_images = cond_images,
            lowres_noise_times = self.lowres_noise_schedule.get_condition(lowres_aug_times),
            lowres_cond_img = lowres_cond_img_noisy,
            cond_drop_prob = self.cond_drop_prob,
        )

        # prediction objective

        if pred_objective == 'noise':
            target = noise
        elif pred_objective == 'x_start':
            target = x_start
        else:
            raise ValueError(f'unknown objective {pred_objective}')

        # losses

        losses = self.loss_fn(pred, target, reduction = 'none')
        losses = reduce(losses, 'b ... -> b', 'mean')

        # p2 loss reweighting

        if p2_loss_weight_gamma > 0:
            loss_weight = (self.p2_loss_weight_k + log_snr.exp()) ** -p2_loss_weight_gamma
            losses = losses * loss_weight

        return losses.mean()

    def forward(
        self,
        images,
        texts: List[str] = None,
        text_embeds = None,
        text_masks = None,
        unet_number = None,
        cond_images = None
    ):
        assert not (len(self.unets) > 1 and not exists(unet_number)), f'you must specify which unet you want trained, from a range of 1 to {len(self.unets)}, if you are training cascading DDPM (multiple unets)'
        unet_number = default(unet_number, 1)
        unet_index = unet_number - 1
        
        unet = self.get_unet(unet_number)

        noise_scheduler      = self.noise_schedulers[unet_index]
        p2_loss_weight_gamma = self.p2_loss_weight_gamma[unet_index]
        pred_objective       = self.pred_objectives[unet_index]
        target_image_size    = self.image_sizes[unet_index]
        prev_image_size      = self.image_sizes[unet_index - 1] if unet_index > 0 else None
        b, c, h, w, device,  = *images.shape, images.device

        check_shape(images, 'b c h w', c = self.channels)
        assert h >= target_image_size and w >= target_image_size

        times = noise_scheduler.sample_random_times(b, device = device)

        if exists(texts) and not exists(text_embeds) and not self.unconditional:
            assert len(texts) == len(images), 'number of text captions does not match up with the number of images given'

            text_embeds, text_masks = t5_encode_text(texts, name = self.text_encoder_name)
            text_embeds, text_masks = map(lambda t: t.to(images.device), (text_embeds, text_masks))

        assert not (self.condition_on_text and not exists(text_embeds)), 'text or text encodings must be passed into decoder if specified'
        assert not (not self.condition_on_text and exists(text_embeds)), 'decoder specified not to be conditioned on text, yet it is presented'

        assert not (exists(text_embeds) and text_embeds.shape[-1] != self.text_embed_dim), f'invalid text embedding dimension being passed in (should be {self.text_embed_dim})'

        lowres_cond_img = lowres_aug_times = None
        if exists(prev_image_size):
            lowres_cond_img = resize_image_to(images, prev_image_size)
            lowres_cond_img = resize_image_to(lowres_cond_img, target_image_size)

            if self.per_sample_random_aug_noise_level:
                lowres_aug_times = self.lowres_noise_schedule.sample_random_times(b, device = device)
            else:
                lowres_aug_time = self.lowres_noise_schedule.sample_random_times(1, device = device)
                lowres_aug_times = repeat(lowres_aug_time, '1 -> b', b = b)

        images = resize_image_to(images, target_image_size)

        return self.p_losses(unet, images, times, text_embeds = text_embeds, text_mask = text_masks, cond_images = cond_images, noise_scheduler = noise_scheduler, lowres_cond_img = lowres_cond_img, lowres_aug_times = lowres_aug_times, pred_objective = pred_objective, p2_loss_weight_gamma = p2_loss_weight_gamma)
