import torch
from transformers import T5Tokenizer, T5EncoderModel, AutoTokenizer, AutoModelForSeq2SeqLM

def exists(val):
    return val is not None

# config

MAX_LENGTH = 256

DEFAULT_T5_NAME = 'google/t5-v1_1-base'

T5_CONFIGS = {
    't5-small': {
        'src': 't5',
        'dim': 512
    },
    't5-base': {
        'src': 't5',
        'dim': 768
    },
    't5-large': {
        'src': 't5',
        'dim': 1024
    },
    'google/t5-v1_1-small': {
        'src': 'auto',
        'dim': 512
    },
    'google/t5-v1_1-base': {
        'src': 'auto',
        'dim': 768
    },
    'google/t5-v1_1-large': {
        'src': 'auto',
        'dim': 1024
    }
}

# singleton globals

def get_klass(name):
    assert name in T5_CONFIGS
    config = T5_CONFIGS[name]
    src = config.get('src')

    if src == 't5':
        return T5Tokenizer, T5EncoderModel
    elif src == 'auto':
        return AutoTokenizer, AutoModelForSeq2SeqLM
    else:
        raise ValueError(f'unknown source {src}')

def get_tokenizer(name):
    assert name in T5_CONFIGS
    tokenizer_klass, _ = get_klass(name)
    tokenizer = tokenizer_klass.from_pretrained(name)
    return tokenizer

def get_model(name):
    assert name in T5_CONFIGS
    _, model_klass = get_klass(name)
    model = model_klass.from_pretrained(name)
    return model

def get_model_and_tokenizer(name):
    global T5_CONFIGS
    assert name in T5_CONFIGS, f'{name} model is not found in the configuration'
    config = T5_CONFIGS[name]

    if not 'model' in config:
        model = get_model(name)
        config['model'] = model

    if not 'tokenizer' in config:
        tokenizer = get_tokenizer(name)
        config['tokenizer'] = tokenizer

    return config['model'], config['tokenizer']

def get_encoded_dim(name):
    assert name in T5_CONFIGS, f'{name} model is not found in configuration'
    return T5_CONFIGS[name]['dim']

# encoding text

def t5_encode_text(texts, name = 't5-small'):
    t5, tokenizer = get_model_and_tokenizer(name)

    if torch.cuda.is_available():
        t5 = t5.cuda()

    device = next(t5.parameters()).device

    encoded = tokenizer.batch_encode_plus(
        texts,
        return_tensors = "pt",
        padding = 'longest',
        max_length = MAX_LENGTH,
        truncation = True
    )

    input_ids = encoded.input_ids.to(device)
    attn_mask = encoded.attention_mask.to(device)

    t5.eval()
    with torch.no_grad():
        output = t5(input_ids = input_ids, attention_mask = attn_mask) # too lazy to figure out how to make it work without decoder inputs
        encoded_text = output.last_hidden_state.detach()

    return encoded_text, attn_mask.bool()
