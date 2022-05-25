import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

def exists(val):
    return val is not None

# config

MAX_LENGTH = 256

T5_CONFIGS = {
    't5-small': {
        'dim': 512
    },
    't5-large': {
        'dim': 1024
    }
}

# singleton globals

def get_tokenizer(name):
    assert name in T5_CONFIGS
    tokenizer = T5Tokenizer.from_pretrained(name)
    return tokenizer

def get_model(name):
    assert name in T5_CONFIGS
    model = T5ForConditionalGeneration.from_pretrained(name)
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
        output = t5(input_ids = input_ids, attention_mask = attn_mask, decoder_input_ids = input_ids[:, :1]) # too lazy to figure out how to make it work without decoder inputs
        encoded_text = output.encoder_last_hidden_state.detach()

    return encoded_text, attn_mask.bool()
