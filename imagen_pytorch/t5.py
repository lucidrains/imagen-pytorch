import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

def exists(val):
    return val is not None

# singleton globals

MODEL = None
TOKENIZER = None
T5_SMALL_EMBED_DIM = 512

def get_tokenizer():
    global TOKENIZER
    if not exists(TOKENIZER):
        TOKENIZER = T5Tokenizer.from_pretrained("t5-small")
    return TOKENIZER

def get_t5():
    global MODEL
    if not exists(MODEL):
        MODEL = T5ForConditionalGeneration.from_pretrained("t5-small")
        if torch.cuda.is_available():
            MODEL = MODEL.cuda()

    return MODEL

# encoding text

def t5_encode_text(texts):
    t5 = get_t5()
    tokenizer = get_tokenizer()

    input_ids = tokenizer.batch_encode_plus(texts, return_tensors = "pt", padding = True, truncation = True).input_ids
    input_ids = input_ids.to(next(t5.parameters()).device)

    t5.eval()
    with torch.no_grad():
        output = t5(input_ids = input_ids, decoder_input_ids = input_ids[:, :1]) # too lazy to figure out how to make it work without decoder inputs

    return output.encoder_last_hidden_state
