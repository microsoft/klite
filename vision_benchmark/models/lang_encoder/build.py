import os
import logging

from transformers import CLIPTokenizer, CLIPTokenizerFast
from transformers import AutoTokenizer

from .registry import lang_encoders
from .registry import is_lang_encoder

logger = logging.getLogger(__name__)


def build_lang_encoder(config_encoder, tokenizer, verbose, **kwargs):
    model_name = config_encoder['NAME']

    if model_name.endswith('pretrain'):
        model_name = 'pretrain'

    if not is_lang_encoder(model_name):
        raise ValueError(f'Unknown model: {model_name}')

    return lang_encoders(model_name)(config_encoder, tokenizer, verbose, **kwargs)


def build_tokenizer(config_encoder):
    tokenizer = None
    os.environ['TOKENIZERS_PARALLELISM'] = 'false' # 'true', avoid hanging

    def post_process_clip(text):
        text['input_ids'].squeeze_() # torch.Size([1, 77])
        text['attention_mask'].squeeze_() # torch.Size([1, 77])
        return text

    if config_encoder['TOKENIZER'] == 'clip':
        os.environ['TOKENIZERS_PARALLELISM'] = 'true'
        pretrained_tokenizer = config_encoder.get(
            'PRETRAINED_TOKENIZER', 'openai/clip-vit-base-patch32'
        )
        tokenizer = CLIPTokenizer.from_pretrained(pretrained_tokenizer)
        tokenizer.add_special_tokens({'cls_token': tokenizer.eos_token})
        tokenizer.post_process = post_process_clip
    elif config_encoder['TOKENIZER'] == 'clip-fast':
        pretrained_tokenizer = config_encoder.get(
            'PRETRAINED_TOKENIZER', 'openai/clip-vit-base-patch32'
        )
        tokenizer = CLIPTokenizerFast.from_pretrained(pretrained_tokenizer, from_slow=True)
    elif config_encoder['TOKENIZER'] == 'zcodepp':
        from .zcodepp import ZCodeppTokenizer
        tokenizer = ZCodeppTokenizer(config_encoder)
        tokenizer.post_process = lambda x: x
    else:
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        pretrained_tokenizer = config_encoder.get('PRETRAINED_TOKENIZER', '')
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_tokenizer
            if pretrained_tokenizer else config_encoder['TOKENIZER']
        )
        tokenizer.post_process = post_process_clip

    return tokenizer
