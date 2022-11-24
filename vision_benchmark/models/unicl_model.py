import pathlib
import tempfile
import logging
import os
import copy

import torch
from torch import nn

from timm.models.layers import trunc_normal_

from .image_encoder import build_image_encoder
from .lang_encoder import build_lang_encoder, build_tokenizer


logger = logging.getLogger(__name__)


class UniCLModel(nn.Module):
    def __init__(self, config: dict):
        super().__init__()

        self.conf_lang_encoder = config['LANG_ENCODER']
        self.tokenizer = build_tokenizer(self.conf_lang_encoder)

        self.lang_encoder = build_lang_encoder(self.conf_lang_encoder, self.tokenizer, config['VERBOSE'])

        dim_projection = config['UNICL_MODEL']['DIM_PROJECTION']
        if hasattr(self.lang_encoder, 'dim_out'):
            dim_out = self.lang_encoder.dim_out
        else:
            with torch.no_grad():
                dim_out = self.lang_encoder(
                    torch.zeros(1,1).type(torch.LongTensor)
                )['last_hidden_state'].size(2)

        self.lang_projection = nn.Parameter(torch.empty(dim_out, dim_projection))

        self.conf_image_encoder = config['IMAGE_ENCODER']
        self.image_encoder = build_image_encoder(self.conf_image_encoder, config['VERBOSE'])

        self.image_projection = nn.Parameter(
            torch.empty(self.image_encoder.dim_out, dim_projection)
        )

        self.logit_scale = nn.Parameter(torch.ones([]))

    def custom_init_weights(self, use_original_init=True):
        self.use_original_init = use_original_init
        logger.info('Custom init: {}'.format('original init' if self.use_original_init else 'muP init'))

        if self.use_original_init:
            # Original initialization. 
            # Note: This is not SP init. We do not implement SP init here.
            custom_trunc_normal_ = trunc_normal_  # Note: This should be the same as torch.nn.init.trunc_normal_


        custom_trunc_normal_(self.lang_projection, std=.02)
        custom_trunc_normal_(self.image_projection, std=.02)

    def _convert_old_weights(self, model_dict):
        model_dict_updated = {}
        for k, v in model_dict.items():
            if k.startswith('visual.'):
                model_dict_updated['image_encoder.'+k[7:]] = v
            elif k.startswith('text.'):
                model_dict_updated['lang_encoder.'+k[5:]] = v
            elif k == 'vision_projection':
                model_dict_updated['image_projection'] = v
            elif k == 'text_projection':
                model_dict_updated['lang_projection'] = v
            else:
                model_dict_updated[k] = v

        return model_dict_updated

    def from_pretrained(self, pretrained='', pretrained_layers=[], verbose=True):
        if not os.path.isfile(pretrained):
            logger.warning(f'=> Pretrained model ({pretrained}) is not a file, skip init weight')
            return

        pretrained_dict = torch.load(pretrained, map_location='cpu')
        logger.info(f'=> Loading pretrained model {pretrained}')
        pretrained_dict = self._convert_old_weights(pretrained_dict)
        pretrained_dict = {
            k: v for k, v in pretrained_dict.items()
        }
        need_init_state_dict = {}
        image_encoder_state_dict = {}
        for k, v in pretrained_dict.items():
            need_init = (
                k.split('.')[0] in pretrained_layers
                or pretrained_layers[0] == '*'
            )

            if need_init:
                if k.startswith('image_encoder.'):
                    image_encoder_state_dict[k] = v
                else:
                    if verbose:
                        logger.info(f'=> init {k} from {pretrained}')

                    need_init_state_dict[k] = v
        self.image_encoder.from_state_dict(image_encoder_state_dict, ['*'], verbose)
        self.load_state_dict(need_init_state_dict, strict=False)

    @torch.jit.ignore
    def no_weight_decay(self):
        no_weight_decay = {'logit_scale'}
        if hasattr(self.lang_encoder, 'no_weight_decay'):
            for k in self.lang_encoder.no_weight_decay():
                no_weight_decay.add('lang_encoder.'+k)

        if hasattr(self.image_encoder, 'no_weight_decay'):
            for k in self.visual.no_weight_decay():
                no_weight_decay.add('image_encoder.'+k)

        return no_weight_decay

    @property
    def dtype(self):
        return self.logit_scale.dtype

    def encode_image(self, image, norm=True):
        x = self.image_encoder.forward_features(image)
        x = x @ self.image_projection

        if norm:
            x = x / x.norm(dim=-1, keepdim=True)

        return x

    def encode_text(self, input_ids, norm=True):
        # import pdb; pdb.set_trace()
        text = {'input_ids': input_ids, 'attention_mask': None}
        x = self.lang_encoder(**text)

        # x = self.lang_encoder(text)
        x = x['last_hidden_state']

        if self.conf_lang_encoder['TOKENIZER'] == 'clip':
            x = x[torch.arange(x.size(0)), text['input_ids'].argmax(dim=-1)]
        else:
            x = x[:, 0]

        x = x @ self.lang_projection

        if norm:
            x = x / x.norm(dim=-1, keepdim=True)

        return x

    def forward(self, image, text):
        features_image = self.encode_image(image)
        features_text = self.encode_text(text)

        # cosine similarity as logits
        T = self.logit_scale.exp()

        return features_image, features_text, T


def create_model(config):
    model = UniCLModel(config)
    return model



def get_zeroshot_model(config, **kwargs):
    standparam = config['UNICL_MODEL'].get('STANDPARAM', True)

    if standparam:
        logger.info('Create model with standard parameterization')
        model = create_model(config)

        use_original_init = True

    # Initialize other parameters.
    model.custom_init_weights(use_original_init=use_original_init)

    if config['UNICL_MODEL']['LOAD_PRETRAINED']:
        pretrained_path = config['UNICL_MODEL']['PRETRAINED']
        model.from_pretrained(pretrained_path, config['UNICL_MODEL']['PRETRAINED_LAYERS'], config['VERBOSE'])

    return model
