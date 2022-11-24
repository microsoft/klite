import pathlib
import tempfile
from collections import OrderedDict
from typing import Tuple, Union
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from timm.models.layers import DropPath, trunc_normal_

from .image_encoder import build_image_encoder
from .text_encoder import build_text_encoder
from .text_encoder import build_tokenizer
from data.imagenet import IMAGENET_CLASSES, IMAGENET_DEFAULT_TEMPLATES

logger = logging.getLogger(__name__)


class KLITEModel(nn.Module):
    def __init__(self, config: dict,):
        super().__init__()

        self.conf_lang_encoder = config['MODEL']['TEXT_ENCODER']
        self.tokenizer = build_tokenizer(self.conf_lang_encoder)

        self.text = build_text_encoder(self.conf_lang_encoder, self.tokenizer, config['VERBOSE'])

        dim_projection = config['MODEL']['DIM_PROJECTION']
        if hasattr(self.text, 'dim_out'):
            dim_out = self.text.dim_out
        else:
            with torch.no_grad():
                dim_out = self.text(
                    torch.zeros(1,1).type(torch.LongTensor)
                )['last_hidden_state'].size(2)

        self.text_projection = nn.Parameter(torch.empty(dim_out, dim_projection))

        self.conf_image_encoder = config['MODEL']['IMAGE_ENCODER']
        self.visual = build_image_encoder(self.conf_image_encoder)

        self.vision_projection = nn.Parameter(
            torch.empty(self.visual.dim_out, dim_projection)
        )

        self.logit_scale = nn.Parameter(torch.ones([]))

        trunc_normal_(self.text_projection, std=.02)
        trunc_normal_(self.vision_projection, std=.02)

    def _convert_old_weights(self, model_dict):
        model_dict_updated = {}
        for k, v in model_dict.items():
            if k.startswith('visual.'):
                model_dict_updated['image_encoder.'+k[7:]] = v
            elif k.startswith('text.'):
                model_dict_updated['lang_encoder.'+k[5:]] = v
            elif k == 'vision_projection':
                model_dict_updated['vision_projection'] = v
            elif k == 'text_projection':
                model_dict_updated['text_projection'] = v
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
        model_dict = self.state_dict()
        pretrained_dict = {
            k: v for k, v in pretrained_dict.items()
            if k in model_dict.keys()
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
        self.visual.from_state_dict(image_encoder_state_dict, ['*'], verbose)
        self.load_state_dict(need_init_state_dict, strict=False)

    @torch.jit.ignore
    def no_weight_decay(self):
        no_weight_decay = {'logit_scale'}
        if hasattr(self.text, 'no_weight_decay'):
            for k in self.text.no_weight_decay():
                no_weight_decay.add('lang_encoder.'+k)

        if hasattr(self.visual, 'no_weight_decay'):
            for k in self.visual.no_weight_decay():
                no_weight_decay.add('image_encoder.'+k)

        return no_weight_decay

    @property
    def dtype(self):
        return self.logit_scale.dtype

    def get_imnet_embeddings(self, use_knowledge=False):
        import json
        from nltk.tokenize import word_tokenize
        if use_knowledge:
            wiki_path = 'vision_benchmark/resources/knowledge/external/'
            wiki_tsv_path = os.path.join(wiki_path,  'imagenet-1k_knowledge.tsv') 
            wiki_anwser_list = json.load(open(wiki_tsv_path, encoding='utf-8'))

            count_has_wiki_knowledge = 0
            wiki_dict = {}
            for k2v in wiki_anwser_list:
                wiki_dict[ k2v['classname'] ] = k2v['def_wiki']   
                if k2v['def_wiki']:
                    count_has_wiki_knowledge += 1
            logger.info(f'coverage is {count_has_wiki_knowledge} / {len(wiki_dict)}')

            gpt3_tsv_path = os.path.join('vision_benchmark/resources/knowledge/gpt3/', 'GPT3_imagenet-1k.tsv') 
            gpt3_anwser_list = json.load(open(gpt3_tsv_path, encoding='utf-8'))

            gpt3_dict = {}
            for k2v in gpt3_anwser_list:
                gpt3_dict[ k2v['classname'] ] = k2v['gpt3']
            NUM_GPT3_ITEMS = 5
            wiki_count, gpt3_count = 0, 0

        templates = IMAGENET_DEFAULT_TEMPLATES
        clss_embeddings = []
        for clss in IMAGENET_CLASSES:
            knowledge_text_list = []
            if use_knowledge:
                if clss in wiki_dict:
                    knowledge_text_list.append(wiki_dict[clss])
                    wiki_count += 1
                if clss in gpt3_dict:
                    for knowledge_text in gpt3_dict[clss][:NUM_GPT3_ITEMS]:
                        knowledge_text_list.append(knowledge_text)
                        gpt3_count += 1 

            knowledge_text_list_aug = []
            for knowledge_text in knowledge_text_list:
                knowledge_text = f' ; {clss} , ' + knowledge_text if knowledge_text is not None else ''
                knowledge_text = ' ' + ' '.join(word_tokenize(knowledge_text))
                knowledge_text_list_aug.append(knowledge_text)

            if len(knowledge_text_list_aug) == 0:
                txts = [template.format(clss) for template in templates ]
            else:
                txts = [template.format(clss) + knowledge_text for knowledge_text in knowledge_text_list_aug for template in templates ]

            # txts = [template.format(clss) for template in templates]
            
            tokens = self.tokenizer(
                txts, padding='max_length', truncation=True, max_length=77, return_tensors='pt'
            )                
            tokens = {key:val.cuda() for key,val in tokens.items()}

            clss_embedding = self.encode_text(tokens)
            clss_embedding = clss_embedding.mean(dim=0)
            clss_embedding /= clss_embedding.norm()
            clss_embeddings.append(clss_embedding)
        imnet_text_embeddings = torch.stack(clss_embeddings, dim=0)
        if use_knowledge: logger.info(f'=> Knowledge source count | knowledge_count: {wiki_count} | gpt3_count {gpt3_count} ')
        return imnet_text_embeddings

    def encode_image(self, image, norm=True):
        x = self.visual.forward_features(image)
        x = x @ self.vision_projection

        if norm:
            x = x / x.norm(dim=-1, keepdim=True)

        return x

    def encode_text(self, text, norm=True):
        x = self.text(**text)
        x = x['last_hidden_state']

        if self.conf_lang_encoder['TOKENIZER'] == 'clip':
            x = x[torch.arange(x.size(0)), text['input_ids'].argmax(dim=-1)]
        else:
            x = x[:, 0]

        x = x @ self.text_projection

        if norm:
            x = x / x.norm(dim=-1, keepdim=True)

        return x

    def forward(self, image, text):
        features_image = self.encode_image(image)
        features_text = self.encode_text(text)

        # cosine similarity as logits
        T = self.logit_scale.exp()

        return features_image, features_text, T


def build_klite_model(config, **kwargs):
    model = KLITEModel(config)
    if config['MODEL']['PRETRAINED'] != '':
        pretrained_path = config['MODEL']['PRETRAINED']
        from ..Utils.Utils import is_valid_url, download_file
        if is_valid_url(pretrained_path):
            with tempfile.TemporaryDirectory() as tmp_path:
                file_local_path = pathlib.Path(tmp_path) / 'base_model.pt'
                download_file(pretrained_path, file_local_path)
                model.from_pretrained(str(file_local_path), config['MODEL']['PRETRAINED_LAYERS'], config['VERBOSE'])
        else:
            model.from_pretrained(pretrained_path, config['MODEL']['PRETRAINED_LAYERS'], config['VERBOSE'])

    return model
