from collections import OrderedDict
import logging
import os
import copy

import torch
from torch import nn

from timm.models.layers import DropPath, trunc_normal_

from .registry import register_lang_encoder


from vision_benchmark.utils.comm import comm
from vision_benchmark.utils.utils import register_norm_module


logger = logging.getLogger(__name__)

@register_norm_module
class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        pdtype = x.dtype
        x = x.float()
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x.to(pdtype) + self.bias


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self,
                 d_model: int,
                 base_d_model: int,
                 n_head: int,
                 base_n_head: int,
                 attn_mask: torch.Tensor = None,
                 drop_path: float = 0.0,
                 standparam: bool = True):
        super().__init__()

        if standparam:
            self.attn = nn.MultiheadAttention(d_model, n_head)

        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def attention(self, x: torch.Tensor, key_padding_mask: torch.Tensor = None):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) \
            if self.attn_mask is not None else None

        q, k, v = x, x, x
        return self.attn(
            q, k, v,
            key_padding_mask=key_padding_mask,
            need_weights=False,
            attn_mask=self.attn_mask
        )[0]

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor = None):
        x = x + self.drop_path(self.attention(self.ln_1(x), key_padding_mask=key_padding_mask))
        x = x + self.drop_path(self.mlp(self.ln_2(x)))
        return x


class Transformer(nn.Module):
    def __init__(self,
                 context_length: int,
                 vocab_size: int,
                 width: int,
                 base_width: int,
                 layers: int,
                 heads: int,
                 base_heads: int,
                 drop_path: float = 0.0,
                 autogressive: bool = True,
                 standparam: bool = True,
                 key_padding_token: int = 0): # key_padding_token = 0 for clip tokenizer, 1 for xlm-roberta-base):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, width)
        self.key_padding_token = key_padding_token

        self.context_length = context_length
        self.positional_embedding = nn.Parameter(
            torch.empty(self.context_length, width)
        )

        self.width = width
        self.layers = layers
        self.autogressive = autogressive
        attn_mask = self.build_attention_mask() if autogressive else None
        dpr = [x.item() for x in torch.linspace(0, drop_path, layers)]  # stochastic depth decay rule
        self.resblocks = nn.ModuleList(
            [
                ResidualAttentionBlock(width, base_width, heads, base_heads, attn_mask, dpr[i], standparam=standparam)
                for i in range(layers)
            ]
        )

        self.ln_final = LayerNorm(width)

    @property
    def dim_out(self):
        return self.width

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def custom_init_weights(self, use_original_init=True):
        self.use_original_init = use_original_init
        logger.info('Custom init: {}'.format('original init' if self.use_original_init else 'muP init'))

        # 1. Initialize parameters by calling reset_parameters.
        # This is used for MuPMultiheadAttention.
        for name, module in self.named_modules():
            if hasattr(module, 'custom_reset_parameters'):
                logger.info(f'Reset parameters in {name}')
                module.custom_reset_parameters(use_original_init=use_original_init)

        # 2. Initialize parameters for pos embed and token embed.
        if self.use_original_init:
            # Original initialization.
            # Note: This is not SP init. We do not implement SP init here.
            custom_trunc_normal_ = trunc_normal_  # Note: This should be the same as torch.nn.init.trunc_normal_

        custom_trunc_normal_(self.positional_embedding, std=.02)
        # normal_(self.token_embedding, std=.02)
        custom_trunc_normal_(self.token_embedding.weight, std=.02)

        # 3. Initialize parameters recursively for other layers.
        self.apply(self._custom_init_weights)

    def _custom_init_weights(self, m):
        if self.use_original_init:
            custom_trunc_normal_ = trunc_normal_  # Note: This should be the same as torch.nn.init.trunc_normal_

        if isinstance(m, (nn.Linear, nn.Conv2d)):
            if comm.is_main_process():
                logger.info('=> init weight of Linear/Conv2d from trunc norm')
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                if comm.is_main_process():
                    logger.info('=> init bias of Linear/Conv2d to zeros')
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            # Note: Actually reset_parameters() in nn.LayerNorm and nn.BatchNorm2d will initialize weights as 1 and bias as 0.

    def load_pretrained(self, pretrained='', pretrained_layers=[], verbose=True):
        if os.path.isfile(pretrained):
            pretrained_dict = torch.load(pretrained, map_location='cpu')
            logging.info(f'=> loading pretrained model {pretrained}')
            model_dict = self.state_dict()
            stripped_key = lambda x: x[13:] if x.startswith('lang_encoder.') else x
            pretrained_dict = {
                stripped_key(k): v for k, v in pretrained_dict.items()
                if stripped_key(k) in model_dict.keys()
            }
            need_init_state_dict = {}
            for k, v in pretrained_dict.items():
                need_init = (
                    k.split('.')[0] in pretrained_layers
                    or pretrained_layers[0] == '*'
                )
                if need_init:
                    if verbose:
                        logger.info(f'=> init {k} from {pretrained}')

                    if 'positional_embedding' in k and v.size() != model_dict[k].size():
                        positional_embedding_pretrained = v
                        positional_embedding_current = model_dict[k]
                        L1, nH1 = positional_embedding_pretrained.size()
                        L2, nH2 = positional_embedding_current.size()
                        if nH1 != nH2:
                            logger.info(f"Error in loading {k}, passing")
                        else:
                            if L1 != L2:
                                logger.info(
                                    '=> load_pretrained: resized variant: {} to {}'
                                        .format((L1, nH1), (L2, nH2))
                                )

                                posemb = positional_embedding_pretrained.float()
                                posemb_grid = posemb.unsqueeze(dim=0).permute(0, 2, 1)
                                posemb_grid = torch.nn.functional.interpolate(posemb_grid, size=L2, mode='linear')
                                posemb_grid = posemb_grid.permute(0, 2, 1).squeeze(dim=0)
                                v = posemb_grid

                    need_init_state_dict[k] = v

            self.load_state_dict(need_init_state_dict, strict=False)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {
            'positional_embedding',
            'token_embedding',
        }

    def forward(self, input_ids, attention_mask=None):
        # import pdb; pdb.set_trace()
        input_ids = input_ids.cuda(non_blocking=True)
        key_padding_mask = (input_ids == self.key_padding_token) if not self.autogressive else None
        # a True value indicates that the corresponding key value will be ignored for the purpose of attention
        x = self.token_embedding(input_ids)  # [batch_size, n_ctx, d_model]
        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        for block in self.resblocks:
            x = block(x, key_padding_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_final(x)

        return {'last_hidden_state': x}


def create_encoder(config_encoder, tokenizer):
    standparam = config_encoder.get('STANDPARAM', True)

    if standparam:
        # Dummy values for muP parameters.
        base_width = config_encoder['WIDTH']
        base_heads = config_encoder['HEADS']
    else:
        base_width = config_encoder['BASE_WIDTH']
        base_heads = config_encoder['BASE_HEADS']

    transformer = Transformer(
        context_length=config_encoder['CONTEXT_LENGTH'],
        vocab_size=tokenizer.vocab_size,
        width=config_encoder['WIDTH'],
        base_width=base_width,
        layers=config_encoder['LAYERS'],
        heads=config_encoder['HEADS'],
        base_heads=base_heads,
        autogressive=config_encoder.get('AUTOGRESSIVE', True),
        key_padding_token=config_encoder.get('KEY_PADDING_TOKEN', 0),
        standparam=standparam
    )
    return transformer


@register_lang_encoder
def lang_encoder(config_encoder, tokenizer, verbose, **kwargs):
    standparam = config_encoder.get('STANDPARAM', True)

    if standparam:
        logger.info('Create model with standard parameterization')
        model = create_encoder(config_encoder, tokenizer)
        use_original_init = True

    # Initialize parameters.
    model.custom_init_weights(use_original_init=use_original_init)

    logger.info('Load model from pretrained checkpoint')
    if config_encoder.get('LOAD_PRETRAINED'):
        model.load_pretrained(config_encoder['PRETRAINED'], config_encoder.get('PRETRAINED_LAYERS', ['*']))

    return model
