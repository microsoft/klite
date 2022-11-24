from .registry import image_encoders
from .registry import is_image_encoder


def build_image_encoder(config_encoder, verbose, **kwargs):
    model_name = config_encoder['NAME']
    if model_name.startswith('cls_'):
        model_name = model_name[4:]

    if not is_image_encoder(model_name):
        raise ValueError(f'Unknown model: {model_name}')

    return image_encoders(model_name)(config_encoder, verbose, **kwargs)
