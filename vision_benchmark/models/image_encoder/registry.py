_image_encoders = {}


def register_image_encoder(fn):
    module_name_split = fn.__module__.split('.')
    model_name = module_name_split[-1]

    _image_encoders[model_name] = fn

    return fn


def image_encoders(model_name):
    return _image_encoders[model_name]


def is_image_encoder(model_name):
    return model_name in _image_encoders
