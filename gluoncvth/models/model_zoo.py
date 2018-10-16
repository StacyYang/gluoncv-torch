# pylint: disable=wildcard-import, unused-wildcard-import

from .resnet import *

__all__ = ['get_model']


def get_model(name, **kwargs):
    """Returns a pre-defined model by name

    Parameters
    ----------
    name : str
        Name of the model.
    pretrained : bool
        Whether to load the pretrained weights for model.
    root : str, default '~/.gluoncvth/models'
        Location for keeping the model parameters.

    Returns
    -------
    Module:
        The model.
    """
    models = {
        'resnet18': resnet18,
        'resnet34': resnet34,
        'resnet50': resnet50,
        'resnet101': resnet101,
        'resnet152': resnet152,
        }
    name = name.lower()
    if name not in models:
        raise ValueError('%s\n\t%s' % (str(e), '\n\t'.join(sorted(models.keys()))))
    net = models[name](**kwargs)
    return net
