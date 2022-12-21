from .cifar10 import CIFAR10
from .mnist import MNIST
from .stl10 import STL10
from .svhn import SVHN
from .usps import USPS


ds_map = {
    'cifar10': CIFAR10, 
    'mnist': MNIST, 
    'stl10' : STL10, 
    'svhn': SVHN,
    'usps' : USPS, 
}

SUPPORTED_DATASETS = list(ds_map.keys())

def get_dataset(name):
    name = name.lower()
    assert name in ds_map.keys(), f'Unsupported dataset {name}'
    return ds_map[name]