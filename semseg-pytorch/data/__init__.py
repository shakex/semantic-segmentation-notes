import json

from data.pascal_voc_loader import pascalVOCLoader
from data.brainweb_loader import brainWebLoader

def get_loader(name):
    """get_loader

    :param name:
    """
    return {
        'pascal': pascalVOCLoader,
        'brainweb': brainWebLoader

    }[name]


def get_data_path(name, config_file='config.json'):
    """get_data_path

    :param name:
    :param config_file:
    """
    data = json.load(open(config_file))
    return data[name]['data_path']

