# from source.plots import svm_accuracy as sv
import configparser
import os

config = configparser.ConfigParser()

BASE_DIR = os.path.dirname(__file__)
config_path = os.path.join(BASE_DIR, 'config.ini')
config.read(config_path)


def get_texts():
    language = config['CONFIGURATION']['language']
    if language == str(0):
        language = 'ENGLISH'
    else:
        language = 'SLOVAK'

    texts = config[language]
    return texts


def get_configuration():
    return config['CONFIGURATION']





