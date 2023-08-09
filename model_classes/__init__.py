from .bert import BERT
from .roberta import RoBERTa


def modelclass_map(model_name):

    if 'roberta' in model_name:
        return RoBERTa
    if 'bert' in model_name:
        return BERT