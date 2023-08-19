from .bert import BERT
from .roberta import RoBERTa
from .albert import ALBERT
from .distilbert import DistilBERT


def modelclass_map(model_name):

    model_name = model_name.lower()

    if 'roberta' in model_name:
        return RoBERTa
    if 'albert' in model_name:
        return ALBERT
    if 'distilbert' in model_name:
        return DistilBERT
    if 'bert' in model_name:
        return BERT