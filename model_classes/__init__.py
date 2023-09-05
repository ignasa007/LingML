from .albert import ALBERT
from .bert import BERT
from .bertweet import BERTweet
from .ct_bert import CTBERT
from .distilbert import DistilBERT
from .roberta import RoBERTa
from .twitter_roberta import TwitterRoBERTa


map = {
    'albert-base-v2': ALBERT,
    'bert-base-uncased': BERT,
    'bertweet-covid19-base-uncased': BERTweet,
    'covid-twitter-bert-v2': CTBERT,
    'distilbert-base-uncased': DistilBERT,
    'roberta-base': RoBERTa,
    'twitter-roberta-base-sentiment-latest': TwitterRoBERTa,
}


def modelclass_map(model_name):

    model_name = model_name.lower()

    return map[model_name]