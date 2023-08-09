import pandas as pd

from .base import BaseDataset


class ConstraintFiltered(BaseDataset):

    def __init__(
        self, 
        root: str,
        train_pct: int,
        valid_pct: int,
    ):

        df = pd.read_csv(f'{root}/dataset.csv', usecols=['tweet_denoised', 'veracity_finetuned'])
        df = df.sample(frac=1).reset_index(drop=True).rename({'tweet_denoised': 'tweet', 'veracity_finetuned': 'label'}, axis=1)
        df = df.loc[~df.tweet.isna() & df.label.isin(('real', 'fake')), :]
        df.label = df.label.replace({'real': 1, 'fake': 0})
        
        train_end, val_end = int(train_pct*df.shape[0]/100), int((train_pct+valid_pct)*df.shape[0]/100)
        self.datasets = {
            'train': df.iloc[:train_end, :],
            'val': df.iloc[train_end:val_end, :],
            'test': df.iloc[val_end:, :]
        }