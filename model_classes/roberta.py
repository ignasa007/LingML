from typing import Union

from torch import cat, device, Tensor
from torch.nn import Linear, CrossEntropyLoss

from .base import BaseModel


class RoBERTa(BaseModel):

    def __init__(
        self,
        base_model,
        emb_table_size: Union[int, None],
        two_labels: bool = False,
        dense_size: Union[int, None] = None,
        device: device = device(type='cpu')
    ):

        super().__init__(base_model, emb_table_size, two_labels, dense_size, device)

    def transformer(
        self,
        input_ids: Tensor = None,
        attention_mask: Tensor = None
    ):

        return self.base_model.roberta(input_ids=input_ids, attention_mask=attention_mask)

    def classifier(
        self,
        output: Tensor
    ):
        
        logits = self.base_model.classifier(output)

        return logits

    def get_dense(
        self
    ):

        return self.base_model.classifier.dense
    
    def set_dense(
        self,
        dense: Linear
    ):

        self.base_model.classifier.dense = dense

    def get_output(
        self
    ):

        return self.base_model.classifier.out_proj

    def set_output(
        self,
        dense: Linear
    ):

        self.base_model.classifier.out_proj = dense