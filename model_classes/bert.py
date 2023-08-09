from typing import Union

from torch import cat, device, Tensor
from torch.nn import Linear, CrossEntropyLoss

from .base import BaseModel


class BERT(BaseModel):

    def __init__(
        self,
        base_model,
        emb_table_size: Union[int, None],
        two_labels: bool = False,
        dense_size: Union[int, None] = None,
        device: device = device(type='cpu')
    ):

        super().__init__(base_model, emb_table_size, two_labels, dense_size, device)

    def __call__(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        dense_features: Union[Tensor, None],
        labels: Tensor
    ):

        outputs = self.base_model.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        if self.add_dense:
            output = outputs[0]    # sequence output
            output = cat((output[:, [0], :], dense_features.unsqueeze(1)), dim=-1)
            output = self.base_model.bert.pooler(output)
        else:
            output = outputs[1]    # pooled output
        
        output = self.base_model.dropout(output)
        logits = self.base_model.classifier(output)

        loss = CrossEntropyLoss()(logits.view(-1, self.base_model.num_labels), labels.view(-1))
        outputs = (loss, logits,) + outputs[2:]

        return outputs

    def get_dense(
        self
    ):

        return self.base_model.bert.pooler.dense
    
    def set_dense(
        self,
        dense: Linear
    ):

        self.base_model.bert.pooler.dense = dense

    def get_output(
        self
    ):

        return self.base_model.classifier

    def set_output(
        self,
        dense: Linear
    ):

        self.base_model.classifier = dense