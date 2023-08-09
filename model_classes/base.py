from typing import Union

from torch import device, Tensor, cat
from torch.nn import Linear, CrossEntropyLoss


class BaseModel:

    def __init__(
        self,
        base_model,
        emb_table_size: Union[int, None],
        two_labels: bool = False,
        dense_size: Union[int, None] = None,
        device: device = device(type='cpu')
    ):

        self.base_model = base_model

        if isinstance(emb_table_size, int):
            self.base_model.resize_token_embeddings(emb_table_size)

        if two_labels:
            self.set_two_labels()

        self.add_dense = False
        if isinstance(dense_size, int):
            self.extend_dense_layer(dense_size)

        self.base_model.to(device)

    def set_two_labels(
        self
    ):

        if self.base_model.num_labels < 2:
            raise RuntimeError(f'Model has only {self.base_model.num_labels} labels, cannot set to 2.')

        output = self.get_output()

        new_output = Linear(in_features=output.weight.size(1), out_features=2, bias=True)
        new_output.weight.data = output.weight[[0, -1], :]
        new_output.bias.data = output.bias[[0, -1]]
        
        self.set_output(new_output)
        self.base_model.num_labels = 2

    def extend_dense_layer(
        self, 
        size: int
    ):

        if self.add_dense:
            print(f'Already extended dense layer by {self.size} units.')
            return
        
        dense = self.get_dense()
        new_dense = Linear(in_features=dense.weight.size(-1)+size, out_features=dense.weight.size(0), bias=True)
        new_dense.weight.data.normal_(mean=0.0, std=self.base_model.config.initializer_range)
        new_dense.weight.data[:, :dense.weight.size(-1)] = dense.weight
        new_dense.bias.data.zero_()
        new_dense.bias.data[:dense.bias.size(-1)] = dense.bias

        self.set_dense(new_dense)
        self.add_dense = True
        self.size = size

    def __call__(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        dense_features: Union[Tensor, None],
        labels: Tensor
    ):

        raise NotImplementedError

    def get_dense(
        self
    ):

        raise NotImplementedError
    
    def set_dense(
        self,
        dense: Linear
    ):

        raise NotImplementedError
    
    def get_output(
        self
    ):

        raise NotImplementedError
    
    def set_output(
        self,
        dense: Linear
    ):

        raise NotImplementedError