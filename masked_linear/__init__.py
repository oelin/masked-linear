def get_hidden_layer_ids(layer_size: int, input_size: int) -> Tuple:
    
    return tuple(torch.randint(low=1, high=input_size, size=(layer_size,)).tolist())
  

def get_layer_mask(this_layer_ids: Tuple, next_layer_ids: Tuple, is_output: bool = False) -> torch.Tensor:

    this_layer_size = len(this_layer_ids)
    next_layer_size = len(next_layer_ids)

    mask = torch.ones((next_layer_size, this_layer_size))

    # Zero out non-causal connections.

    for i, this_id in enumerate(this_layer_ids):
        for j, next_id in enumerate(next_layer_ids):

            if is_output:
                mask[j, i] = int(next_id > this_id)
            else:
                mask[j, i] = int(next_id >= this_id)
    return mask.to(float)


class MaskedLinear(nn.Module):
    """A masked linear layer."""

    def __init__(
        self,
        in_features: int, 
        out_features: int,
        input_layer_size: int,
        previous_layer_ids: Tuple,
    ) -> None:
        """Initializes the module."""

        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        std = math.sqrt(2 / (in_features + out_features))
        self.weight = nn.Parameter(std * torch.randn((out_features, in_features)))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.layer_ids = get_hidden_layer_ids(out_features, input_layer_size)

        self.mask = get_layer_mask(previous_layer_ids, self.layer_ids)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""

        return x @ (self.weight * self.mask).T + self.bias
