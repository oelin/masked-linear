# Masked Linear

An implementation of a masked linear layer, as used by MADE.

![](https://i.ibb.co/ysq0P1w/made.png)

## Usage

```python
from masked_linear import MaskedLinear

linear = MaskedLinear(
  in_features=5,
  out_features=3,
  input_layer_size = 28 * 28,  # Size of the autoencoder's input.
  previous_layer_ids = (1, 2, 3, ...),  # A tuple specifying the IDs for each unit of the previous layer.
)

x = linear(torch.ones((10, 28 * 28)))
```
