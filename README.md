# Refrax

**Refrax** is a tiny library implementing the *optics* functional pattern for [JAX](https://github.com/jax-ml/jax) PyTrees.

## Installation

Refrax can be installed using pip:

``
pip install refrax
``

## Quick example

This example demonstrates the Mixin approach on an [Equinox](https://github.com/patrick-kidger/equinox) module, which adds `.at` to your class and simply returns `refrax.Lens(self)`.

We define a dummy model with the mixin:
```python
from refrax import OpticsMixin

import equinox as eqx
import jax

class Model(eqx.Module, OpticsMixin):
    core: eqx.nn.Linear
    head: eqx.nn.Linear
    dropout: float

key1, key2 = jax.random.split(jax.random.key(0))
model = Model(
    core=eqx.nn.Linear(in_features=5, out_features=5, key=key1),
    head=eqx.nn.Linear(in_features=5, out_features=2, key=key2),
    dropout=0.5
)
```

Then we can do updates using `.at`:
```python
model = model.at.dropout.set(0.1)
model = model.at.select("core", "head").bias.apply(lambda b: b + 1.0)

model.dropout
# 0.1

model.core.bias
# [0.646, 0.860, 0.670 , 1.277 , 0.727]

model.head.bias
# [1.634, 1.877]
```

## Documentation

Documentation is available [here](https://gvcallen.github.io/refrax/).

## Related

The library uses [Equinox](https://github.com/patrick-kidger/equinox) (specifically `eqx.tree_at`) under the hood.