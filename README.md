# Refrax

**Refrax** is a small library implementing the *optics* functional pattern for [JAX](https://github.com/jax-ml/jax) PyTrees.

It focuses on elegant syntax and deep conditional chains as opposed to strict static typing.

## Installation

Refrax can be installed using pip:

``
pip install refrax
``

## Quick example

```python
import equinox as eqx
import jax

class Model(eqx.Module):
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

Then we can do updates using `focus`:
```python
from refrax import focus

model = focus(model).dropout.set(0.1)
model = focus(model).select("core", "head").bias.apply(lambda b: b + 1.0)

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