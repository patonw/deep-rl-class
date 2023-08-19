# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import numpy as np
import esnpy as ep

# %%
rb = ep.ReservoirBuilder(
    size=64,
    leaky=0.9,
    fn=np.tanh,
    input_size=2,
    input_init=ep.init.UniformDenseInit(-2, 2),
    intern_init=ep.init.NormalSparseInit(0, 1, density=0.01),
    intern_tuners=[ep.tune.SpectralRadiusTuner(0.9)],
)

# %%
res1 = rb.build(42)
res2 = rb.build(42)
res1._state, res2._state

# %%
X = np.array([[1, 5]])
(res1(X) == res2(X)).all()

# %%
res1._state

# %%
res1(X), res1._state

# %%
