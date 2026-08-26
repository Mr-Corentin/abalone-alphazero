"""
Helpers to feed data to `jax.pmap` without the removed device-placement helpers.

`jax.device_put_sharded` and `jax.device_put_replicated` were removed in JAX 0.11.
Neither is needed here:

* Sharding an array across local devices is what `pmap` does by itself. Any array
  whose leading axis equals `jax.local_device_count()` is split automatically, so
  `jax.device_put_sharded(list(xs), devices)` can simply be dropped and `xs`
  passed straight through.

* Replicating a pytree only requires adding a leading device axis, which
  `jnp.broadcast_to` does on every JAX version.

Longer term, the direction JAX is moving is `jax.jit` with explicit
`jax.sharding.NamedSharding` over a `Mesh` instead of `pmap` -- that is the
"unified jit" model the pmap migration guide describes. It replaces the whole
manual replicate / reshape / pmap dance with sharding annotations, but it is a
rewrite of the training loop rather than a drop-in change, so it is deliberately
not done here.
"""

import jax
import jax.numpy as jnp


def replicate(tree, num_devices: int):
    """
    Add a leading axis of size `num_devices` to every leaf of `tree`.

    Drop-in replacement for `jax.device_put_replicated(tree, devices)` that works
    on every JAX version. Used for values that must be identical on each device
    (model parameters, optimiser state).
    """
    return jax.tree.map(
        lambda x: jnp.broadcast_to(jnp.asarray(x), (num_devices,) + jnp.shape(x)),
        tree,
    )


def unreplicate(tree):
    """Take device 0's copy of a replicated pytree."""
    return jax.tree.map(lambda x: x[0], tree)
