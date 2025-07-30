import jax
import jax.numpy as jnp
import optax
import numpy as np
from functools import partial


@partial(jax.jit, static_argnames=['network', 'value_weight'])
def compute_loss(params, batch, network, value_weight=1.0):
    """
    Calculate loss function for network training
    """
    inputs, target_policies, target_values = batch
    
    # Handle input format with or without history
    if len(inputs) == 3:
        # New format with history: (board_states, marbles_states, history_states)
        board_states, marbles_states, history_states = inputs
        predicted_policies, predicted_values = network.apply(params, board_states, marbles_states, history_states)
    else:
        # Old format without history: (board_states, marbles_states)
        board_states, marbles_states = inputs
        predicted_policies, predicted_values = network.apply(params, board_states, marbles_states)
    
    epsilon = 1e-7
    target_policies = target_policies * (1.0 - epsilon) + epsilon / target_policies.shape[-1]
    
    policy_loss = optax.softmax_cross_entropy(predicted_policies, target_policies).mean()

    squeezed_predicted_values = predicted_values.squeeze()
    value_loss = jnp.mean((target_values - squeezed_predicted_values) ** 2)

    policy_accuracy = jnp.mean(jnp.argmax(predicted_policies, axis=1) ==
                              jnp.argmax(target_policies, axis=1))
    value_sign_match = jnp.mean(jnp.sign(squeezed_predicted_values) ==
                               jnp.sign(target_values))

    total_loss = policy_loss + value_weight * value_loss

    return total_loss, (policy_loss, value_loss, policy_accuracy, value_sign_match)

@partial(jax.jit, static_argnames=['network', 'value_weight'])
def train_step_pmap_impl(params, inputs, target_policies, target_values, network, value_weight=1.0):

    def loss_fn(p):
        batch = (inputs, target_policies, target_values)
        # compute_loss returns (scalar, tuple)
        total_loss, metrics_tuple = compute_loss( # Rename variable that receives tuple
            p, batch, network, value_weight
        )
        # Explicitly create metrics dictionary
        metrics_dict_created = {
           'total_loss': total_loss, # Also include total_loss for consistency
           'policy_loss': metrics_tuple[0],
           'value_loss': metrics_tuple[1],
           'policy_accuracy': metrics_tuple[2],
           'value_sign_accuracy': metrics_tuple[3]
        }
        # Return created dictionary as auxiliary output
        return total_loss, metrics_dict_created


    (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)

    # Average gradients
    grads = jax.lax.pmean(grads, axis_name='devices')

    return metrics, grads