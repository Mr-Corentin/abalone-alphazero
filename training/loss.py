import jax
import jax.numpy as jnp
import optax
import numpy as np
from functools import partial

@partial(jax.jit, static_argnames=['network', 'value_weight'])
def compute_loss(params, batch, network, value_weight=1.0):
    """
    Calcule la fonction de perte pour l'entraînement du réseau

    Args:
        params: Paramètres du réseau
        batch: Tuple (states, target_policies, target_values)
            - states: États du jeu (board_2d, marbles_out)
            - target_policies: Politiques cibles (distributions sur les actions)
            - target_values: Valeurs cibles (-1 à 1)
        network: Le réseau de neurones (AbaloneModel)
        value_weight: Poids relatif de la value loss

    Returns:
        total_loss: La perte totale
        (policy_loss, value_loss, policy_accuracy, value_sign_accuracy): Les composantes et métriques
    """
    (board_states, marbles_states), target_policies, target_values = batch

    # Forward pass du réseau
    predicted_policies, predicted_values = network.apply(params, board_states, marbles_states)

    # Policy loss: Cross-entropy avec optax
    policy_loss = optax.softmax_cross_entropy(predicted_policies, target_policies).mean()

    # Value loss: Erreur quadratique moyenne
    value_loss = jnp.mean((target_values - predicted_values.squeeze()) ** 2)

    # Métriques additionnelles
    policy_accuracy = jnp.mean(jnp.argmax(predicted_policies, axis=1) ==
                              jnp.argmax(target_policies, axis=1))
    value_sign_match = jnp.mean(jnp.sign(predicted_values.squeeze()) ==
                               jnp.sign(target_values))

    # Perte totale avec poids configurable
    total_loss = policy_loss + value_weight * value_loss

    return total_loss, (policy_loss, value_loss, policy_accuracy, value_sign_match)

@partial(jax.jit, static_argnames=['network', 'value_weight'])
def train_step_pmap_impl(params, inputs, target_policies, target_values, network, value_weight=1.0):
    """
    Implémentation de train_step pour utilisation avec pmap.

    Args:
        params: Paramètres actuels du réseau
        inputs: Tuple (boards, marbles) - entrées du réseau
        target_policies: Politiques cibles
        target_values: Valeurs cibles
        network: Le réseau de neurones
        value_weight: Poids de la loss de valeur

    Returns:
        Tuple (métriques, gradients)
    """
    def loss_fn(p):
        batch = (inputs, target_policies, target_values)
        total_loss, (policy_loss, value_loss, policy_accuracy, value_sign_match) = compute_loss(
            p, batch, network, value_weight
        )

        metrics = {
            'total_loss': total_loss,
            'policy_loss': policy_loss,
            'value_loss': value_loss,
            'policy_accuracy': policy_accuracy,
            'value_sign_accuracy': value_sign_match
        }

        return total_loss, metrics

    # Calculer la perte et les gradients
    (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)

    return metrics, grads