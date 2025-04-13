import jax
import jax.numpy as jnp
import optax
import numpy as np
from functools import partial

# @partial(jax.jit, static_argnames=['network', 'value_weight'])
# def compute_loss(params, batch, network, value_weight=1.0):
#     """
#     Calcule la fonction de perte pour l'entraînement du réseau

#     Args:
#         params: Paramètres du réseau
#         batch: Tuple (states, target_policies, target_values)
#             - states: États du jeu (board_2d, marbles_out)
#             - target_policies: Politiques cibles (distributions sur les actions)
#             - target_values: Valeurs cibles (-1 à 1)
#         network: Le réseau de neurones (AbaloneModel)
#         value_weight: Poids relatif de la value loss

#     Returns:
#         total_loss: La perte totale
#         (policy_loss, value_loss, policy_accuracy, value_sign_accuracy): Les composantes et métriques
#     """
#     (board_states, marbles_states), target_policies, target_values = batch

#     # Forward pass du réseau
#     predicted_policies, predicted_values = network.apply(params, board_states, marbles_states)

#     # Policy loss: Cross-entropy avec optax
#     policy_loss = optax.softmax_cross_entropy(predicted_policies, target_policies).mean()

#     # Value loss: Erreur quadratique moyenne
#     value_loss = jnp.mean((target_values - predicted_values.squeeze()) ** 2)

#     # Métriques additionnelles
#     policy_accuracy = jnp.mean(jnp.argmax(predicted_policies, axis=1) ==
#                               jnp.argmax(target_policies, axis=1))
#     value_sign_match = jnp.mean(jnp.sign(predicted_values.squeeze()) ==
#                                jnp.sign(target_values))

#     # Perte totale avec poids configurable
#     total_loss = policy_loss + value_weight * value_loss

#     return total_loss, (policy_loss, value_loss, policy_accuracy, value_sign_match)

@partial(jax.jit, static_argnames=['network', 'value_weight'])
def compute_loss(params, batch, network, value_weight=1.0):
    """
    Calcule la fonction de perte pour l'entraînement du réseau
    """
    (board_states, marbles_states), target_policies, target_values = batch

    predicted_policies, predicted_values = network.apply(params, board_states, marbles_states)
    
    epsilon = 1e-7
    target_policies = target_policies * (1.0 - epsilon) + epsilon / target_policies.shape[-1]
    
    policy_loss = optax.softmax_cross_entropy(predicted_policies, target_policies).mean()

    value_loss = jnp.mean((target_values - predicted_values.squeeze()) ** 2)

    policy_accuracy = jnp.mean(jnp.argmax(predicted_policies, axis=1) ==
                              jnp.argmax(target_policies, axis=1))
    value_sign_match = jnp.mean(jnp.sign(predicted_values.squeeze()) ==
                               jnp.sign(target_values))

    total_loss = policy_loss + value_weight * value_loss
    #print(f"DEBUG compute_loss accuracy: {policy_accuracy}")

    return total_loss, (policy_loss, value_loss, policy_accuracy, value_sign_match)


@partial(jax.jit, static_argnames=['network', 'value_weight'])
def train_step_pmap_impl(params, inputs, target_policies, target_values, network, value_weight=1.0):

    def loss_fn(p):
        batch = (inputs, target_policies, target_values)
        # compute_loss renvoie (scalaire, tuple)
        total_loss, metrics_tuple = compute_loss( # Renommer la variable qui reçoit le tuple
            p, batch, network, value_weight
        )
        # Créer explicitement le dictionnaire de métriques
        metrics_dict_created = {
           'total_loss': total_loss, # Inclure aussi total_loss pour cohérence
           'policy_loss': metrics_tuple[0],
           'value_loss': metrics_tuple[1],
           'policy_accuracy': metrics_tuple[2],
           'value_sign_accuracy': metrics_tuple[3]
        }
        # Retourner le dictionnaire créé comme sortie auxiliaire
        return total_loss, metrics_dict_created


    (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)

    # Moyennage des gradients
    grads = jax.lax.pmean(grads, axis_name='devices')

    return metrics, grads