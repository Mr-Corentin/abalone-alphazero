import jax
import jax.numpy as jnp
import optax
import numpy as np
from functools import partial
from typing import Dict, Any, Tuple


@partial(jax.jit, static_argnames=['value_weight'])
def calculate_losses_and_metrics(predicted_policies,
                                 predicted_values,
                                 target_policies,
                                 target_values,
                                 value_weight=1.0):
    """
    Calcule les composantes de la perte et les métriques à partir des prédictions et des cibles.
    """
    epsilon = 1e-7
    # Label smoothing appliqué aux cibles
    target_policies_smoothed = target_policies * (1.0 - epsilon) + epsilon / target_policies.shape[-1]

    policy_loss = optax.softmax_cross_entropy(predicted_policies, target_policies_smoothed).mean()

    # S'assurer que predicted_values est bien squeezé si nécessaire,
    # dépendant de la sortie de votre tête de valeur.
    # Si votre tête de valeur sort (batch_size, 1), squeeze() est correct.
    # Si elle sort déjà (batch_size,), alors squeeze() n'est pas nécessaire ou pourrait causer une erreur.
    # Supposons que predicted_values peut avoir une dimension de 1 à la fin.
    squeezed_predicted_values = predicted_values.squeeze()

    value_loss = jnp.mean((target_values - squeezed_predicted_values) ** 2)

    policy_accuracy = jnp.mean(jnp.argmax(predicted_policies, axis=1) ==
                               jnp.argmax(target_policies, axis=1)) # Utiliser target_policies brutes pour l'accuracy
    value_sign_match = jnp.mean(jnp.sign(squeezed_predicted_values) ==
                                jnp.sign(target_values))

    total_loss = policy_loss + value_weight * value_loss

    # Retourner un tuple pour les métriques, comme avant.
    metrics_tuple = (policy_loss, value_loss, policy_accuracy, value_sign_match)
    return total_loss, metrics_tuple


@partial(jax.jit, static_argnames=['network', 'value_weight'])
def train_step_pmap_impl(params: Dict[str, Any],
                         inputs: Tuple[jnp.ndarray, jnp.ndarray], # (board_states, marbles_states)
                         target_policies: jnp.ndarray,
                         target_values: jnp.ndarray,
                         network: Any, # Type de votre modèle Flax
                         value_weight: float = 1.0):
    """
    Effectue une étape d'entraînement avec params seulement.
    
    Args:
        params: Paramètres du modèle.
        inputs: Tuple (board_states, marbles_states).
        target_policies: Politiques cibles.
        target_values: Valeurs cibles.
        network: L'instance du modèle Flax.
        value_weight: Poids pour la perte de valeur.
        
    Returns:
        metrics_dict: Dictionnaire des métriques.
        grads: Gradients moyennés.
    """

    # La fonction de perte interne pour jax.value_and_grad.
    def loss_fn_for_grad(current_params: Dict[str, Any]):
        board_states, marbles_states = inputs

        # Appel au réseau
        predicted_policies, predicted_values = network.apply(
            current_params,
            board_states,
            marbles_states
        )

        # Utiliser la fonction refactorée pour calculer les pertes et métriques
        total_loss, metrics_tuple = calculate_losses_and_metrics(
            predicted_policies, predicted_values, target_policies, target_values, value_weight
        )
        
        # Construire le dictionnaire de métriques
        metrics_dict = {
            'total_loss': total_loss,
            'policy_loss': metrics_tuple[0],
            'value_loss': metrics_tuple[1],
            'policy_accuracy': metrics_tuple[2],
            'value_sign_accuracy': metrics_tuple[3]
        }
        
        return total_loss, metrics_dict

    # Calculer la perte, les gradients et les métriques
    (loss_value, metrics), grads = jax.value_and_grad(
        loss_fn_for_grad, has_aux=True
    )(params)

    # Synchronisation (moyennage) des gradients et des métriques sur les devices
    grads = jax.lax.pmean(grads, axis_name='devices')
    metrics = jax.lax.pmean(metrics, axis_name='devices')

    return metrics, grads