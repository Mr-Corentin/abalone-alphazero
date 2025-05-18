import jax
import jax.numpy as jnp
import optax
import numpy as np
from functools import partial

# MODIFIÉ: Cette fonction ne prend plus params ni network, mais directement les prédictions.
# Elle ne fait plus l'appel network.apply.
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
def train_step_pmap_impl(model_variables: Dict[str, Any], # MODIFIÉ: Prend model_variables
                         inputs: Tuple[jnp.ndarray, jnp.ndarray], # (board_states, marbles_states)
                         target_policies: jnp.ndarray,
                         target_values: jnp.ndarray,
                         network: Any, # Type de votre modèle Flax
                         value_weight: float = 1.0):
    """
    Effectue une étape d'entraînement, gère la BN et la synchronisation pmap.
    
    Args:
        model_variables: Dictionnaire contenant 'params' et 'batch_stats'.
        inputs: Tuple (board_states, marbles_states).
        target_policies: Politiques cibles.
        target_values: Valeurs cibles.
        network: L'instance du modèle Flax.
        value_weight: Poids pour la perte de valeur.
        
    Returns:
        metrics_dict: Dictionnaire des métriques.
        grads: Gradients moyennés.
        updated_batch_stats: batch_stats mises à jour et moyennées.
    """

    # La fonction de perte interne pour jax.value_and_grad.
    # Elle est différenciée par rapport à `current_params`.
    # `model_variables['batch_stats']` est capturé par la closure.
    def loss_fn_for_grad(current_params: Dict[str, Any]):
        # Préparer les variables pour l'appel à network.apply
        variables_for_apply = {'params': current_params, 'batch_stats': model_variables['batch_stats']}
        board_states, marbles_states = inputs

        # Appel au réseau avec train=True et gestion de la mutabilité des batch_stats
        (predicted_policies, predicted_values), updated_model_state = network.apply(
            variables_for_apply,
            board_states,
            marbles_states,
            train=True,           # Crucial pour l'entraînement
            mutable=['batch_stats'] # Pour mettre à jour les batch_stats
        )
        # Récupérer les batch_stats mises à jour
        new_batch_stats = updated_model_state['batch_stats']

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
            'value_sign_accuracy': metrics_tuple[3] # Nom corrigé pour correspondre à votre code original
        }
        # Retourner la perte totale et, en auxiliaire, les nouvelles batch_stats et les métriques
        return total_loss, (new_batch_stats, metrics_dict)

    # Calculer la perte, les gradients (par rapport à current_params, donc model_variables['params'])
    # et les sorties auxiliaires (new_batch_stats, metrics_dict).
    (loss_value, (updated_batch_stats, metrics)), grads = jax.value_and_grad(
        loss_fn_for_grad, has_aux=True
    )(model_variables['params']) # On passe seulement les params ici pour la différentiation

    # Synchronisation (moyennage) des gradients, des métriques et des batch_stats sur les devices
    grads = jax.lax.pmean(grads, axis_name='devices')
    metrics = jax.lax.pmean(metrics, axis_name='devices') # Moyenner aussi les métriques
    updated_batch_stats = jax.lax.pmean(updated_batch_stats, axis_name='devices')

    return metrics, grads, updated_batch_stats # Retourner les batch_stats mises à jour