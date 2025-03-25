import jax
import jax.numpy as jnp
import time
import gc

# Intégrer jax-smi pour le suivi mémoire
from jax_smi import initialise_tracking

# Imports de votre projet
from environment.env import AbaloneEnv
from model.neural_net import AbaloneModel
from training.trainer import AbaloneTrainerSync

def test_memory_capacity(num_simulations=800):
    """Teste différentes configurations pour trouver la capacité maximale avec 800 simulations."""
    
    # Initialiser le suivi mémoire
    initialise_tracking()
    
    print(f"Devices: {jax.devices()}")
    print(f"TPU cores: {len(jax.devices())}")
    print(f"Test avec {num_simulations} simulations MCTS par coup")
    print(f"Suivi mémoire activé - vous pouvez exécuter 'jax-smi' dans un autre terminal")
    
    env = AbaloneEnv()
    model = AbaloneModel(num_filters=64, num_blocks=6)
    
    # Tester des valeurs croissantes de parties par cœur
    for games_per_core in [ 4, 8, 16, 32]:
        try:
            # Forcer le garbage collection entre les tests
            gc.collect()
            
            total_games = games_per_core * len(jax.devices())
            print(f"\n--- Test: {games_per_core} parties/cœur ({total_games} total) ---")
            print(f"Exécutez 'jax-smi' dans un autre terminal pour voir l'utilisation mémoire")
            
            trainer = AbaloneTrainerSync(
                network=model,
                env=env,
                buffer_size=10000,
                batch_size=32,
                games_per_device=games_per_core,
                num_simulations=num_simulations
            )
            
            # Mesurer temps pour génération
            start_time = time.time()
            rng_key = jax.random.PRNGKey(42)
            
            print("Génération de parties...")
            games_data = trainer._generate_games(rng_key, total_games)
            
            gen_time = time.time() - start_time
            
            # Afficher résultats
            print(f"Succès! {total_games} parties générées en {gen_time:.2f}s")
            print(f"Vitesse: {total_games/gen_time:.2f} parties/seconde")
            print(f"Temps moyen par partie: {gen_time/total_games:.2f}s")
            
            # Pause pour permettre de vérifier la mémoire avec jax-smi
            print("Pause de 5 secondes pour vérifier la mémoire avec jax-smi...")
            time.sleep(5)
            
        except (RuntimeError, jax.errors.OutOfMemoryError) as e:
            print(f"Erreur mémoire à {games_per_core} parties/cœur: {str(e)}")
            print(f"Limite maximum semble être {games_per_core-1} parties/cœur")
            break

if __name__ == "__main__":
    test_memory_capacity(num_simulations=800)