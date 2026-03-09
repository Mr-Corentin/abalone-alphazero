"""
Profiling utilities for MCTS to identify bottlenecks
"""
import time
import jax
import jax.numpy as jnp
from functools import partial
from environment.env import AbaloneEnv, AbaloneState
from model.neural_net import AbaloneModel
from mcts.core import AbaloneMCTSRecurrentFn, get_root_output_batch
import mctx


@partial(jax.jit, static_argnames=['network', 'env', 'num_simulations'])
def profile_mcts_single_search(
    state: AbaloneState,
    params,
    network: AbaloneModel,
    env: AbaloneEnv,
    rng_key,
    iteration: int = 0,
    num_simulations: int = 600
):
    """
    Version de run_search_batch pour profiling avec un seul état
    """
    # Transformer en batch de taille 1
    batch_state = AbaloneState(
        board=state.board[None, ...],
        history=state.history[None, ...],
        actual_player=jnp.array([state.actual_player]),
        black_out=jnp.array([state.black_out]),
        white_out=jnp.array([state.white_out]),
        moves_count=jnp.array([state.moves_count])
    )

    recurrent_fn = AbaloneMCTSRecurrentFn(env, network)

    # Get root
    root = get_root_output_batch(batch_state, network, params, env, iteration)

    # Get legal moves
    legal_moves = jax.vmap(env.get_legal_moves)(batch_state)
    invalid_actions = ~legal_moves

    # MCTS search
    policy_output = mctx.gumbel_muzero_policy(
        params=params,
        rng_key=rng_key,
        root=root,
        recurrent_fn=recurrent_fn.recurrent_fn,
        num_simulations=num_simulations,
        max_num_considered_actions=16,
        invalid_actions=invalid_actions,
        gumbel_scale=1.0
    )

    return policy_output


def benchmark_mcts_components(
    env: AbaloneEnv,
    network: AbaloneModel,
    params,
    batch_size: int = 8,
    num_simulations: int = 600,
    num_warmup: int = 2,
    num_runs: int = 10
):
    """
    Benchmark des différents composants de MCTS

    Args:
        env: Environment
        network: Neural network
        params: Network parameters
        batch_size: Nombre de parties en parallèle
        num_simulations: Nombre de simulations MCTS
        num_warmup: Nombre d'itérations de warmup
        num_runs: Nombre de runs pour la moyenne
    """
    print("\n" + "="*70)
    print(f"PROFILING MCTS (batch_size={batch_size}, num_simulations={num_simulations})")
    print("="*70)

    # Créer un état initial
    rng = jax.random.PRNGKey(42)
    init_states = env.reset_batch(rng, batch_size)

    # Setup
    recurrent_fn = AbaloneMCTSRecurrentFn(env, network)

    # ========================================================================
    # 1. BENCHMARK: Récupération des coups légaux
    # ========================================================================
    print("\n[1/5] Benchmarking: get_legal_moves (batched)")

    @jax.jit
    def bench_legal_moves(states):
        return jax.vmap(env.get_legal_moves)(states)

    # Warmup
    for _ in range(num_warmup):
        legal_moves = bench_legal_moves(init_states)
        legal_moves.block_until_ready()

    # Mesure
    times = []
    for _ in range(num_runs):
        start = time.time()
        legal_moves = bench_legal_moves(init_states)
        legal_moves.block_until_ready()  # CRUCIAL: Force la synchro
        times.append(time.time() - start)

    avg_time_legal = sum(times) / len(times)
    print(f"  Temps moyen: {avg_time_legal*1000:.3f} ms")
    print(f"  Par état: {avg_time_legal*1000/batch_size:.3f} ms")

    # ========================================================================
    # 2. BENCHMARK: Root evaluation (forward pass initial)
    # ========================================================================
    print("\n[2/5] Benchmarking: Root evaluation (get_root_output_batch)")

    @jax.jit
    def bench_root(states):
        return get_root_output_batch(states, network, params, env, 0)

    # Warmup
    for _ in range(num_warmup):
        root = bench_root(init_states)
        root.prior_logits.block_until_ready()

    # Mesure
    times = []
    for _ in range(num_runs):
        start = time.time()
        root = bench_root(init_states)
        root.prior_logits.block_until_ready()
        times.append(time.time() - start)

    avg_time_root = sum(times) / len(times)
    print(f"  Temps moyen: {avg_time_root*1000:.3f} ms")
    print(f"  Par état: {avg_time_root*1000/batch_size:.3f} ms")

    # ========================================================================
    # 3. BENCHMARK: Recurrent function (simulation step)
    # ========================================================================
    print("\n[3/5] Benchmarking: Recurrent function (network forward pass)")

    # Préparer des actions aléatoires
    test_actions = jnp.zeros(batch_size, dtype=jnp.int32)
    test_embedding = {
        'board_3d': init_states.board,
        'history_3d': init_states.history,
        'actual_player': init_states.actual_player,
        'black_out': init_states.black_out,
        'white_out': init_states.white_out,
        'moves_count': init_states.moves_count,
        'iteration': jnp.zeros_like(init_states.actual_player)
    }

    @jax.jit
    def bench_recurrent(params, rng, actions, embedding):
        return recurrent_fn.recurrent_fn(params, rng, actions, embedding)

    # Warmup
    for _ in range(num_warmup):
        rng, key = jax.random.split(rng)
        output, _ = bench_recurrent(params, key, test_actions, test_embedding)
        output.prior_logits.block_until_ready()

    # Mesure
    times = []
    for _ in range(num_runs):
        rng, key = jax.random.split(rng)
        start = time.time()
        output, _ = bench_recurrent(params, key, test_actions, test_embedding)
        output.prior_logits.block_until_ready()
        times.append(time.time() - start)

    avg_time_recurrent = sum(times) / len(times)
    print(f"  Temps moyen: {avg_time_recurrent*1000:.3f} ms")
    print(f"  Par état: {avg_time_recurrent*1000/batch_size:.3f} ms")

    # ========================================================================
    # 4. BENCHMARK: Recurrent avec batch étendu (simule MCTS réel)
    # ========================================================================
    print("\n[4/5] Benchmarking: Recurrent avec batch MCTS réel")
    print(f"  Batch effectif: {batch_size} parties × 16 actions = {batch_size*16} positions")

    # Simuler le batch réel de MCTS
    expanded_batch_size = batch_size * 16  # max_num_considered_actions
    expanded_actions = jnp.zeros(expanded_batch_size, dtype=jnp.int32)
    expanded_embedding = {
        'board_3d': jnp.repeat(init_states.board, 16, axis=0),
        'history_3d': jnp.repeat(init_states.history, 16, axis=0),
        'actual_player': jnp.repeat(init_states.actual_player, 16),
        'black_out': jnp.repeat(init_states.black_out, 16),
        'white_out': jnp.repeat(init_states.white_out, 16),
        'moves_count': jnp.repeat(init_states.moves_count, 16),
        'iteration': jnp.zeros(expanded_batch_size, dtype=jnp.int32)
    }

    # Warmup
    for _ in range(num_warmup):
        rng, key = jax.random.split(rng)
        output, _ = bench_recurrent(params, key, expanded_actions, expanded_embedding)
        output.prior_logits.block_until_ready()

    # Mesure
    times = []
    for _ in range(num_runs):
        rng, key = jax.random.split(rng)
        start = time.time()
        output, _ = bench_recurrent(params, key, expanded_actions, expanded_embedding)
        output.prior_logits.block_until_ready()
        times.append(time.time() - start)

    avg_time_recurrent_expanded = sum(times) / len(times)
    print(f"  Temps moyen: {avg_time_recurrent_expanded*1000:.3f} ms")
    print(f"  Par position: {avg_time_recurrent_expanded*1000/expanded_batch_size:.3f} ms")

    # ========================================================================
    # 5. BENCHMARK: MCTS complet
    # ========================================================================
    print("\n[5/5] Benchmarking: MCTS complet (gumbel_muzero_policy)")

    @jax.jit
    def bench_full_mcts(states, params, rng):
        root = get_root_output_batch(states, network, params, env, 0)
        legal_moves = jax.vmap(env.get_legal_moves)(states)
        invalid_actions = ~legal_moves

        policy_output = mctx.gumbel_muzero_policy(
            params=params,
            rng_key=rng,
            root=root,
            recurrent_fn=recurrent_fn.recurrent_fn,
            num_simulations=num_simulations,
            max_num_considered_actions=16,
            invalid_actions=invalid_actions,
            gumbel_scale=1.0
        )
        return policy_output

    # Warmup
    for _ in range(num_warmup):
        rng, key = jax.random.split(rng)
        output = bench_full_mcts(init_states, params, key)
        output.action.block_until_ready()

    # Mesure
    times = []
    for _ in range(num_runs):
        rng, key = jax.random.split(rng)
        start = time.time()
        output = bench_full_mcts(init_states, params, key)
        output.action.block_until_ready()
        times.append(time.time() - start)

    avg_time_full = sum(times) / len(times)
    print(f"  Temps moyen: {avg_time_full*1000:.3f} ms")
    print(f"  Par état: {avg_time_full*1000/batch_size:.3f} ms")
    print(f"  Par simulation: {avg_time_full*1000/num_simulations:.3f} ms")

    # ========================================================================
    # ANALYSE & BREAKDOWN
    # ========================================================================
    print("\n" + "="*70)
    print("ANALYSE DES RÉSULTATS")
    print("="*70)

    # Estimation du temps par simulation
    time_per_sim = avg_time_full / num_simulations

    # Overhead MCTS (sélection, backprop, etc.)
    overhead = time_per_sim - avg_time_recurrent_expanded

    print(f"\nTemps par simulation MCTS: {time_per_sim*1000:.3f} ms")
    print(f"  └─ Inférence réseau:     {avg_time_recurrent_expanded*1000:.3f} ms ({avg_time_recurrent_expanded/time_per_sim*100:.1f}%)")
    print(f"  └─ Overhead MCTS:        {overhead*1000:.3f} ms ({overhead/time_per_sim*100:.1f}%)")

    print(f"\nCoût par opération:")
    print(f"  - get_legal_moves:       {avg_time_legal*1000:.3f} ms (négligeable)")
    print(f"  - Root evaluation:       {avg_time_root*1000:.3f} ms (1x par coup)")
    print(f"  - Recurrent step:        {avg_time_recurrent_expanded*1000:.3f} ms ({num_simulations}x par coup)")

    print(f"\nEstimation temps par coup de jeu:")
    estimated_time_per_move = avg_time_root + (num_simulations * time_per_sim)
    print(f"  Total: {estimated_time_per_move:.3f} s")
    print(f"    = Root ({avg_time_root*1000:.1f} ms) + {num_simulations} sims × {time_per_sim*1000:.3f} ms")

    print(f"\nEstimation temps par partie (100 coups):")
    estimated_time_per_game = 100 * estimated_time_per_move
    print(f"  Total: {estimated_time_per_game:.1f} s")
    print(f"  Pour {batch_size} parties: {estimated_time_per_game:.1f} s")
    print(f"  Throughput: {batch_size/estimated_time_per_game:.2f} parties/s")

    # Recommandations
    print("\n" + "="*70)
    print("RECOMMANDATIONS")
    print("="*70)

    inference_ratio = avg_time_recurrent_expanded / time_per_sim

    if inference_ratio > 0.8:
        print("\n✅ L'inférence réseau domine (>{:.0f}%)".format(inference_ratio*100))
        print("   → Optimisations réseau prioritaires:")
        print("     - Augmenter batch_size (games_per_device)")
        print("     - Utiliser bfloat16 si pas déjà fait")
        print("     - Vérifier compilation XLA")
    else:
        print("\n⚠️  L'overhead MCTS est significatif ({:.0f}%)".format((1-inference_ratio)*100))
        print("   → Le goulot n'est pas l'inférence:")
        print("     - Vérifier les opérations JAX dans l'arbre MCTS")
        print("     - Possibles transferts CPU↔TPU")
        print("     - Sélection d'actions non optimale")

    print(f"\n💡 Gain potentiel si num_simulations réduit à 400:")
    print(f"   Temps/coup: {estimated_time_per_move * 400/600:.3f} s (-33%)")
    print(f"   Temps/partie: {estimated_time_per_game * 400/600:.1f} s")
    print(f"   Throughput: {batch_size/(estimated_time_per_game * 400/600):.2f} parties/s (+50%)")

    return {
        'legal_moves_ms': avg_time_legal * 1000,
        'root_eval_ms': avg_time_root * 1000,
        'recurrent_ms': avg_time_recurrent * 1000,
        'recurrent_expanded_ms': avg_time_recurrent_expanded * 1000,
        'full_mcts_ms': avg_time_full * 1000,
        'time_per_sim_ms': time_per_sim * 1000,
        'estimated_time_per_move_s': estimated_time_per_move,
        'estimated_time_per_game_s': estimated_time_per_game,
        'throughput_games_per_s': batch_size / estimated_time_per_game
    }


if __name__ == "__main__":
    # Test rapide
    from model.neural_net import AbaloneModel
    from environment.env import AbaloneEnv
    import jax

    # Détecter le type de device
    device = jax.devices()[0]
    is_cpu = device.platform == 'cpu'
    is_tpu = device.platform == 'tpu'

    print(f"Device détecté: {device.platform}")

    if is_cpu:
        print("⚠️  Mode CPU détecté - Configuration réduite pour vitesse")
        num_filters = 32
        num_blocks = 3
        batch_size = 2
        num_simulations = 20
        num_warmup = 1
        num_runs = 3
    elif is_tpu:
        print("✅ Mode TPU détecté - Configuration complète")
        num_filters = 128
        num_blocks = 10
        batch_size = 8
        num_simulations = 600
        num_warmup = 3
        num_runs = 10
    else:  # GPU
        print("✅ Mode GPU détecté - Configuration intermédiaire")
        num_filters = 128
        num_blocks = 10
        batch_size = 8
        num_simulations = 200
        num_warmup = 2
        num_runs = 5

    print("Initialisation...")
    env = AbaloneEnv()
    network = AbaloneModel(num_filters=num_filters, num_blocks=num_blocks)

    # Initialiser les paramètres
    rng = jax.random.PRNGKey(42)
    sample_board = jnp.zeros((1, 9, 9), dtype=jnp.int8)
    sample_marbles = jnp.zeros((1, 2), dtype=jnp.int8)
    sample_history = jnp.zeros((1, 8, 9, 9), dtype=jnp.int8)
    sample_moves_count = jnp.zeros((1,), dtype=jnp.float32)
    params = network.init(rng, sample_board, sample_marbles, sample_history, sample_moves_count)

    print(f"\nConfiguration:")
    print(f"  - Réseau: {num_filters} filters, {num_blocks} blocks")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Simulations MCTS: {num_simulations}")
    print(f"  - Runs: {num_runs} (après {num_warmup} warmup)")

    # Benchmark
    results = benchmark_mcts_components(
        env=env,
        network=network,
        params=params,
        batch_size=batch_size,
        num_simulations=num_simulations,
        num_warmup=num_warmup,
        num_runs=num_runs
    )
