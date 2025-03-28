import numpy as np
import jax
import jax.numpy as jnp
class CPUReplayBuffer:
    def __init__(self, capacity, board_size=9, action_space=1734):
        self.capacity = capacity
        self.size = 0
        self.position = 0

        # Structures pour stocker les données sur CPU (numpy arrays)
        self.buffer = {
            'board': np.zeros((capacity, board_size, board_size), dtype=np.int8),
            'marbles_out': np.zeros((capacity, 2), dtype=np.int8),
            'policy': np.zeros((capacity, action_space), dtype=np.float32),
            'outcome': np.zeros(capacity, dtype=np.int8),
            'player': np.zeros(capacity, dtype=np.int8),
            'game_id': np.zeros(capacity, dtype=np.int32),  # ID unique de partie
            'move_num': np.zeros(capacity, dtype=np.int16),  # Numéro du coup dans la partie
            'iteration': np.zeros(capacity, dtype=np.int32),  # Itération d'entraînement
            'model_version': np.zeros(capacity, dtype=np.int32)  # Version du modèle
        }

        self.current_game_id = 0  # Compteur pour les IDs de partie

    def add(self, board, marbles_out, policy, outcome, player, game_id=None, move_num=0, 
            iteration=0, model_version=0):
        """Ajoute une transition individuelle au buffer"""
        idx = self.position

        # Convertir en numpy si nécessaire
        if hasattr(board, 'device'):  # Détecte si c'est un tableau JAX
            board = np.array(board)
        if hasattr(marbles_out, 'device'):
            marbles_out = np.array(marbles_out)
        if hasattr(policy, 'device'):
            policy = np.array(policy)

        # Si game_id n'est pas fourni, incrémenter le compteur interne
        if game_id is None:
            game_id = self.current_game_id

        # Stocker les données
        self.buffer['board'][idx] = board
        self.buffer['marbles_out'][idx] = marbles_out
        self.buffer['policy'][idx] = policy
        self.buffer['outcome'][idx] = outcome
        self.buffer['player'][idx] = player
        self.buffer['game_id'][idx] = game_id
        self.buffer['move_num'][idx] = move_num
        self.buffer['iteration'][idx] = iteration
        self.buffer['model_version'][idx] = model_version

        # Mettre à jour les compteurs
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def add_batch(self, batch):
        """Ajoute un batch de transitions"""
        batch_size = batch['board'].shape[0]

        # Vérifier la présence des champs optionnels
        has_game_id = 'game_id' in batch
        has_move_num = 'move_num' in batch
        has_iteration = 'iteration' in batch
        has_model_version = 'model_version' in batch

        for i in range(batch_size):
            game_id = batch['game_id'][i] if has_game_id else None
            move_num = batch['move_num'][i] if has_move_num else 0
            iteration = batch['iteration'][i] if has_iteration else 0
            model_version = batch['model_version'][i] if has_model_version else 0

            self.add(
                batch['board'][i],
                batch['marbles_out'][i],
                batch['policy'][i],
                batch['outcome'][i],
                batch['player'][i],
                game_id=game_id,
                move_num=move_num,
                iteration=iteration,
                model_version=model_version
            )

    def start_new_game(self):
        """Incrémente l'ID de partie pour commencer une nouvelle partie"""
        self.current_game_id += 1
        return self.current_game_id

    def sample(self, batch_size, rng_key=None):
        """Échantillonne un batch aléatoire"""
        if self.size == 0:
            raise ValueError("Buffer vide, impossible d'échantillonner")

        if rng_key is None:
            indices = np.random.randint(0, self.size, size=batch_size)
        else:
            indices = jax.random.randint(
                rng_key, shape=(batch_size,), minval=0, maxval=self.size
            ).astype(np.int32)
            indices = np.array(indices)

        batch = {
            'board': self.buffer['board'][indices],
            'marbles_out': self.buffer['marbles_out'][indices],
            'policy': self.buffer['policy'][indices],
            'outcome': self.buffer['outcome'][indices],
            'player': self.buffer['player'][indices],
            'iteration': self.buffer['iteration'][indices],
            'model_version': self.buffer['model_version'][indices]
        }

        return batch

    def sample_with_recency_bias(self, batch_size, temperature=1.0, rng_key=None):
        """Échantillonne avec priorité aux données récentes"""
        if self.size == 0:
            raise ValueError("Buffer vide, impossible d'échantillonner")

        # Calculer les poids basés sur la récence
        if self.position == 0 and self.size == self.capacity:
            # Buffer cyclique plein, la position 0 est la plus récente
            indices = np.arange(self.size)
        else:
            # Position est l'indice du prochain élément à écrire
            indices = np.arange(self.size)
            # Réorganiser pour que les indices plus élevés soient les plus récents
            indices = (indices + self.capacity - self.position) % self.capacity

        # Les indices plus élevés correspondent aux entrées plus récentes
        recency_weights = np.exp((indices / self.size) * temperature)
        sampling_probs = recency_weights / np.sum(recency_weights)

        # Échantillonner avec ces probabilités
        if rng_key is None:
            sampled_indices = np.random.choice(
                self.size, size=batch_size, p=sampling_probs, replace=True
            )
        else:
            sampled_indices = np.array(jax.random.choice(
                rng_key, self.size, shape=(batch_size,), p=sampling_probs, replace=True
            ))

        # Récupérer les échantillons
        actual_indices = np.arange(self.size)[sampled_indices]
 
        batch = {
            'board': self.buffer['board'][actual_indices],
            'marbles_out': self.buffer['marbles_out'][actual_indices],
            'policy': self.buffer['policy'][actual_indices],
            'outcome': self.buffer['outcome'][actual_indices],
            'player': self.buffer['player'][actual_indices],
            'iteration': self.buffer['iteration'][actual_indices],
            'model_version': self.buffer['model_version'][actual_indices]
        }

        return batch