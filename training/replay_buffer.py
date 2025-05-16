import numpy as np
import jax
import jax.numpy as jnp
import time
import os
import tensorflow as tf
from google.cloud import storage
from typing import Dict, List, Tuple, Any, Optional
import math
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - Process %(process)d - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("alphazero.buffer")

class CPUReplayBuffer:
    def __init__(self, capacity, board_size=9, action_space=1734):
        self.capacity = capacity
        self.size = 0
        self.position = 0

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

        self.current_game_id = 0  

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
    

# logger = logging.getLogger("alphazero.buffer")
# class GCSReplayBufferSync:
#     """
#     Buffer d'expérience synchrone utilisant Google Cloud Storage comme stockage principal.
#     - Permet le partage des expériences entre plusieurs nœuds TPU
#     - Maintient une taille fixe avec échantillonnage basé sur la récence
#     - Version synchrone pour simplicité et fiabilité
#     """
#     def __init__(self, 
#                 bucket_name: str,
#                 buffer_dir: str = 'buffer',
#                 max_local_size: int = 10000,
#                 max_buffer_size: int = 20_000_000,
#                 buffer_cleanup_threshold: float = 0.95,
#                 board_size: int = 9,
#                 action_space: int = 1734,
#                 recency_enabled: bool = True,
#                 recency_temperature: float = 0.8,
#                 cleanup_temperature: float = 2.0,
#                 log_level: str = 'INFO'):
#         """
#         Initialise le buffer d'expérience synchrone basé sur GCS.
        
#         Args:
#             bucket_name: Nom du bucket GCS
#             buffer_dir: Dossier dans le bucket pour stocker les données
#             max_local_size: Taille maximale du cache local
#             max_buffer_size: Taille maximale du buffer global (en nombre de positions)
#             buffer_cleanup_threshold: Seuil de remplissage déclenchant le nettoyage (entre 0 et 1)
#             board_size: Taille du plateau (par défaut: 9 pour Abalone 2D)
#             action_space: Nombre d'actions possibles
#             recency_enabled: Activer l'échantillonnage avec biais de récence
#             recency_temperature: Température pour le biais de récence pour l'échantillonnage
#             cleanup_temperature: Température pour l'échantillonnage lors du nettoyage
#             log_level: Niveau de logging ('INFO', 'DEBUG', 'WARNING')
#         """
#         self.bucket_name = bucket_name
#         self.buffer_dir = buffer_dir
#         self.max_local_size = max_local_size
#         self.max_buffer_size = max_buffer_size
#         self.buffer_cleanup_threshold = buffer_cleanup_threshold
#         self.board_size = board_size
#         self.action_space = action_space
#         self.recency_enabled = recency_enabled
#         self.recency_temperature = recency_temperature
#         self.cleanup_temperature = cleanup_temperature
        
#         # Configurer le niveau de log
#         self.verbose = log_level == 'DEBUG'
#         self.log_level = log_level
        
#         # Identifiant de processus et d'hôte pour éviter les conflits
#         self.process_id = jax.process_index()
#         self.host_id = f"{os.uname().nodename}_{self.process_id}"
        
#         # Cache local de données
#         self.local_buffer = {
#             'board': np.zeros((max_local_size, board_size, board_size), dtype=np.int8),
#             'marbles_out': np.zeros((max_local_size, 2), dtype=np.int8),
#             'policy': np.zeros((max_local_size, action_space), dtype=np.float32),
#             'outcome': np.zeros(max_local_size, dtype=np.int8),
#             'player': np.zeros(max_local_size, dtype=np.int8),
#             'game_id': np.zeros(max_local_size, dtype=np.int32),
#             'move_num': np.zeros(max_local_size, dtype=np.int16),
#             'iteration': np.zeros(max_local_size, dtype=np.int32),
#             'model_version': np.zeros(max_local_size, dtype=np.int32)
#         }
        
#         # Métadonnées sur le buffer
#         self.local_size = 0
#         self.position = 0
#         self.current_game_id = 0
#         self.total_size = 0  # Taille totale en comptant GCS
        
#         # Initialiser le client GCS
#         self.client = storage.Client()
#         self.bucket = self.client.bucket(bucket_name)
        
#         # Index des données disponibles sur GCS
#         self.gcs_index = {}
#         self.gcs_file_metadata = {}
#         self.last_index_update = 0
#         self.index_update_interval = 30  # Secondes avant de forcer une mise à jour de l'index
        
#         # Initialiser l'index à partir de GCS
#         self._update_gcs_index()
        
#         # Statistiques
#         self.metrics = {
#             "samples_served": 0,
#             "files_added": 0,
#             "files_removed": 0,
#             "cleanup_operations": 0
#         }
        
#         logger.info(f"GCSReplayBufferSync initialisé - Max buffer size: {self.max_buffer_size} positions")
    
#     def add(self, board, marbles_out, policy, outcome, player, game_id=None, move_num=0, 
#             iteration=0, model_version=0):
#         """Ajoute une transition individuelle au buffer"""
#         # Convertir en numpy si nécessaire
#         if hasattr(board, 'device'):
#             board = np.array(board)
#         if hasattr(marbles_out, 'device'):
#             marbles_out = np.array(marbles_out)
#         if hasattr(policy, 'device'):
#             policy = np.array(policy)
        
#         # Si game_id n'est pas fourni, incrémenter le compteur interne
#         if game_id is None:
#             game_id = self.current_game_id
        
#         # Stocker dans le cache local
#         idx = self.position
#         self.local_buffer['board'][idx] = board
#         self.local_buffer['marbles_out'][idx] = marbles_out
#         self.local_buffer['policy'][idx] = policy
#         self.local_buffer['outcome'][idx] = outcome
#         self.local_buffer['player'][idx] = player
#         self.local_buffer['game_id'][idx] = game_id
#         self.local_buffer['move_num'][idx] = move_num
#         self.local_buffer['iteration'][idx] = iteration
#         self.local_buffer['model_version'][idx] = model_version
        
#         # Mettre à jour les compteurs
#         self.position = (self.position + 1) % self.max_local_size
#         self.local_size = min(self.local_size + 1, self.max_local_size)
#         self.total_size += 1
    
#     def add_batch(self, batch):
#         """Ajoute un batch de transitions"""
#         batch_size = batch['board'].shape[0]
        
#         # Vérifier la présence des champs optionnels
#         has_game_id = 'game_id' in batch
#         has_move_num = 'move_num' in batch
#         has_iteration = 'iteration' in batch
#         has_model_version = 'model_version' in batch
        
#         for i in range(batch_size):
#             game_id = batch['game_id'][i] if has_game_id else None
#             move_num = batch['move_num'][i] if has_move_num else 0
#             iteration = batch['iteration'][i] if has_iteration else 0
#             model_version = batch['model_version'][i] if has_model_version else 0
            
#             self.add(
#                 batch['board'][i],
#                 batch['marbles_out'][i],
#                 batch['policy'][i],
#                 batch['outcome'][i],
#                 batch['player'][i],
#                 game_id=game_id,
#                 move_num=move_num,
#                 iteration=iteration,
#                 model_version=model_version
#             )
    
#     def flush_to_gcs(self):
#         """Écrit synchroniquement le contenu du buffer local sur GCS."""
#         if self.local_size == 0:
#             return 0  
        
#         logger.info(f"Début du flush vers GCS: {self.local_size} positions à écrire")
#         # else:
            
#         #     logger.info(f"Flush vers GCS: {self.local_size} positions")
        
#         # Préparer les données du buffer local
#         data_to_write = {}
#         for key in self.local_buffer:
#             data_to_write[key] = self.local_buffer[key][:self.local_size].copy()
        
#         # Compteur pour les positions écrites
#         total_written = 0
#         files_created = 0
        
#         # Générer un ID de batch unique
#         timestamp = int(time.time())
#         batch_id = f"{self.host_id}_{timestamp}"
        
#         iterations = np.unique(data_to_write['iteration'])
        
#         # Écrire les données pour chaque itération
#         for iteration in iterations:
#             # Filtrer les données pour cette itération
#             iter_mask = data_to_write['iteration'] == iteration
#             if not np.any(iter_mask):
#                 continue
            
#             # Créer un sous-ensemble pour cette itération
#             iter_data = {k: v[iter_mask] for k, v in data_to_write.items()}
#             positions_in_iter = iter_data['board'].shape[0]
            
#             # Créer le chemin dans le bucket
#             iter_path = f"{self.buffer_dir}/iteration_{iteration}"
#             file_path = f"{iter_path}/{batch_id}.tfrecord"
            
            
#             logger.info(f"Écriture de {positions_in_iter} positions pour l'itération {iteration}")
            
#             # Écrire en format TFRecord
#             example_count = self._write_tfrecord(file_path, iter_data)
#             total_written += example_count
#             files_created += 1
            
#             # Mettre à jour l'index local
#             if iteration not in self.gcs_index:
#                 self.gcs_index[iteration] = []
#             self.gcs_index[iteration].append(file_path)
            
#             # Stocker les métadonnées du fichier
#             self.gcs_file_metadata[file_path] = {
#                 'size': example_count,
#                 'timestamp': timestamp,
#                 'iteration': iteration
#             }
            
#             self.metrics["files_added"] += 1
        
#         # Réinitialiser le buffer local après écriture
#         self.local_size = 0
#         self.position = 0
        
#         # Mettre à jour la taille totale du buffer
#         self._update_total_size()
        
#         # Mettre à jour l'index après l'écriture
#         self._update_gcs_index(force=False)  # Mise à jour légère
            
#         # Vérifier si nettoyage nécessaire
#         if self.total_size > self.max_buffer_size * self.buffer_cleanup_threshold:
#             self._cleanup_buffer()
        
#         if self.verbose:
#             logger.info(f"Flush terminé: {total_written} positions dans {files_created} fichiers")
            
#         return total_written
    
#     def _write_tfrecord(self, file_path: str, data: Dict[str, np.ndarray]):
#         """Écrit les données en format TFRecord sur GCS avec métadonnées de comptage"""
#         temp_path = f"/tmp/{os.path.basename(file_path)}"
        
#         example_count = len(data['board'])
        
#         with tf.io.TFRecordWriter(temp_path) as writer:
#             for i in range(example_count):
#                 # Créer un exemple TF avec les caractéristiques
#                 example = tf.train.Example(features=tf.train.Features(feature={
#                     'board': tf.train.Feature(
#                         bytes_list=tf.train.BytesList(value=[data['board'][i].tobytes()])),
#                     'marbles_out': tf.train.Feature(
#                         bytes_list=tf.train.BytesList(value=[data['marbles_out'][i].tobytes()])),
#                     'policy': tf.train.Feature(
#                         bytes_list=tf.train.BytesList(value=[data['policy'][i].tobytes()])),
#                     'outcome': tf.train.Feature(
#                         int64_list=tf.train.Int64List(value=[data['outcome'][i]])),
#                     'player': tf.train.Feature(
#                         int64_list=tf.train.Int64List(value=[data['player'][i]])),
#                     'game_id': tf.train.Feature(
#                         int64_list=tf.train.Int64List(value=[data['game_id'][i]])),
#                     'move_num': tf.train.Feature(
#                         int64_list=tf.train.Int64List(value=[data['move_num'][i]])),
#                     'iteration': tf.train.Feature(
#                         int64_list=tf.train.Int64List(value=[data['iteration'][i]])),
#                     'model_version': tf.train.Feature(
#                         int64_list=tf.train.Int64List(value=[data['model_version'][i]]))
#                 }))
#                 writer.write(example.SerializeToString())
        
#         # Créer le dossier d'itération si nécessaire (pour GCS)
#         iter_dir = os.path.dirname(file_path)
#         try:
#             # Vérifier si le dossier existe déjà
#             check_blob = self.bucket.blob(f"{iter_dir}/.placeholder")
#             if not check_blob.exists():
#                 # Créer un marqueur de dossier
#                 placeholder = self.bucket.blob(f"{iter_dir}/.placeholder")
#                 placeholder.upload_from_string("")
#         except Exception as e:
#             logger.warning(f"Impossible de vérifier/créer le dossier {iter_dir}: {e}")
        
#         # Télécharger le fichier
#         blob = self.bucket.blob(file_path)
#         blob.metadata = {'example_count': str(example_count)}
#         blob.upload_from_filename(temp_path)
        
#         # Nettoyer
#         os.remove(temp_path)
        
#         return example_count
    
#     def _update_gcs_index(self, force=False):
#         """
#         Met à jour l'index des fichiers disponibles sur GCS.
        
#         Args:
#             force: Si True, force une mise à jour complète même si récemment mise à jour
        
#         Returns:
#             bool: True si l'index a été mis à jour avec succès
#         """
#         current_time = time.time()
        
#         # Vérifier si une mise à jour est nécessaire
#         if not force and (current_time - self.last_index_update) < self.index_update_interval:
#             return True  # Pas besoin de mise à jour
        
#         try:
#             # Lister tous les blobs dans le dossier buffer
#             prefix = f"{self.buffer_dir}/"
#             if self.verbose:
#                 logger.info(f"Mise à jour de l'index GCS pour {prefix}")
            
#             blobs = list(self.bucket.list_blobs(prefix=prefix))
            
#             if not blobs:
#                 # Vérifier si le dossier existe
#                 check_blob = self.bucket.blob(f"{prefix}.placeholder")
#                 if not check_blob.exists() and self.verbose:
#                     logger.warning(f"Le dossier {prefix} n'existe peut-être pas")
#                 return False
            
#             new_index = {}
#             new_metadata = {}
#             tfrecord_files_found = 0
#             total_examples = 0
            
#             for blob in blobs:
#                 path = blob.name
#                 if not path.endswith('.tfrecord'):
#                     continue
                
#                 tfrecord_files_found += 1
#                 parts = path.split('/')
#                 if len(parts) >= 3 and parts[-2].startswith('iteration_'):
#                     iteration = int(parts[-2].replace('iteration_', ''))
                    
#                     if iteration not in new_index:
#                         new_index[iteration] = []
                    
#                     new_index[iteration].append(path)
                    
#                     try:
#                         # Format attendu: {host_id}_{timestamp}.tfrecord
#                         file_basename = os.path.basename(path)
#                         timestamp_part = file_basename.split('_')[-1].split('.')[0]
#                         timestamp = int(timestamp_part)
#                     except (IndexError, ValueError):
#                         # Fallback si format non reconnu
#                         timestamp = int(blob.time_created.timestamp()) if hasattr(blob, 'time_created') else 0
                    
#                     # Récupérer le nombre d'exemples depuis les métadonnées
#                     if hasattr(blob, 'metadata') and blob.metadata and 'example_count' in blob.metadata:
#                         example_count = int(blob.metadata['example_count'])
#                     else:
#                         # Si pas de métadonnées, estimer (sera corrigé lors du chargement)
#                         example_count = 1000
                    
#                     total_examples += example_count
                    
#                     new_metadata[path] = {
#                         'size': example_count,
#                         'timestamp': timestamp,
#                         'iteration': iteration
#                     }
            
#             # Mettre à jour l'index uniquement s'il contient des données
#             if tfrecord_files_found > 0:
#                 # Remplacer l'index et les métadonnées
#                 self.gcs_index = new_index
#                 self.gcs_file_metadata = new_metadata
#                 self.total_size = total_examples + self.local_size
#                 self.last_index_update = current_time
                
#                 if self.verbose:
#                     iterations_found = list(new_index.keys())
#                     logger.info(f"Index GCS: {tfrecord_files_found} fichiers, {len(iterations_found)} itérations, {total_examples} positions")
                
#                 return True
#             elif self.verbose:
#                 logger.warning("Aucun fichier TFRecord trouvé dans le dossier buffer")
            
#             return False
            
#         except Exception as e:
#             logger.error(f"Erreur lors de la mise à jour de l'index GCS: {e}")
#             return False
    
#     def _update_total_size(self):
#         """Met à jour la taille totale du buffer en comptant les exemples dans les métadonnées"""
#         total = 0
        
#         # Compter à partir des métadonnées des fichiers
#         for file_path, metadata in self.gcs_file_metadata.items():
#             total += metadata['size']
        
#         # Ajouter le buffer local
#         total += self.local_size
        
#         # Mettre à jour le total
#         self.total_size = total
        
#         return total
    
#     def _cleanup_buffer(self):
#         """
#         Nettoie le buffer lorsqu'il dépasse sa taille maximale.
#         Utilise une distribution de probabilité décroissante basée sur l'âge
#         pour décider quels fichiers supprimer.
#         """
#         # Si le buffer est vide ou sous la limite, ne rien faire
#         if self.total_size <= self.max_buffer_size:
#             return
        
#         # Calculer combien d'exemples doivent être supprimés
#         overflow = self.total_size - int(self.max_buffer_size * 0.8)  # Viser 80% de remplissage
#         if overflow <= 0:
#             return
        
#         self.metrics["cleanup_operations"] += 1
#         logger.info(f"Nettoyage du buffer: besoin de supprimer {overflow}/{self.total_size} positions")
        
#         # Collecter tous les fichiers avec leurs métadonnées
#         all_files = []
        
#         for iteration, files in self.gcs_index.items():
#             for file_path in files:
#                 if file_path in self.gcs_file_metadata:
#                     metadata = self.gcs_file_metadata[file_path]
#                     all_files.append((file_path, metadata))
        
#         # Sortir si pas de fichiers
#         if not all_files:
#             return
        
#         # Trier par timestamp (du plus ancien au plus récent)
#         all_files.sort(key=lambda x: x[1]['timestamp'])
        
#         # Normaliser les âges (0 = le plus ancien, 1 = le plus récent)
#         if len(all_files) > 1:
#             oldest_time = all_files[0][1]['timestamp']
#             newest_time = all_files[-1][1]['timestamp']
#             time_range = max(1, newest_time - oldest_time)
            
#             for i in range(len(all_files)):
#                 file_path, metadata = all_files[i]
#                 timestamp = metadata['timestamp']
#                 age_normalized = 1.0 - ((timestamp - oldest_time) / time_range)  # 1 = oldest, 0 = newest
#                 all_files[i] = (file_path, metadata, age_normalized)
#         else:
#             # Un seul fichier, lui donner un âge de 0.5
#             file_path, metadata = all_files[0]
#             all_files[0] = (file_path, metadata, 0.5)
        
#         # Calculer les probabilités de suppression avec température
#         probabilities = []
#         for _, _, age in all_files:
#             # Plus l'âge est grand (plus ancien), plus la probabilité est élevée
#             prob = math.exp(age * self.cleanup_temperature)
#             probabilities.append(prob)
        
#         # Normaliser les probabilités
#         total_prob = sum(probabilities)
#         if total_prob > 0:
#             probabilities = [p / total_prob for p in probabilities]
#         else:
#             # Fallback vers distribution uniforme
#             probabilities = [1.0 / len(all_files)] * len(all_files)
        
#         # Sélectionner des fichiers à supprimer jusqu'à atteindre la limite
#         examples_removed = 0
#         files_to_remove = []
        
#         # Créer une copie pour l'échantillonnage sans remplacement
#         remaining_files = list(range(len(all_files)))
#         remaining_probs = probabilities.copy()
        
#         while examples_removed < overflow and remaining_files:
#             # Normaliser les probabilités restantes
#             total_prob = sum(remaining_probs)
#             if total_prob <= 0:
#                 break
            
#             norm_probs = [p / total_prob for p in remaining_probs]
            
#             # Sélectionner un fichier selon la distribution
#             idx = np.random.choice(len(remaining_files), p=norm_probs)
#             file_idx = remaining_files[idx]
#             file_path, metadata, _ = all_files[file_idx]
#             file_size = metadata['size']
            
#             # Ajouter à la liste de suppression
#             files_to_remove.append(file_path)
#             examples_removed += file_size
            
#             # Supprimer de la liste des candidats restants
#             del remaining_files[idx]
#             del remaining_probs[idx]
        
#         # Supprimer les fichiers sélectionnés
#         removed_count = 0
#         for file_path in files_to_remove:
#             try:
#                 blob = self.bucket.blob(file_path)
#                 blob.delete()
                
#                 # Mettre à jour l'index
#                 iteration = self.gcs_file_metadata[file_path]['iteration']
#                 if iteration in self.gcs_index and file_path in self.gcs_index[iteration]:
#                     self.gcs_index[iteration].remove(file_path)
                    
#                     # Si cette itération n'a plus de fichiers, la supprimer de l'index
#                     if not self.gcs_index[iteration]:
#                         del self.gcs_index[iteration]
                
#                 # Nettoyer les métadonnées
#                 if file_path in self.gcs_file_metadata:
#                     del self.gcs_file_metadata[file_path]
                
#                 removed_count += 1
#                 self.metrics["files_removed"] += 1
                
#             except Exception as e:
#                 logger.warning(f"Erreur lors de la suppression de {file_path}: {e}")
        
#         # Mettre à jour la taille totale
#         self._update_total_size()
        
#         logger.info(f"Nettoyage terminé: {removed_count} fichiers supprimés, nouvelle taille: {self.total_size}")
    
#     def _parse_tfrecord(self, example):
#         """Parse un exemple TFRecord en dictionnaire numpy"""
#         # Définir le schéma de fonctionnalités
#         feature_description = {
#             'board': tf.io.FixedLenFeature([], tf.string),
#             'marbles_out': tf.io.FixedLenFeature([], tf.string),
#             'policy': tf.io.FixedLenFeature([], tf.string),
#             'outcome': tf.io.FixedLenFeature([], tf.int64),
#             'player': tf.io.FixedLenFeature([], tf.int64),
#             'game_id': tf.io.FixedLenFeature([], tf.int64),
#             'move_num': tf.io.FixedLenFeature([], tf.int64),
#             'iteration': tf.io.FixedLenFeature([], tf.int64),
#             'model_version': tf.io.FixedLenFeature([], tf.int64)
#         }
        
#         parsed = tf.io.parse_single_example(example, feature_description)
        
#         return {
#             'board': tf.io.decode_raw(parsed['board'], tf.int8).numpy().reshape(self.board_size, self.board_size),
#             'marbles_out': tf.io.decode_raw(parsed['marbles_out'], tf.int8).numpy().reshape(2),
#             'policy': tf.io.decode_raw(parsed['policy'], tf.float32).numpy().reshape(self.action_space),
#             'outcome': parsed['outcome'].numpy(),
#             'player': parsed['player'].numpy(),
#             'game_id': parsed['game_id'].numpy(),
#             'move_num': parsed['move_num'].numpy(),
#             'iteration': parsed['iteration'].numpy(),
#             'model_version': parsed['model_version'].numpy()
#         }
    
#     def sample(self, batch_size, rng_key=None):
#         """
#         Échantillonne un batch de transitions du buffer global sur GCS.
        
#         Args:
#             batch_size: Nombre d'exemples à échantillonner
#             rng_key: Clé JAX pour la génération de nombres aléatoires
            
#         Returns:
#             Dict contenant les données échantillonnées
#         """
#         # Vérifier si l'index a besoin d'être mis à jour
#         current_time = time.time()
#         if current_time - self.last_index_update > self.index_update_interval:
#             self._update_gcs_index()
        
#         has_valid_data = bool(self.gcs_index)
        
#         # Si pas de données GCS ou elles sont inaccessibles, utiliser le buffer local
#         if not has_valid_data:
#             if self.local_size == 0:
#                 raise ValueError("Buffer vide (aucune donnée locale ni sur GCS)")
            
#             # Échantillonnage depuis le buffer local
#             if rng_key is None:
#                 local_indices = np.random.randint(0, self.local_size, size=batch_size)
#             else:
#                 local_indices = jax.random.randint(
#                     rng_key, shape=(batch_size,), minval=0, maxval=self.local_size
#                 ).astype(np.int32)
#                 local_indices = np.array(local_indices)
            
#             result = {}
#             for k in self.local_buffer:
#                 result[k] = self.local_buffer[k][local_indices]
            
#             self.metrics["samples_served"] += batch_size
#             return result
        
#         # Échantillonnage depuis GCS
#         try:
#             result = self._sample_from_gcs(batch_size, rng_key)
#             self.metrics["samples_served"] += batch_size
#             return result
#         except Exception as e:
#             logger.warning(f"Erreur lors de l'échantillonnage depuis GCS, fallback sur le buffer local: {e}")
            
#             # Fallback sur le buffer local si disponible
#             if self.local_size > 0:
#                 if rng_key is None:
#                     local_indices = np.random.randint(0, self.local_size, size=batch_size)
#                 else:
#                     local_indices = jax.random.randint(
#                         rng_key, shape=(batch_size,), minval=0, maxval=self.local_size
#                     ).astype(np.int32)
#                     local_indices = np.array(local_indices)
                
#                 result = {}
#                 for k in self.local_buffer:
#                     result[k] = self.local_buffer[k][local_indices]
                
#                 return result
#             else:
#                 raise ValueError("Échec de l'échantillonnage GCS et buffer local vide")
            
#     def _sample_from_gcs(self, n_samples, rng_key=None):
#         """Échantillonne des exemples depuis GCS avec biais de récence."""
#         # Construire une distribution pour l'échantillonnage des itérations
#         iterations = sorted(list(self.gcs_index.keys()))
#         if not iterations:
#             return {}
        
#         # Appliquer un biais de récence si activé
#         if self.recency_enabled and len(iterations) > 1:
#             # Normaliser les itérations entre 0 et 1
#             min_iter = min(iterations)
#             max_iter = max(iterations)
#             range_iter = max(1, max_iter - min_iter)
            
#             # Calculer les poids avec température
#             weights = [(iter_num - min_iter) / range_iter for iter_num in iterations]
#             weights = [np.exp(w * self.recency_temperature) for w in weights]
#             total_weight = sum(weights)
#             probs = [w / total_weight for w in weights]
#         else:
#             # Distribution uniforme
#             probs = [1.0 / len(iterations)] * len(iterations)
        
#         # Sélectionner les itérations
#         if rng_key is None:
#             selected_iters = np.random.choice(
#                 iterations, 
#                 size=min(3, len(iterations)), 
#                 p=probs, 
#                 replace=True
#             )
#         else:
#             rng_key, subkey = jax.random.split(rng_key)
#             selected_iters = jax.random.choice(
#                 subkey, 
#                 np.array(iterations), 
#                 shape=(min(3, len(iterations)),),
#                 p=np.array(probs),
#                 replace=True
#             )
#             selected_iters = np.array(selected_iters)
        
#         # Collecter des exemples de chaque itération sélectionnée
#         all_examples = []
#         examples_per_iter = n_samples // len(selected_iters) + 1
        
#         for iter_num in selected_iters:
#             files = self.gcs_index[iter_num]
#             if not files:
#                 continue
            
#             # Sélectionner aléatoirement quelques fichiers pour diversité
#             num_files_to_sample = min(2, len(files))
#             if rng_key is None:
#                 file_indices = np.random.choice(len(files), size=num_files_to_sample, replace=False)
#             else:
#                 rng_key, subkey = jax.random.split(rng_key)
#                 file_indices = jax.random.choice(
#                     subkey, 
#                     len(files), 
#                     shape=(num_files_to_sample,), 
#                     replace=False
#                 )
#                 file_indices = np.array(file_indices)
            
#             # Répartir les exemples entre les fichiers sélectionnés
#             examples_per_file = examples_per_iter // num_files_to_sample + 1
            
#             for file_idx in file_indices:
#                 file_path = files[int(file_idx)]
                
#                 # Charger et échantillonner des exemples de ce fichier
#                 try:
#                     examples = self._load_examples_from_gcs(file_path, examples_per_file)
#                     all_examples.extend(examples)
                    
#                     # Si nous avons assez d'exemples, arrêter l'échantillonnage
#                     if len(all_examples) >= n_samples:
#                         break
#                 except Exception as e:
#                     logger.warning(f"Erreur lors du chargement des exemples de {file_path}: {e}")
            
#             # Si nous avons assez d'exemples, arrêter l'échantillonnage
#             if len(all_examples) >= n_samples:
#                 break
        
#         # Gérer le cas où nous n'avons pas assez d'exemples
#         if not all_examples:
#             raise ValueError("Aucun exemple n'a pu être chargé depuis GCS")
            
#         if len(all_examples) < n_samples:
#             # Dupliquer des exemples existants pour atteindre la taille demandée
#             if all_examples:  
#                 indices_to_duplicate = np.random.choice(
#                     len(all_examples), size=n_samples-len(all_examples), replace=True)
                
#                 for idx in indices_to_duplicate:
#                     all_examples.append(all_examples[idx])
#         elif len(all_examples) > n_samples:
#             # Tronquer si trop d'exemples
#             all_examples = all_examples[:n_samples]
        
#         # Consolider les exemples en un seul dict
#         result = {}
#         for k in all_examples[0].keys():
#             result[k] = np.array([ex[k] for ex in all_examples])
        
#         return result

#     def _load_examples_from_gcs(self, file_path, max_examples):
#         """Charge des exemples depuis un fichier TFRecord sur GCS"""
#         blob = self.bucket.blob(file_path)
#         temp_path = f"/tmp/{os.path.basename(file_path)}"
#         blob.download_to_filename(temp_path)
        
#         raw_dataset = tf.data.TFRecordDataset(temp_path)
        
#         examples = []
#         for i, raw_example in enumerate(raw_dataset):
#             if i >= max_examples:
#                 break
#             example = self._parse_tfrecord(raw_example)
#             examples.append(example)
        
#         os.remove(temp_path)
        
#         return examples
    
#     def sample_with_recency_bias(self, batch_size, temperature=None, rng_key=None):
#         """
#         Échantillonne avec biais de récence depuis GCS.
        
#         Args:
#             batch_size: Nombre d'exemples à échantillonner
#             temperature: Température pour le biais de récence (None pour utiliser la valeur par défaut)
#             rng_key: Clé JAX pour la génération de nombres aléatoires
            
#         Returns:
#             Dict contenant les données échantillonnées
#         """
#         original_temp = self.recency_temperature
#         if temperature is not None:
#             self.recency_temperature = temperature
        
#         result = self.sample(batch_size, rng_key)
        
#         if temperature is not None:
#             self.recency_temperature = original_temp
        
#         return result
    
#     def start_new_game(self):
#         """Incrémente l'ID de partie pour commencer une nouvelle partie"""
#         self.current_game_id += 1
#         return self.current_game_id
    
#     def get_stats(self):
#         """Renvoie des statistiques sur le buffer"""
#         stats = {
#             "total_size": self.total_size,
#             "local_size": self.local_size,
#             "max_size": self.max_buffer_size,
#             "fill_percentage": 100 * self.total_size / self.max_buffer_size if self.max_buffer_size > 0 else 0,
#             "iterations": len(self.gcs_index),
#             "files": sum(len(files) for files in self.gcs_index.values()),
#         }
        
#         stats.update(self.metrics)
        
#         return stats
    
#     def close(self):
#         """Ferme proprement le buffer et assure que toutes les données sont écrites"""
#         if self.local_size > 0:
#             positions_flushed = self.flush_to_gcs()
#             logger.info(f"Flush final: {positions_flushed} positions écrites sur GCS")
#         else :
#             logger.info(f"Local empty")
        
#         logger.info(f"Buffer GCS fermé. Total: {self.total_size} positions")
    
#     def __del__(self):
#         """Destructeur pour assurer la fermeture propre"""
#         try:
#             self.close()
#         except:
#             pass


# import time
# import logging
# import os
# import pickle # Si vous l'utilisez ailleurs, sinon pas nécessaire pour cette classe directement
# from functools import partial # Non utilisé directement ici, mais peut-être dans votre projet
# import numpy as np
# import tensorflow as tf # Requis pour TFRecordWriter et parsing
# from google.cloud import storage # Requis pour l'interaction GCS
# import jax # Requis pour jax.process_index()
import math # Requis pour math.exp dans _cleanup_buffer

# Configuration du logger (peut aussi être fait globalement dans votre application)
#logger = logging.getLogger("alphazero.buffer")
# Si vous n'avez pas de configuration globale, vous pouvez en ajouter une simple ici pour les tests :
# if not logger.hasHandlers():
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - PID %(process)d - %(levelname)s - %(message)s')
logger = logging.getLogger("alphazero.buffer")
class GCSReplayBufferSync:
    """
    Buffer d'expérience synchrone utilisant Google Cloud Storage comme stockage principal.
    - Permet le partage des expériences entre plusieurs nœuds TPU
    - Maintient une taille fixe avec échantillonnage basé sur la récence
    - Version synchrone pour simplicité et fiabilité
    """
    def __init__(self,
                 bucket_name: str,
                 buffer_dir: str = 'buffer',
                 max_local_size: int = 10000,
                 max_buffer_size: int = 20_000_000,
                 buffer_cleanup_threshold: float = 0.95,
                 board_size: int = 9,
                 action_space: int = 1734,
                 recency_enabled: bool = True,
                 recency_temperature: float = 0.8,
                 cleanup_temperature: float = 2.0,
                 log_level: str = 'INFO'): # log_level est pour info, pas pour configurer le logger ici
        """
        Initialise le buffer d'expérience synchrone basé sur GCS.
        """
        self.bucket_name = bucket_name
        self.buffer_dir = buffer_dir
        self.max_local_size = max_local_size
        self.max_buffer_size = max_buffer_size
        self.buffer_cleanup_threshold = buffer_cleanup_threshold
        self.board_size = board_size
        self.action_space = action_space
        self.recency_enabled = recency_enabled
        self.recency_temperature = recency_temperature
        self.cleanup_temperature = cleanup_temperature
        
        self.process_id = jax.process_index()
        self.host_id = f"{os.uname().nodename}_{self.process_id}"
        
        self.local_buffer = {
            'board': np.zeros((max_local_size, board_size, board_size), dtype=np.int8),
            'marbles_out': np.zeros((max_local_size, 2), dtype=np.int8),
            'policy': np.zeros((max_local_size, action_space), dtype=np.float32),
            'outcome': np.zeros(max_local_size, dtype=np.int8),
            'player': np.zeros(max_local_size, dtype=np.int8),
            'game_id': np.zeros(max_local_size, dtype=np.int32),
            'move_num': np.zeros(max_local_size, dtype=np.int16),
            'iteration': np.zeros(max_local_size, dtype=np.int32),
            'model_version': np.zeros(max_local_size, dtype=np.int32)
        }
        
        self.local_size = 0
        self.position = 0
        self.current_game_id = 0 # Utilisé si game_id n'est pas fourni à add()
        self.total_size = 0 
        
        self.client = storage.Client()
        self.bucket = self.client.bucket(bucket_name)
        
        self.gcs_index = {}
        self.gcs_file_metadata = {}
        self.last_index_update = 0
        self.index_update_interval = 30 
        
        logger.info(f"[P:{self.process_id}] GCSReplayBufferSync initializing...")
        t_init_idx_start = time.time()
        self._update_gcs_index(force=True) # Force update at init
        logger.info(f"[P:{self.process_id}] GCSReplayBufferSync: Initial _update_gcs_index took {time.time() - t_init_idx_start:.4f}s")
        
        self.metrics = {
            "samples_served": 0,
            "files_added": 0,
            "files_removed": 0,
            "cleanup_operations": 0
        }
        
        logger.info(f"[P:{self.process_id}] GCSReplayBufferSync initialized. Max buffer: {self.max_buffer_size}, Max local: {self.max_local_size}, Total GCS (est.): {self.total_size}")

    def add(self, board, marbles_out, policy, outcome, player, game_id=None, move_num=0, 
            iteration=0, model_version=0):
        if hasattr(board, 'device'): board = np.array(board)
        if hasattr(marbles_out, 'device'): marbles_out = np.array(marbles_out)
        if hasattr(policy, 'device'): policy = np.array(policy)
        
        _game_id = game_id if game_id is not None else self.current_game_id
        
        idx = self.position
        self.local_buffer['board'][idx] = board
        self.local_buffer['marbles_out'][idx] = marbles_out
        self.local_buffer['policy'][idx] = policy
        self.local_buffer['outcome'][idx] = outcome
        self.local_buffer['player'][idx] = player
        self.local_buffer['game_id'][idx] = _game_id
        self.local_buffer['move_num'][idx] = move_num
        self.local_buffer['iteration'][idx] = iteration
        self.local_buffer['model_version'][idx] = model_version
        
        self.position = (self.position + 1) % self.max_local_size
        self.local_size = min(self.local_size + 1, self.max_local_size)
        # self.total_size increment was here, but it's more accurately updated after GCS operations

    def add_batch(self, batch):
        batch_size_val = batch['board'].shape[0]
        has_game_id = 'game_id' in batch
        has_move_num = 'move_num' in batch
        has_iteration = 'iteration' in batch
        has_model_version = 'model_version' in batch
        
        for i in range(batch_size_val):
            game_id = batch['game_id'][i] if has_game_id else None
            move_num = batch['move_num'][i] if has_move_num else 0
            iteration_val = batch['iteration'][i] if has_iteration else 0 # Renamed to avoid conflict
            model_version = batch['model_version'][i] if has_model_version else 0
            
            self.add(
                batch['board'][i], batch['marbles_out'][i], batch['policy'][i],
                batch['outcome'][i], batch['player'][i], game_id=game_id,
                move_num=move_num, iteration=iteration_val, model_version=model_version
            )

    def flush_to_gcs(self):
        t_flush_start = time.time()
        if self.local_size == 0:
            logger.info(f"[P:{self.process_id}] flush_to_gcs: No local data to flush.")
            return 0

        logger.info(f"[P:{self.process_id}] flush_to_gcs: Starting flush for {self.local_size} local positions.")
        
        data_to_write = {}
        for key in self.local_buffer:
            data_to_write[key] = self.local_buffer[key][:self.local_size].copy()
        
        current_local_size_flushing = self.local_size # Store for accurate accounting
        total_written_this_flush = 0
        files_created_this_flush = 0
        
        flush_timestamp = int(time.time())
        batch_id_suffix = f"{self.host_id}_{flush_timestamp}"
        
        unique_iterations_in_local = np.unique(data_to_write['iteration'])
        
        t_write_loop_start = time.time()
        for iteration_val in unique_iterations_in_local:
            iter_mask = data_to_write['iteration'] == iteration_val
            if not np.any(iter_mask):
                continue
            
            iter_data = {k: v[iter_mask] for k, v in data_to_write.items()}
            positions_in_iter_file = iter_data['board'].shape[0]
            
            iter_path_on_gcs = f"{self.buffer_dir}/iteration_{iteration_val}"
            file_path_on_gcs = f"{iter_path_on_gcs}/{batch_id_suffix}.tfrecord"
            
            logger.info(f"[P:{self.process_id}] flush_to_gcs: Writing {positions_in_iter_file} positions for iteration {iteration_val} to {file_path_on_gcs}")
            t_tfrecord_write_start = time.time()
            example_count = self._write_tfrecord(file_path_on_gcs, iter_data)
            t_tfrecord_write_end = time.time()
            logger.info(f"[P:{self.process_id}] flush_to_gcs: _write_tfrecord for iter {iteration_val} ({example_count} pos) to {file_path_on_gcs} took {t_tfrecord_write_end - t_tfrecord_write_start:.4f}s")

            if example_count > 0:
                total_written_this_flush += example_count
                files_created_this_flush += 1
                
                if iteration_val not in self.gcs_index: self.gcs_index[iteration_val] = []
                self.gcs_index[iteration_val].append(file_path_on_gcs)
                self.gcs_file_metadata[file_path_on_gcs] = {
                    'size': example_count, 'timestamp': flush_timestamp, 'iteration': iteration_val
                }
                self.metrics["files_added"] += 1
        
        logger.info(f"[P:{self.process_id}] flush_to_gcs: Loop for _write_tfrecord calls took {time.time() - t_write_loop_start:.4f}s for {files_created_this_flush} files.")

        self.local_size = 0
        self.position = 0
        
        # Update total_size with what was actually written and known GCS state
        t_update_total_start = time.time()
        self._update_total_size() # This recalculates based on self.gcs_file_metadata
        logger.info(f"[P:{self.process_id}] flush_to_gcs: _update_total_size took {time.time() - t_update_total_start:.4f}s. New total_size: {self.total_size}")
        
        # Light index update (adds new files, doesn't re-list all of GCS unless interval passed)
        t_update_idx_start = time.time()
        self._update_gcs_index(force=False) 
        logger.info(f"[P:{self.process_id}] flush_to_gcs: _update_gcs_index(force=False) took {time.time() - t_update_idx_start:.4f}s. Current total_size: {self.total_size}")
            
        if self.total_size > self.max_buffer_size * self.buffer_cleanup_threshold:
            logger.info(f"[P:{self.process_id}] flush_to_gcs: Cleanup needed. total_size ({self.total_size}) > threshold ({self.max_buffer_size * self.buffer_cleanup_threshold}).")
            t_cleanup_start = time.time()
            self._cleanup_buffer()
            logger.info(f"[P:{self.process_id}] flush_to_gcs: _cleanup_buffer took {time.time() - t_cleanup_start:.4f}s. New total_size: {self.total_size}")
        
        logger.info(f"[P:{self.process_id}] flush_to_gcs: Finished flush. Total {total_written_this_flush} positions in {files_created_this_flush} files. Total GCSReplayBuffer.flush_to_gcs time: {time.time() - t_flush_start:.4f}s")
        return total_written_this_flush

    def _write_tfrecord(self, file_path_on_gcs: str, data: dict):
        t_start_write_tf = time.time()
        logger.info(f"[P:{self.process_id}] _write_tfrecord: Starting to write to GCS path {file_path_on_gcs}")
        
        # Ensure /tmp directory exists or handle errors
        tmp_dir = "/tmp"
        if not os.path.exists(tmp_dir):
            try:
                os.makedirs(tmp_dir, exist_ok=True)
            except OSError as e:
                logger.error(f"[P:{self.process_id}] _write_tfrecord: Failed to create /tmp directory: {e}")
                return 0 # Cannot proceed without /tmp

        temp_local_path = os.path.join(tmp_dir, os.path.basename(file_path_on_gcs))
        
        example_count = len(data['board'])
        if example_count == 0:
            logger.warning(f"[P:{self.process_id}] _write_tfrecord: No examples to write for {file_path_on_gcs}.")
            return 0
            
        try:
            with tf.io.TFRecordWriter(temp_local_path) as writer:
                for i in range(example_count):
                    example = tf.train.Example(features=tf.train.Features(feature={
                        'board': tf.train.Feature(bytes_list=tf.train.BytesList(value=[data['board'][i].tobytes()])),
                        'marbles_out': tf.train.Feature(bytes_list=tf.train.BytesList(value=[data['marbles_out'][i].tobytes()])),
                        'policy': tf.train.Feature(bytes_list=tf.train.BytesList(value=[data['policy'][i].tobytes()])),
                        'outcome': tf.train.Feature(int64_list=tf.train.Int64List(value=[data['outcome'][i].item()])), # Use .item() for numpy scalars
                        'player': tf.train.Feature(int64_list=tf.train.Int64List(value=[data['player'][i].item()])),
                        'game_id': tf.train.Feature(int64_list=tf.train.Int64List(value=[data['game_id'][i].item()])),
                        'move_num': tf.train.Feature(int64_list=tf.train.Int64List(value=[data['move_num'][i].item()])),
                        'iteration': tf.train.Feature(int64_list=tf.train.Int64List(value=[data['iteration'][i].item()])),
                        'model_version': tf.train.Feature(int64_list=tf.train.Int64List(value=[data['model_version'][i].item()]))
                    }))
                    writer.write(example.SerializeToString())
        except Exception as e:
            logger.error(f"[P:{self.process_id}] _write_tfrecord: Failed to write local TFRecord {temp_local_path}: {e}", exc_info=True)
            return 0 # Failed to write locally

        # GCS Upload
        gcs_blob = self.bucket.blob(file_path_on_gcs)
        gcs_blob.metadata = {'example_count': str(example_count)}
        
        t_upload_start = time.time()
        try:
            gcs_blob.upload_from_filename(temp_local_path)
            t_upload_end = time.time()
            logger.info(f"[P:{self.process_id}] _write_tfrecord: GCS upload of {temp_local_path} to {file_path_on_gcs} ({example_count} pos) took {t_upload_end - t_upload_start:.4f}s")
        except Exception as e:
            logger.error(f"[P:{self.process_id}] _write_tfrecord: GCS upload failed for {file_path_on_gcs}: {e}", exc_info=True)
            # Attempt to clean up local file even if upload fails
            if os.path.exists(temp_local_path): os.remove(temp_local_path)
            return 0 # Upload failed

        if os.path.exists(temp_local_path): os.remove(temp_local_path)
        logger.info(f"[P:{self.process_id}] _write_tfrecord: Finished writing {file_path_on_gcs}. Total _write_tfrecord time: {time.time() - t_start_write_tf:.4f}s")
        return example_count

    def _update_gcs_index(self, force=False):
        t_start_update_idx = time.time()
        current_time = time.time()
        
        if not force and (current_time - self.last_index_update) < self.index_update_interval:
            logger.info(f"[P:{self.process_id}] _update_gcs_index: Skipped due to interval. Last update {current_time - self.last_index_update:.2f}s ago.")
            return True
        
        logger.info(f"[P:{self.process_id}] _update_gcs_index: Starting index update (force={force}).")
        updated_successfully = False
        try:
            prefix_to_list = f"{self.buffer_dir}/"
            
            t_list_blobs_start = time.time()
            all_blobs_iterator = self.bucket.list_blobs(prefix=prefix_to_list)
            # Convert iterator to list carefully if memory is a concern for millions of files
            # For very large buckets, consider paginated listing or alternative index
            blobs_list = list(all_blobs_iterator) 
            t_list_blobs_end = time.time()
            logger.info(f"[P:{self.process_id}] _update_gcs_index: list_blobs('{prefix_to_list}') found {len(blobs_list)} items, took {t_list_blobs_end - t_list_blobs_start:.4f}s")
            
            if not blobs_list and prefix_to_list == f"{self.buffer_dir}/": # Check if buffer_dir itself might not exist
                 # Attempt to check if the bucket/prefix is truly empty or if listing failed
                try:
                    # Check if the bucket itself is accessible by trying to get its metadata
                    self.bucket.reload() 
                    logger.info(f"[P:{self.process_id}] _update_gcs_index: Bucket {self.bucket_name} accessible. Prefix {prefix_to_list} is likely empty or contains no TFRecords.")
                except Exception as bucket_err:
                    logger.error(f"[P:{self.process_id}] _update_gcs_index: Failed to access bucket {self.bucket_name}. Error: {bucket_err}")
                    self.last_index_update = current_time # Mark update attempt
                    logger.info(f"[P:{self.process_id}] _update_gcs_index: Finished (bucket access error). Total time: {time.time() - t_start_update_idx:.4f}s")
                    return False # Cannot proceed if bucket not accessible

            new_index_map = {}
            new_metadata_map = {}
            tfrecord_files_count = 0
            total_gcs_examples = 0
            
            for blob_item in blobs_list:
                path = blob_item.name
                if not path.endswith('.tfrecord'): continue
                
                tfrecord_files_count += 1
                parts = path.split('/')
                try: # Robust parsing of iteration from path
                    iteration_str = next(p for p in reversed(parts) if p.startswith('iteration_'))
                    iteration_val = int(iteration_str.replace('iteration_', ''))
                except (StopIteration, ValueError):
                    logger.warning(f"[P:{self.process_id}] _update_gcs_index: Could not parse iteration from path {path}. Skipping.")
                    continue
                
                if iteration_val not in new_index_map: new_index_map[iteration_val] = []
                new_index_map[iteration_val].append(path)
                
                try:
                    filename_base = os.path.basename(path)
                    timestamp_val = int(filename_base.split('_')[-1].split('.')[0])
                except (IndexError, ValueError):
                    timestamp_val = int(blob_item.time_created.timestamp()) if blob_item.time_created else 0
                
                example_count = 0
                if hasattr(blob_item, 'metadata') and blob_item.metadata and 'example_count' in blob_item.metadata:
                    try:
                        example_count = int(blob_item.metadata['example_count'])
                    except ValueError:
                        logger.warning(f"[P:{self.process_id}] _update_gcs_index: Invalid example_count in metadata for {path}. Estimating as 0.")
                        example_count = 0 # Or re-evaluate by downloading, but that's slow here.
                else: # Fallback: try to get from filename, or estimate (less reliable)
                    logger.warning(f"[P:{self.process_id}] _update_gcs_index: Missing 'example_count' metadata for {path}. Estimating as 0.")
                    example_count = 0 

                total_gcs_examples += example_count
                new_metadata_map[path] = {'size': example_count, 'timestamp': timestamp_val, 'iteration': iteration_val}
            
            if tfrecord_files_count > 0:
                self.gcs_index = new_index_map
                self.gcs_file_metadata = new_metadata_map
                # total_size should reflect GCS only here, local_size added separately or in get_stats
                self.total_size = total_gcs_examples 
                logger.info(f"[P:{self.process_id}] _update_gcs_index: Index updated. {tfrecord_files_count} TFRecords, {len(new_index_map)} unique iterations, {total_gcs_examples} total GCS positions.")
                updated_successfully = True
            elif not blobs_list: # No blobs found at all
                 logger.info(f"[P:{self.process_id}] _update_gcs_index: No blobs found under prefix {prefix_to_list}. Index is empty.")
                 self.gcs_index = {}
                 self.gcs_file_metadata = {}
                 self.total_size = 0
                 updated_successfully = True # Empty is a valid state
            else: # Blobs found, but no .tfrecord files
                 logger.warning(f"[P:{self.process_id}] _update_gcs_index: No TFRecord files found among {len(blobs_list)} items. Index is empty.")
                 self.gcs_index = {}
                 self.gcs_file_metadata = {}
                 self.total_size = 0
                 updated_successfully = True


        except Exception as e:
            logger.error(f"[P:{self.process_id}] _update_gcs_index: Error during full update: {e}", exc_info=True)
            updated_successfully = False # Mark as failed
        
        self.last_index_update = current_time # Mark update attempt time regardless of outcome
        logger.info(f"[P:{self.process_id}] _update_gcs_index: Finished (success={updated_successfully}). Total time: {time.time() - t_start_update_idx:.4f}s")
        return updated_successfully

    def _update_total_size(self):
        # This method recalculates total_size based on gcs_file_metadata.
        # It should NOT include local_size here, as local_size is transient before a flush.
        # The true "global" size visible for sampling is what's on GCS.
        # self.total_size will be updated in _update_gcs_index and _cleanup_buffer directly.
        # flush_to_gcs updates its local view of total_size after calling _update_gcs_index.
        # get_stats() can provide a combined view if needed.
        current_gcs_size = sum(meta['size'] for meta in self.gcs_file_metadata.values())
        self.total_size = current_gcs_size # Reflects GCS persisted state
        return self.total_size


    def _cleanup_buffer(self):
        t_start_cleanup = time.time()
        # Recalculate GCS size from metadata to be sure, as local_size is not part of persisted state for cleanup
        current_gcs_persisted_size = sum(meta['size'] for meta in self.gcs_file_metadata.values())
        
        logger.info(f"[P:{self.process_id}] _cleanup_buffer: Starting. GCS persisted size: {current_gcs_persisted_size}, Max buffer: {self.max_buffer_size}, Target fill: 80%")

        if current_gcs_persisted_size <= self.max_buffer_size * 0.8: # Check against target fill
            logger.info(f"[P:{self.process_id}] _cleanup_buffer: No cleanup needed (GCS size {current_gcs_persisted_size} is within 80% target of max {self.max_buffer_size}).")
            return

        self.metrics["cleanup_operations"] += 1
        # Target 80% fill after cleanup
        amount_to_remove = current_gcs_persisted_size - int(self.max_buffer_size * 0.8)
        if amount_to_remove <= 0:
            logger.info(f"[P:{self.process_id}] _cleanup_buffer: No cleanup needed (amount_to_remove={amount_to_remove}).")
            return

        logger.info(f"[P:{self.process_id}] _cleanup_buffer: Need to remove ~{amount_to_remove} positions from GCS current size {current_gcs_persisted_size}.")
        
        all_files_meta = list(self.gcs_file_metadata.items()) # List of (file_path, metadata_dict)
        if not all_files_meta:
            logger.warning(f"[P:{self.process_id}] _cleanup_buffer: No files in GCS metadata to clean up.")
            return
        
        # Sort by timestamp (oldest first) for recency-based deletion probability
        all_files_meta.sort(key=lambda x: x[1]['timestamp'])
        
        logger.info(f"[P:{self.process_id}] _cleanup_buffer: Found {len(all_files_meta)} files in GCS metadata for potential deletion.")

        # Prepare files with normalized age for probability calculation
        files_with_age = []
        if len(all_files_meta) > 1:
            oldest_time = all_files_meta[0][1]['timestamp']
            newest_time = all_files_meta[-1][1]['timestamp']
            time_span = float(max(1, newest_time - oldest_time))
            for path, meta in all_files_meta:
                # Normalized age: 0 for newest, 1 for oldest
                norm_age = 1.0 - ((meta['timestamp'] - oldest_time) / time_span)
                files_with_age.append({'path': path, 'size': meta['size'], 'age': norm_age})
        elif len(all_files_meta) == 1:
             files_with_age.append({'path': all_files_meta[0][0], 'size': all_files_meta[0][1]['size'], 'age': 0.5})


        # Calculate deletion probabilities (higher probability for older files)
        probs = np.array([math.exp(f['age'] * self.cleanup_temperature) for f in files_with_age])
        if np.sum(probs) == 0 : # Avoid division by zero if all probs are zero (e.g. large negative temp)
            if len(probs)>0: probs = np.ones(len(probs)) / len(probs) # Uniform if exp fails
            else: 
                logger.warning(f"[P:{self.process_id}] _cleanup_buffer: No files or probabilities for cleanup.")
                return

        probs /= np.sum(probs)
        
        files_to_delete_paths = []
        examples_marked_for_removal = 0
        
        # Shuffle available files according to probabilities for selection
        # Use a copy for modification if needed, or sample indices
        candidate_indices = np.arange(len(files_with_age))
        
        # Keep selecting files until enough examples are marked for removal
        # This loop ensures we don't get stuck if a file is huge and overshoots target a lot
        # Or if probabilities are skewed. Simpler: iterate through oldest first for aggressive cleanup.
        # For now, respecting the probability distribution for selection:
        
        # Create a list of files sorted by probability (descending) to prioritize deletion
        # This is a greedy approach based on probability; true random sampling would be different.
        sorted_candidates_indices = np.argsort([-p for p in probs]) # Sort by descending probability

        for idx in sorted_candidates_indices:
            if examples_marked_for_removal >= amount_to_remove:
                break
            file_info = files_with_age[idx]
            files_to_delete_paths.append(file_info['path'])
            examples_marked_for_removal += file_info['size']
        
        logger.info(f"[P:{self.process_id}] _cleanup_buffer: Selected {len(files_to_delete_paths)} files to remove, targeting {amount_to_remove} examples, marked {examples_marked_for_removal} for removal.")

        deleted_files_count = 0
        actually_removed_examples_count = 0
        t_delete_loop_start = time.time()

        # Perform batch deletion if possible, otherwise individual
        # For simplicity, using individual delete as in original code.
        # Batch delete would require grouping and specific GCS API calls.
        for file_path_to_delete in files_to_delete_paths:
            try:
                file_metadata = self.gcs_file_metadata.get(file_path_to_delete)
                file_size_for_log = file_metadata['size'] if file_metadata else 0
                
                t_delete_single_start = time.time()
                blob_to_delete = self.bucket.blob(file_path_to_delete)
                blob_to_delete.delete() # Actual GCS call
                logger.info(f"[P:{self.process_id}] _cleanup_buffer: Deleted {file_path_to_delete} ({file_size_for_log} pos) in {time.time() - t_delete_single_start:.4f}s")
                
                if file_metadata: # Update local index and metadata
                    iteration_of_deleted = file_metadata['iteration']
                    if iteration_of_deleted in self.gcs_index and file_path_to_delete in self.gcs_index[iteration_of_deleted]:
                        self.gcs_index[iteration_of_deleted].remove(file_path_to_delete)
                        if not self.gcs_index[iteration_of_deleted]:
                            del self.gcs_index[iteration_of_deleted]
                    del self.gcs_file_metadata[file_path_to_delete]
                    actually_removed_examples_count += file_size_for_log
                
                deleted_files_count += 1
                self.metrics["files_removed"] += 1
            except Exception as e:
                logger.warning(f"[P:{self.process_id}] _cleanup_buffer: Error deleting {file_path_to_delete}: {e}", exc_info=True)
        
        logger.info(f"[P:{self.process_id}] _cleanup_buffer: Loop for GCS file deletions took {time.time() - t_delete_loop_start:.4f}s")
        
        self._update_total_size() # Recalculate GCS persisted size
        logger.info(f"[P:{self.process_id}] _cleanup_buffer: Finished. {deleted_files_count} files removed ({actually_removed_examples_count} examples). New GCS persisted size: {self.total_size}. Total _cleanup_buffer time: {time.time() - t_start_cleanup:.4f}s")

    def _parse_tfrecord(self, example_proto):
        feature_description = {
            'board': tf.io.FixedLenFeature([], tf.string),
            'marbles_out': tf.io.FixedLenFeature([], tf.string),
            'policy': tf.io.FixedLenFeature([], tf.string),
            'outcome': tf.io.FixedLenFeature([], tf.int64),
            'player': tf.io.FixedLenFeature([], tf.int64),
            'game_id': tf.io.FixedLenFeature([], tf.int64),
            'move_num': tf.io.FixedLenFeature([], tf.int64),
            'iteration': tf.io.FixedLenFeature([], tf.int64),
            'model_version': tf.io.FixedLenFeature([], tf.int64)
        }
        parsed = tf.io.parse_single_example(example_proto, feature_description)
        return {
            'board': tf.io.decode_raw(parsed['board'], tf.int8).numpy().reshape(self.board_size, self.board_size),
            'marbles_out': tf.io.decode_raw(parsed['marbles_out'], tf.int8).numpy().reshape(2),
            'policy': tf.io.decode_raw(parsed['policy'], tf.float32).numpy().reshape(self.action_space),
            'outcome': parsed['outcome'].numpy().astype(np.int8),
            'player': parsed['player'].numpy().astype(np.int8),
            'game_id': parsed['game_id'].numpy().astype(np.int32),
            'move_num': parsed['move_num'].numpy().astype(np.int16),
            'iteration': parsed['iteration'].numpy().astype(np.int32),
            'model_version': parsed['model_version'].numpy().astype(np.int32)
        }

    def sample(self, batch_size, rng_key=None):
        t_sample_start = time.time()
        logger.info(f"[P:{self.process_id}] sample: Requesting {batch_size} samples.")
        
        current_time_for_sample = time.time()
        if current_time_for_sample - self.last_index_update > self.index_update_interval:
            logger.info(f"[P:{self.process_id}] sample: GCS index update triggered by interval ({self.index_update_interval}s). Last update {current_time_for_sample - self.last_index_update:.2f}s ago.")
            t_idx_update_start_in_sample = time.time()
            self._update_gcs_index() 
            logger.info(f"[P:{self.process_id}] sample: _update_gcs_index call took {time.time() - t_idx_update_start_in_sample:.4f}s")
        
        gcs_has_data = bool(self.gcs_index) and self.total_size > 0 # self.total_size reflects GCS persisted data
        
        sampled_data = None
        source = "N/A"

        if not gcs_has_data:
            logger.warning(f"[P:{self.process_id}] sample: No GCS data available (index empty or total_size is 0). Attempting local buffer (size: {self.local_size}).")
            if self.local_size == 0:
                logger.error(f"[P:{self.process_id}] sample: Buffer completely empty (no local data, no GCS data).")
                raise ValueError("Buffer empty (no data locally or on GCS)")
            
            indices = np.random.randint(0, self.local_size, size=batch_size) if rng_key is None else np.array(jax.random.randint(rng_key, (batch_size,), 0, self.local_size))
            sampled_data = {k: self.local_buffer[k][indices] for k in self.local_buffer}
            source = "local_buffer_fallback (no GCS data)"
        else: # Try GCS
            try:
                t_gcs_sample_start = time.time()
                sampled_data = self._sample_from_gcs(batch_size, rng_key)
                logger.info(f"[P:{self.process_id}] sample: _sample_from_gcs for {batch_size} samples took {time.time() - t_gcs_sample_start:.4f}s")
                source = "GCS"
            except Exception as e:
                logger.warning(f"[P:{self.process_id}] sample: Error sampling from GCS: {e}. Falling back to local buffer (size: {self.local_size}).", exc_info=True)
                if self.local_size > 0:
                    indices = np.random.randint(0, self.local_size, size=batch_size) if rng_key is None else np.array(jax.random.randint(rng_key, (batch_size,), 0, self.local_size))
                    sampled_data = {k: self.local_buffer[k][indices] for k in self.local_buffer}
                    source = f"local_buffer_fallback (GCS sample error)"
                else:
                    logger.error(f"[P:{self.process_id}] sample: GCS sample failed AND local buffer is empty.")
                    raise ValueError("GCS sampling failed and local buffer empty")

        if sampled_data: self.metrics["samples_served"] += batch_size
        logger.info(f"[P:{self.process_id}] sample: Finished. Got {batch_size} samples from {source}. Total sample() time: {time.time() - t_sample_start:.4f}s")
        return sampled_data

    def _sample_from_gcs(self, n_samples, rng_key=None):
        t_start_sample_gcs = time.time()
        # ... (logique de sélection d'itérations et de fichiers telle que vous l'aviez)
        #       Ajoutez des logs si vous voulez tracer quelles itérations/fichiers sont choisis
        iterations = sorted(list(self.gcs_index.keys()))
        if not iterations:
            logger.warning(f"[P:{self.process_id}] _sample_from_gcs: No iterations in GCS index.")
            raise ValueError("No iterations in GCS index to sample from")

        # (Recency bias logic as before)
        if self.recency_enabled and len(iterations) > 1:
            min_iter, max_iter = min(iterations), max(iterations)
            iter_range = float(max(1, max_iter - min_iter))
            weights = np.array([math.exp(((i - min_iter) / iter_range) * self.recency_temperature) for i in iterations])
            probs = weights / np.sum(weights)
        else:
            probs = np.ones(len(iterations)) / len(iterations)

        num_iters_to_sample_from = min(3, len(iterations)) # Sample from up to 3 different iteration groups
        
        if rng_key is None:
            chosen_iteration_indices = np.random.choice(len(iterations), size=num_iters_to_sample_from, p=probs, replace=True)
            selected_iteration_values = [iterations[i] for i in chosen_iteration_indices]
        else:
            rng_key, iter_choice_key = jax.random.split(rng_key)
            chosen_iteration_indices = jax.random.choice(iter_choice_key, np.arange(len(iterations)), shape=(num_iters_to_sample_from,), p=probs, replace=True)
            selected_iteration_values = [iterations[i] for i in chosen_iteration_indices]
        
        logger.info(f"[P:{self.process_id}] _sample_from_gcs: Selected iterations to sample from: {selected_iteration_values} with probs {probs.round(3)}")

        all_loaded_examples = []
        # Aim to get roughly even samples from each selected iteration's files
        # This logic might need refinement for precise distribution
        target_samples_per_iteration_group = n_samples // num_iters_to_sample_from + 1 

        for iter_val in selected_iteration_values:
            if len(all_loaded_examples) >= n_samples: break
            files_in_iter = self.gcs_index.get(iter_val, [])
            if not files_in_iter: continue

            num_files_to_try_from_iter = min(2, len(files_in_iter)) # Load from up to 2 files per iter group
            if rng_key is None:
                chosen_file_indices_in_iter = np.random.choice(len(files_in_iter), size=num_files_to_try_from_iter, replace=False)
            else:
                rng_key, file_choice_key = jax.random.split(rng_key)
                chosen_file_indices_in_iter = jax.random.choice(file_choice_key, np.arange(len(files_in_iter)), shape=(num_files_to_try_from_iter,), replace=False)
            
            selected_files_paths = [files_in_iter[i] for i in chosen_file_indices_in_iter]
            logger.info(f"[P:{self.process_id}] _sample_from_gcs: For iteration {iter_val}, attempting to load from files: {selected_files_paths}")

            samples_needed_from_this_iter_group = min(target_samples_per_iteration_group, n_samples - len(all_loaded_examples))
            target_per_file = samples_needed_from_this_iter_group // num_files_to_try_from_iter + 1

            for file_path_to_load in selected_files_paths:
                if len(all_loaded_examples) >= n_samples: break
                try:
                    # Pass how many samples are roughly needed from this file
                    loaded_from_file = self._load_examples_from_gcs(file_path_to_load, target_per_file)
                    all_loaded_examples.extend(loaded_from_file)
                except Exception as e_load:
                    logger.warning(f"[P:{self.process_id}] _sample_from_gcs: Error loading from {file_path_to_load}: {e_load}", exc_info=True)
        
        if not all_loaded_examples:
            logger.error(f"[P:{self.process_id}] _sample_from_gcs: Failed to load any examples from GCS.")
            raise ValueError("No examples could be loaded from GCS")
            
        # Ensure exact n_samples by duplicating or truncating
        if len(all_loaded_examples) < n_samples:
            logger.warning(f"[P:{self.process_id}] _sample_from_gcs: Loaded {len(all_loaded_examples)} but needed {n_samples}. Duplicating.")
            indices_to_dup = np.random.choice(len(all_loaded_examples), size=n_samples - len(all_loaded_examples), replace=True)
            all_loaded_examples.extend([all_loaded_examples[i] for i in indices_to_dup])
        elif len(all_loaded_examples) > n_samples:
            all_loaded_examples = all_loaded_examples[:n_samples] # Simple truncation

        # Consolidate list of dicts to dict of numpy arrays
        result_batch = {k: np.array([ex[k] for ex in all_loaded_examples]) for k in all_loaded_examples[0].keys()}
        logger.info(f"[P:{self.process_id}] _sample_from_gcs: Finished. Loaded {len(all_loaded_examples)} examples. Total time: {time.time() - t_start_sample_gcs:.4f}s")
        return result_batch


    def _load_examples_from_gcs(self, file_path_on_gcs: str, max_examples_to_load: int):
        t_start_load_gcs_file = time.time()
        logger.info(f"[P:{self.process_id}] _load_examples_from_gcs: Attempting to load up to {max_examples_to_load} examples from GCS file {file_path_on_gcs}")
        
        gcs_blob_to_load = self.bucket.blob(file_path_on_gcs)
        
        # Ensure /tmp exists
        tmp_dir = "/tmp"
        if not os.path.exists(tmp_dir): os.makedirs(tmp_dir, exist_ok=True)
        temp_local_download_path = os.path.join(tmp_dir, os.path.basename(file_path_on_gcs) + f"_dl_{self.process_id}") # Add PID to avoid conflict if multiple processes on same host download same file
        
        examples_list = []
        try:
            t_download_start_gcs = time.time()
            gcs_blob_to_load.download_to_filename(temp_local_download_path)
            logger.info(f"[P:{self.process_id}] _load_examples_from_gcs: Downloaded {file_path_on_gcs} to {temp_local_download_path} in {time.time() - t_download_start_gcs:.4f}s")
            
            raw_tf_dataset = tf.data.TFRecordDataset(temp_local_download_path)
            
            loaded_count = 0
            for raw_example_proto in raw_tf_dataset:
                if loaded_count >= max_examples_to_load: break
                examples_list.append(self._parse_tfrecord(raw_example_proto))
                loaded_count +=1
        except Exception as e:
            logger.error(f"[P:{self.process_id}] _load_examples_from_gcs: Error processing file {file_path_on_gcs} (downloaded to {temp_local_download_path}): {e}", exc_info=True)
            # examples_list will remain as is, possibly empty
        finally:
            if os.path.exists(temp_local_download_path): os.remove(temp_local_download_path)
        
        logger.info(f"[P:{self.process_id}] _load_examples_from_gcs: Loaded {len(examples_list)} examples from {file_path_on_gcs}. Total _load_examples_from_gcs time: {time.time() - t_start_load_gcs_file:.4f}s")
        return examples_list
    
    def sample_with_recency_bias(self, batch_size, temperature=None, rng_key=None):
        # This method primarily just sets temperature and calls sample.
        # Logging within sample() should cover most needs.
        original_temp = self.recency_temperature
        if temperature is not None: self.recency_temperature = temperature
        
        result = self.sample(batch_size, rng_key) # This will now have its own detailed logs
        
        if temperature is not None: self.recency_temperature = original_temp # Restore
        return result
    
    def start_new_game(self):
        self.current_game_id += 1
        return self.current_game_id
    
    def get_stats(self):
        # Recalculate GCS persisted size for accurate "total_size" here
        gcs_persisted_size = sum(meta['size'] for meta in self.gcs_file_metadata.values())
        
        stats = {
            "total_gcs_persisted_size": gcs_persisted_size, # What's actually on GCS
            "local_cache_size": self.local_size,
            "combined_current_size": gcs_persisted_size + self.local_size, # GCS + current local cache
            "max_buffer_size_config": self.max_buffer_size,
            "fill_percentage_gcs": 100 * gcs_persisted_size / self.max_buffer_size if self.max_buffer_size > 0 else 0,
            "gcs_iterations_count": len(self.gcs_index),
            "gcs_files_count": sum(len(files) for files in self.gcs_index.values()),
        }
        stats.update(self.metrics)
        return stats
    
    def close(self):
        logger.info(f"[P:{self.process_id}] close: Closing GCSReplayBufferSync. Current local_size: {self.local_size}")
        if self.local_size > 0:
            logger.info(f"[P:{self.process_id}] close: Performing final flush of {self.local_size} local positions.")
            t_final_flush_start = time.time()
            positions_flushed = self.flush_to_gcs() 
            logger.info(f"[P:{self.process_id}] close: Final flush wrote {positions_flushed} positions, took {time.time() - t_final_flush_start:.4f}s")
        else:
            logger.info(f"[P:{self.process_id}] close: Local buffer already empty. No final flush needed.")
        
        # Log final stats from GCS after potential last flush
        final_gcs_size = sum(meta['size'] for meta in self.gcs_file_metadata.values())
        logger.info(f"[P:{self.process_id}] close: GCSReplayBufferSync closed. Final GCS persisted size: {final_gcs_size}")
    
    def __del__(self):

        try:
            if self.local_size > 0: # Attempt a last-ditch flush if not closed properly
                logger.warning(f"[P:{self.process_id}] __del__: Buffer not closed explicitly and local_size > 0. Attempting a final flush from __del__.")
                self.close()
        except Exception as e:
            logger.error(f"[P:{self.process_id}] __del__: Error during final cleanup/flush: {e}", exc_info=True)
            pass