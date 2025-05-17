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
                log_level: str = 'INFO'):
        """
        Initialise le buffer d'expérience synchrone basé sur GCS.
        
        Args:
            bucket_name: Nom du bucket GCS
            buffer_dir: Dossier dans le bucket pour stocker les données
            max_local_size: Taille maximale du cache local
            max_buffer_size: Taille maximale du buffer global (en nombre de positions)
            buffer_cleanup_threshold: Seuil de remplissage déclenchant le nettoyage (entre 0 et 1)
            board_size: Taille du plateau (par défaut: 9 pour Abalone 2D)
            action_space: Nombre d'actions possibles
            recency_enabled: Activer l'échantillonnage avec biais de récence
            recency_temperature: Température pour le biais de récence pour l'échantillonnage
            cleanup_temperature: Température pour l'échantillonnage lors du nettoyage
            log_level: Niveau de logging ('INFO', 'DEBUG', 'WARNING')
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
        
        # Configurer le niveau de log
        self.verbose = log_level == 'DEBUG'
        self.log_level = log_level
        
        # Identifiant de processus et d'hôte pour éviter les conflits
        self.process_id = jax.process_index()
        self.host_id = f"{os.uname().nodename}_{self.process_id}"
        
        # Cache local de données
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
        
        # Métadonnées sur le buffer
        self.local_size = 0
        self.position = 0
        self.current_game_id = 0
        self.total_size = 0  # Taille totale en comptant GCS
        
        # Initialiser le client GCS
        self.client = storage.Client()
        self.bucket = self.client.bucket(bucket_name)
        
        # Index des données disponibles sur GCS
        self.gcs_index = {}
        self.gcs_file_metadata = {}
        self.last_index_update = 0
        self.index_update_interval = 30  # Secondes avant de forcer une mise à jour de l'index
        
        # Initialiser l'index à partir de GCS
        self._update_gcs_index()
        
        # Statistiques
        self.metrics = {
            "samples_served": 0,
            "files_added": 0,
            "files_removed": 0,
            "cleanup_operations": 0
        }
        
        logger.info(f"GCSReplayBufferSync initialisé - Max buffer size: {self.max_buffer_size} positions")
    
    def add(self, board, marbles_out, policy, outcome, player, game_id=None, move_num=0, 
            iteration=0, model_version=0):
        """Ajoute une transition individuelle au buffer"""
        # Convertir en numpy si nécessaire
        if hasattr(board, 'device'):
            board = np.array(board)
        if hasattr(marbles_out, 'device'):
            marbles_out = np.array(marbles_out)
        if hasattr(policy, 'device'):
            policy = np.array(policy)
        
        # Si game_id n'est pas fourni, incrémenter le compteur interne
        if game_id is None:
            game_id = self.current_game_id
        
        # Stocker dans le cache local
        idx = self.position
        self.local_buffer['board'][idx] = board
        self.local_buffer['marbles_out'][idx] = marbles_out
        self.local_buffer['policy'][idx] = policy
        self.local_buffer['outcome'][idx] = outcome
        self.local_buffer['player'][idx] = player
        self.local_buffer['game_id'][idx] = game_id
        self.local_buffer['move_num'][idx] = move_num
        self.local_buffer['iteration'][idx] = iteration
        self.local_buffer['model_version'][idx] = model_version
        
        # Mettre à jour les compteurs
        self.position = (self.position + 1) % self.max_local_size
        self.local_size = min(self.local_size + 1, self.max_local_size)
        self.total_size += 1
    
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
    
    def flush_to_gcs(self):
        """Écrit synchroniquement le contenu du buffer local sur GCS."""
        if self.local_size == 0:
            return 0  
        
        logger.info(f"Début du flush vers GCS: {self.local_size} positions à écrire")
        # else:
            
        #     logger.info(f"Flush vers GCS: {self.local_size} positions")
        
        # Préparer les données du buffer local
        data_to_write = {}
        for key in self.local_buffer:
            data_to_write[key] = self.local_buffer[key][:self.local_size].copy()
        
        # Compteur pour les positions écrites
        total_written = 0
        files_created = 0
        
        # Générer un ID de batch unique
        timestamp = int(time.time())
        batch_id = f"{self.host_id}_{timestamp}"
        
        iterations = np.unique(data_to_write['iteration'])
        
        # Écrire les données pour chaque itération
        for iteration in iterations:
            # Filtrer les données pour cette itération
            iter_mask = data_to_write['iteration'] == iteration
            if not np.any(iter_mask):
                continue
            
            # Créer un sous-ensemble pour cette itération
            iter_data = {k: v[iter_mask] for k, v in data_to_write.items()}
            positions_in_iter = iter_data['board'].shape[0]
            
            # Créer le chemin dans le bucket
            iter_path = f"{self.buffer_dir}/iteration_{iteration}"
            file_path = f"{iter_path}/{batch_id}.tfrecord"
            
            
            logger.info(f"Écriture de {positions_in_iter} positions pour l'itération {iteration}")
            
            # Écrire en format TFRecord
            example_count = self._write_tfrecord(file_path, iter_data)
            total_written += example_count
            files_created += 1
            
            # Mettre à jour l'index local
            if iteration not in self.gcs_index:
                self.gcs_index[iteration] = []
            self.gcs_index[iteration].append(file_path)
            
            # Stocker les métadonnées du fichier
            self.gcs_file_metadata[file_path] = {
                'size': example_count,
                'timestamp': timestamp,
                'iteration': iteration
            }
            
            self.metrics["files_added"] += 1
        
        # Réinitialiser le buffer local après écriture
        self.local_size = 0
        self.position = 0
        
        # Mettre à jour la taille totale du buffer
        self._update_total_size()
        
        # Mettre à jour l'index après l'écriture
        self._update_gcs_index(force=False)  # Mise à jour légère
            
        # Vérifier si nettoyage nécessaire
        if self.total_size > self.max_buffer_size * self.buffer_cleanup_threshold:
            self._cleanup_buffer()
        
        if self.verbose:
            logger.info(f"Flush terminé: {total_written} positions dans {files_created} fichiers")
            
        return total_written
    
    def _write_tfrecord(self, file_path: str, data: Dict[str, np.ndarray]):
        """Écrit les données en format TFRecord sur GCS avec métadonnées de comptage"""
        temp_path = f"/tmp/{os.path.basename(file_path)}"
        
        example_count = len(data['board'])
        
        with tf.io.TFRecordWriter(temp_path) as writer:
            for i in range(example_count):
                # Créer un exemple TF avec les caractéristiques
                example = tf.train.Example(features=tf.train.Features(feature={
                    'board': tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[data['board'][i].tobytes()])),
                    'marbles_out': tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[data['marbles_out'][i].tobytes()])),
                    'policy': tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[data['policy'][i].tobytes()])),
                    'outcome': tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[data['outcome'][i]])),
                    'player': tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[data['player'][i]])),
                    'game_id': tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[data['game_id'][i]])),
                    'move_num': tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[data['move_num'][i]])),
                    'iteration': tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[data['iteration'][i]])),
                    'model_version': tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[data['model_version'][i]]))
                }))
                writer.write(example.SerializeToString())
        
        # Créer le dossier d'itération si nécessaire (pour GCS)
        iter_dir = os.path.dirname(file_path)
        try:
            # Vérifier si le dossier existe déjà
            check_blob = self.bucket.blob(f"{iter_dir}/.placeholder")
            if not check_blob.exists():
                # Créer un marqueur de dossier
                placeholder = self.bucket.blob(f"{iter_dir}/.placeholder")
                placeholder.upload_from_string("")
        except Exception as e:
            logger.warning(f"Impossible de vérifier/créer le dossier {iter_dir}: {e}")
        
        # Télécharger le fichier
        blob = self.bucket.blob(file_path)
        blob.metadata = {'example_count': str(example_count)}
        blob.upload_from_filename(temp_path)
        
        # Nettoyer
        os.remove(temp_path)
        
        return example_count
    
    def _update_gcs_index(self, force=False):
        """
        Met à jour l'index des fichiers disponibles sur GCS.
        
        Args:
            force: Si True, force une mise à jour complète même si récemment mise à jour
        
        Returns:
            bool: True si l'index a été mis à jour avec succès
        """
        current_time = time.time()
        
        # Vérifier si une mise à jour est nécessaire
        if not force and (current_time - self.last_index_update) < self.index_update_interval:
            return True  # Pas besoin de mise à jour
        
        try:
            # Lister tous les blobs dans le dossier buffer
            prefix = f"{self.buffer_dir}/"
            if self.verbose:
                logger.info(f"Mise à jour de l'index GCS pour {prefix}")
            
            blobs = list(self.bucket.list_blobs(prefix=prefix))
            
            if not blobs:
                # Vérifier si le dossier existe
                check_blob = self.bucket.blob(f"{prefix}.placeholder")
                if not check_blob.exists() and self.verbose:
                    logger.warning(f"Le dossier {prefix} n'existe peut-être pas")
                return False
            
            new_index = {}
            new_metadata = {}
            tfrecord_files_found = 0
            total_examples = 0
            
            for blob in blobs:
                path = blob.name
                if not path.endswith('.tfrecord'):
                    continue
                
                tfrecord_files_found += 1
                parts = path.split('/')
                if len(parts) >= 3 and parts[-2].startswith('iteration_'):
                    iteration = int(parts[-2].replace('iteration_', ''))
                    
                    if iteration not in new_index:
                        new_index[iteration] = []
                    
                    new_index[iteration].append(path)
                    
                    try:
                        # Format attendu: {host_id}_{timestamp}.tfrecord
                        file_basename = os.path.basename(path)
                        timestamp_part = file_basename.split('_')[-1].split('.')[0]
                        timestamp = int(timestamp_part)
                    except (IndexError, ValueError):
                        # Fallback si format non reconnu
                        timestamp = int(blob.time_created.timestamp()) if hasattr(blob, 'time_created') else 0
                    
                    # Récupérer le nombre d'exemples depuis les métadonnées
                    if hasattr(blob, 'metadata') and blob.metadata and 'example_count' in blob.metadata:
                        example_count = int(blob.metadata['example_count'])
                    else:
                        # Si pas de métadonnées, estimer (sera corrigé lors du chargement)
                        example_count = 1000
                    
                    total_examples += example_count
                    
                    new_metadata[path] = {
                        'size': example_count,
                        'timestamp': timestamp,
                        'iteration': iteration
                    }
            
            # Mettre à jour l'index uniquement s'il contient des données
            if tfrecord_files_found > 0:
                # Remplacer l'index et les métadonnées
                self.gcs_index = new_index
                self.gcs_file_metadata = new_metadata
                self.total_size = total_examples + self.local_size
                self.last_index_update = current_time
                
                if self.verbose:
                    iterations_found = list(new_index.keys())
                    logger.info(f"Index GCS: {tfrecord_files_found} fichiers, {len(iterations_found)} itérations, {total_examples} positions")
                
                return True
            elif self.verbose:
                logger.warning("Aucun fichier TFRecord trouvé dans le dossier buffer")
            
            return False
            
        except Exception as e:
            logger.error(f"Erreur lors de la mise à jour de l'index GCS: {e}")
            return False
    
    def _update_total_size(self):
        """Met à jour la taille totale du buffer en comptant les exemples dans les métadonnées"""
        total = 0
        
        # Compter à partir des métadonnées des fichiers
        for file_path, metadata in self.gcs_file_metadata.items():
            total += metadata['size']
        
        # Ajouter le buffer local
        total += self.local_size
        
        # Mettre à jour le total
        self.total_size = total
        
        return total
    
    def _cleanup_buffer(self):
        """
        Nettoie le buffer lorsqu'il dépasse sa taille maximale.
        Utilise une distribution de probabilité décroissante basée sur l'âge
        pour décider quels fichiers supprimer.
        """
        # Si le buffer est vide ou sous la limite, ne rien faire
        if self.total_size <= self.max_buffer_size:
            return
        
        # Calculer combien d'exemples doivent être supprimés
        overflow = self.total_size - int(self.max_buffer_size * 0.8)  # Viser 80% de remplissage
        if overflow <= 0:
            return
        
        self.metrics["cleanup_operations"] += 1
        logger.info(f"Nettoyage du buffer: besoin de supprimer {overflow}/{self.total_size} positions")
        
        # Collecter tous les fichiers avec leurs métadonnées
        all_files = []
        
        for iteration, files in self.gcs_index.items():
            for file_path in files:
                if file_path in self.gcs_file_metadata:
                    metadata = self.gcs_file_metadata[file_path]
                    all_files.append((file_path, metadata))
        
        # Sortir si pas de fichiers
        if not all_files:
            return
        
        # Trier par timestamp (du plus ancien au plus récent)
        all_files.sort(key=lambda x: x[1]['timestamp'])
        
        # Normaliser les âges (0 = le plus ancien, 1 = le plus récent)
        if len(all_files) > 1:
            oldest_time = all_files[0][1]['timestamp']
            newest_time = all_files[-1][1]['timestamp']
            time_range = max(1, newest_time - oldest_time)
            
            for i in range(len(all_files)):
                file_path, metadata = all_files[i]
                timestamp = metadata['timestamp']
                age_normalized = 1.0 - ((timestamp - oldest_time) / time_range)  # 1 = oldest, 0 = newest
                all_files[i] = (file_path, metadata, age_normalized)
        else:
            # Un seul fichier, lui donner un âge de 0.5
            file_path, metadata = all_files[0]
            all_files[0] = (file_path, metadata, 0.5)
        
        # Calculer les probabilités de suppression avec température
        probabilities = []
        for _, _, age in all_files:
            # Plus l'âge est grand (plus ancien), plus la probabilité est élevée
            prob = math.exp(age * self.cleanup_temperature)
            probabilities.append(prob)
        
        # Normaliser les probabilités
        total_prob = sum(probabilities)
        if total_prob > 0:
            probabilities = [p / total_prob for p in probabilities]
        else:
            # Fallback vers distribution uniforme
            probabilities = [1.0 / len(all_files)] * len(all_files)
        
        # Sélectionner des fichiers à supprimer jusqu'à atteindre la limite
        examples_removed = 0
        files_to_remove = []
        
        # Créer une copie pour l'échantillonnage sans remplacement
        remaining_files = list(range(len(all_files)))
        remaining_probs = probabilities.copy()
        
        while examples_removed < overflow and remaining_files:
            # Normaliser les probabilités restantes
            total_prob = sum(remaining_probs)
            if total_prob <= 0:
                break
            
            norm_probs = [p / total_prob for p in remaining_probs]
            
            # Sélectionner un fichier selon la distribution
            idx = np.random.choice(len(remaining_files), p=norm_probs)
            file_idx = remaining_files[idx]
            file_path, metadata, _ = all_files[file_idx]
            file_size = metadata['size']
            
            # Ajouter à la liste de suppression
            files_to_remove.append(file_path)
            examples_removed += file_size
            
            # Supprimer de la liste des candidats restants
            del remaining_files[idx]
            del remaining_probs[idx]
        
        # Supprimer les fichiers sélectionnés
        removed_count = 0
        for file_path in files_to_remove:
            try:
                blob = self.bucket.blob(file_path)
                blob.delete()
                
                # Mettre à jour l'index
                iteration = self.gcs_file_metadata[file_path]['iteration']
                if iteration in self.gcs_index and file_path in self.gcs_index[iteration]:
                    self.gcs_index[iteration].remove(file_path)
                    
                    # Si cette itération n'a plus de fichiers, la supprimer de l'index
                    if not self.gcs_index[iteration]:
                        del self.gcs_index[iteration]
                
                # Nettoyer les métadonnées
                if file_path in self.gcs_file_metadata:
                    del self.gcs_file_metadata[file_path]
                
                removed_count += 1
                self.metrics["files_removed"] += 1
                
            except Exception as e:
                logger.warning(f"Erreur lors de la suppression de {file_path}: {e}")
        
        # Mettre à jour la taille totale
        self._update_total_size()
        
        logger.info(f"Nettoyage terminé: {removed_count} fichiers supprimés, nouvelle taille: {self.total_size}")
    
    def _parse_tfrecord(self, example):
        """Parse un exemple TFRecord en dictionnaire numpy"""
        # Définir le schéma de fonctionnalités
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
        
        parsed = tf.io.parse_single_example(example, feature_description)
        
        return {
            'board': tf.io.decode_raw(parsed['board'], tf.int8).numpy().reshape(self.board_size, self.board_size),
            'marbles_out': tf.io.decode_raw(parsed['marbles_out'], tf.int8).numpy().reshape(2),
            'policy': tf.io.decode_raw(parsed['policy'], tf.float32).numpy().reshape(self.action_space),
            'outcome': parsed['outcome'].numpy(),
            'player': parsed['player'].numpy(),
            'game_id': parsed['game_id'].numpy(),
            'move_num': parsed['move_num'].numpy(),
            'iteration': parsed['iteration'].numpy(),
            'model_version': parsed['model_version'].numpy()
        }
    
    def sample(self, batch_size, rng_key=None):
        """
        Échantillonne un batch de transitions du buffer global sur GCS.
        
        Args:
            batch_size: Nombre d'exemples à échantillonner
            rng_key: Clé JAX pour la génération de nombres aléatoires
            
        Returns:
            Dict contenant les données échantillonnées
        """
        # Vérifier si l'index a besoin d'être mis à jour
        current_time = time.time()
        if current_time - self.last_index_update > self.index_update_interval:
            self._update_gcs_index()
        
        has_valid_data = bool(self.gcs_index)
        
        # Si pas de données GCS ou elles sont inaccessibles, utiliser le buffer local
        if not has_valid_data:
            if self.local_size == 0:
                raise ValueError("Buffer vide (aucune donnée locale ni sur GCS)")
            
            # Échantillonnage depuis le buffer local
            if rng_key is None:
                local_indices = np.random.randint(0, self.local_size, size=batch_size)
            else:
                local_indices = jax.random.randint(
                    rng_key, shape=(batch_size,), minval=0, maxval=self.local_size
                ).astype(np.int32)
                local_indices = np.array(local_indices)
            
            result = {}
            for k in self.local_buffer:
                result[k] = self.local_buffer[k][local_indices]
            
            self.metrics["samples_served"] += batch_size
            return result
        
        # Échantillonnage depuis GCS
        try:
            result = self._sample_from_gcs(batch_size, rng_key)
            self.metrics["samples_served"] += batch_size
            return result
        except Exception as e:
            logger.warning(f"Erreur lors de l'échantillonnage depuis GCS, fallback sur le buffer local: {e}")
            
            # Fallback sur le buffer local si disponible
            if self.local_size > 0:
                if rng_key is None:
                    local_indices = np.random.randint(0, self.local_size, size=batch_size)
                else:
                    local_indices = jax.random.randint(
                        rng_key, shape=(batch_size,), minval=0, maxval=self.local_size
                    ).astype(np.int32)
                    local_indices = np.array(local_indices)
                
                result = {}
                for k in self.local_buffer:
                    result[k] = self.local_buffer[k][local_indices]
                
                return result
            else:
                raise ValueError("Échec de l'échantillonnage GCS et buffer local vide")
            
    def _sample_from_gcs(self, n_samples, rng_key=None):
        """Échantillonne des exemples depuis GCS avec biais de récence."""
        # Construire une distribution pour l'échantillonnage des itérations
        iterations = sorted(list(self.gcs_index.keys()))
        if not iterations:
            return {}
        
        # Appliquer un biais de récence si activé
        if self.recency_enabled and len(iterations) > 1:
            # Normaliser les itérations entre 0 et 1
            min_iter = min(iterations)
            max_iter = max(iterations)
            range_iter = max(1, max_iter - min_iter)
            
            # Calculer les poids avec température
            weights = [(iter_num - min_iter) / range_iter for iter_num in iterations]
            weights = [np.exp(w * self.recency_temperature) for w in weights]
            total_weight = sum(weights)
            probs = [w / total_weight for w in weights]
        else:
            # Distribution uniforme
            probs = [1.0 / len(iterations)] * len(iterations)
        
        # Sélectionner les itérations
        if rng_key is None:
            selected_iters = np.random.choice(
                iterations, 
                size=min(3, len(iterations)), 
                p=probs, 
                replace=True
            )
        else:
            rng_key, subkey = jax.random.split(rng_key)
            selected_iters = jax.random.choice(
                subkey, 
                np.array(iterations), 
                shape=(min(3, len(iterations)),),
                p=np.array(probs),
                replace=True
            )
            selected_iters = np.array(selected_iters)
        
        # Collecter des exemples de chaque itération sélectionnée
        all_examples = []
        examples_per_iter = n_samples // len(selected_iters) + 1
        
        for iter_num in selected_iters:
            files = self.gcs_index[iter_num]
            if not files:
                continue
            
            # Sélectionner aléatoirement quelques fichiers pour diversité
            num_files_to_sample = min(2, len(files))
            if rng_key is None:
                file_indices = np.random.choice(len(files), size=num_files_to_sample, replace=False)
            else:
                rng_key, subkey = jax.random.split(rng_key)
                file_indices = jax.random.choice(
                    subkey, 
                    len(files), 
                    shape=(num_files_to_sample,), 
                    replace=False
                )
                file_indices = np.array(file_indices)
            
            # Répartir les exemples entre les fichiers sélectionnés
            examples_per_file = examples_per_iter // num_files_to_sample + 1
            
            for file_idx in file_indices:
                file_path = files[int(file_idx)]
                
                # Charger et échantillonner des exemples de ce fichier
                try:
                    examples = self._load_examples_from_gcs(file_path, examples_per_file)
                    all_examples.extend(examples)
                    
                    # Si nous avons assez d'exemples, arrêter l'échantillonnage
                    if len(all_examples) >= n_samples:
                        break
                except Exception as e:
                    logger.warning(f"Erreur lors du chargement des exemples de {file_path}: {e}")
            
            # Si nous avons assez d'exemples, arrêter l'échantillonnage
            if len(all_examples) >= n_samples:
                break
        
        # Gérer le cas où nous n'avons pas assez d'exemples
        if not all_examples:
            raise ValueError("Aucun exemple n'a pu être chargé depuis GCS")
            
        if len(all_examples) < n_samples:
            # Dupliquer des exemples existants pour atteindre la taille demandée
            if all_examples:  
                indices_to_duplicate = np.random.choice(
                    len(all_examples), size=n_samples-len(all_examples), replace=True)
                
                for idx in indices_to_duplicate:
                    all_examples.append(all_examples[idx])
        elif len(all_examples) > n_samples:
            # Tronquer si trop d'exemples
            all_examples = all_examples[:n_samples]
        
        # Consolider les exemples en un seul dict
        result = {}
        for k in all_examples[0].keys():
            result[k] = np.array([ex[k] for ex in all_examples])
        
        return result

    def _load_examples_from_gcs(self, file_path, max_examples):
        """Charge des exemples depuis un fichier TFRecord sur GCS"""
        blob = self.bucket.blob(file_path)
        temp_path = f"/tmp/{os.path.basename(file_path)}"
        blob.download_to_filename(temp_path)
        
        raw_dataset = tf.data.TFRecordDataset(temp_path)
        
        examples = []
        for i, raw_example in enumerate(raw_dataset):
            if i >= max_examples:
                break
            example = self._parse_tfrecord(raw_example)
            examples.append(example)
        
        os.remove(temp_path)
        
        return examples
    
    def sample_with_recency_bias(self, batch_size, temperature=None, rng_key=None):
        """
        Échantillonne avec biais de récence depuis GCS.
        
        Args:
            batch_size: Nombre d'exemples à échantillonner
            temperature: Température pour le biais de récence (None pour utiliser la valeur par défaut)
            rng_key: Clé JAX pour la génération de nombres aléatoires
            
        Returns:
            Dict contenant les données échantillonnées
        """
        original_temp = self.recency_temperature
        if temperature is not None:
            self.recency_temperature = temperature
        
        result = self.sample(batch_size, rng_key)
        
        if temperature is not None:
            self.recency_temperature = original_temp
        
        return result
    
    def start_new_game(self):
        """Incrémente l'ID de partie pour commencer une nouvelle partie"""
        self.current_game_id += 1
        return self.current_game_id
    
    def get_stats(self):
        """Renvoie des statistiques sur le buffer"""
        stats = {
            "total_size": self.total_size,
            "local_size": self.local_size,
            "max_size": self.max_buffer_size,
            "fill_percentage": 100 * self.total_size / self.max_buffer_size if self.max_buffer_size > 0 else 0,
            "iterations": len(self.gcs_index),
            "files": sum(len(files) for files in self.gcs_index.values()),
        }
        
        stats.update(self.metrics)
        
        return stats
    
    def close(self):
        """Ferme proprement le buffer et assure que toutes les données sont écrites"""
        if self.local_size > 0:
            positions_flushed = self.flush_to_gcs()
            logger.info(f"Flush final: {positions_flushed} positions écrites sur GCS")
        else :
            logger.info(f"Local empty")
        
        logger.info(f"Buffer GCS fermé. Total: {self.total_size} positions")
    
    def __del__(self):
        """Destructeur pour assurer la fermeture propre"""
        try:
            self.close()
        except:
            pass


# import math 
# logger = logging.getLogger("alphazero.buffer")
# # Assurez-vous que votre logger principal est configuré au niveau INFO
# # exemple: logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - PID %(process)d - %(levelname)s - %(message)s')


# class GCSReplayBufferSync:
#     def __init__(self,
#                  bucket_name: str,
#                  buffer_dir: str = 'buffer',
#                  max_local_size: int = 10000,
#                  max_buffer_size: int = 20_000_000,
#                  buffer_cleanup_threshold: float = 0.95,
#                  board_size: int = 9,
#                  action_space: int = 1734,
#                  recency_enabled: bool = True,
#                  recency_temperature: float = 0.8,
#                  cleanup_temperature: float = 2.0,
#                  log_level: str = 'INFO'): # Argument pour info, ne configure pas le logger ici

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
        
#         self.process_id = jax.process_index()
#         self.host_id = f"{os.uname().nodename}_{self.process_id}"
        
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
        
#         self.local_size = 0
#         self.position = 0
#         self.current_game_id = 0
#         self.total_size = 0 
        
#         self.client = storage.Client()
#         self.bucket = self.client.bucket(bucket_name)
        
#         self.gcs_index = {}
#         self.gcs_file_metadata = {}
#         self.last_index_update = 0
#         self.index_update_interval = 60  # Augmenté pour réduire la fréquence des list_blobs
        
#         logger.info(f"[P:{self.process_id}] GCSReplayBufferSync initializing...")
#         t_init_idx_start = time.time()
#         self._update_gcs_index(force=True)
#         logger.info(f"[P:{self.process_id}] GCSReplayBufferSync: Initial _update_gcs_index took {time.time() - t_init_idx_start:.2f}s. GCS Total: {self.total_size}")
        
#         self.metrics = {"samples_served": 0, "files_added": 0, "files_removed": 0, "cleanup_operations": 0}
#         logger.info(f"[P:{self.process_id}] GCSReplayBufferSync initialized. Max GCS: {self.max_buffer_size}, Max local cache: {self.max_local_size}")

#     def add(self, board, marbles_out, policy, outcome, player, game_id=None, move_num=0, iteration=0, model_version=0):
#         # Pas de logs ici, opération locale et rapide
#         if hasattr(board, 'device'): board = np.array(board)
#         if hasattr(marbles_out, 'device'): marbles_out = np.array(marbles_out)
#         if hasattr(policy, 'device'): policy = np.array(policy)
#         _game_id = game_id if game_id is not None else self.current_game_id
#         idx = self.position
#         self.local_buffer['board'][idx] = board
#         self.local_buffer['marbles_out'][idx] = marbles_out
#         self.local_buffer['policy'][idx] = policy
#         self.local_buffer['outcome'][idx] = outcome
#         self.local_buffer['player'][idx] = player
#         self.local_buffer['game_id'][idx] = _game_id
#         self.local_buffer['move_num'][idx] = move_num
#         self.local_buffer['iteration'][idx] = iteration
#         self.local_buffer['model_version'][idx] = model_version
#         self.position = (self.position + 1) % self.max_local_size
#         self.local_size = min(self.local_size + 1, self.max_local_size)

#     def add_batch(self, batch):
#         # Pas de logs ici non plus
#         batch_size_val = batch['board'].shape[0]
#         # ... (logique de add_batch comme avant) ...
#         has_game_id = 'game_id' in batch; has_move_num = 'move_num' in batch; has_iteration = 'iteration' in batch; has_model_version = 'model_version' in batch
#         for i in range(batch_size_val):
#             game_id = batch['game_id'][i] if has_game_id else None
#             move_num = batch['move_num'][i] if has_move_num else 0
#             iteration_val = batch['iteration'][i] if has_iteration else 0
#             model_version = batch['model_version'][i] if has_model_version else 0
#             self.add(batch['board'][i], batch['marbles_out'][i], batch['policy'][i], batch['outcome'][i], batch['player'][i], game_id=game_id, move_num=move_num, iteration=iteration_val, model_version=model_version)

#     def flush_to_gcs(self):
#         t_flush_start = time.time()
#         if self.local_size == 0: return 0

#         logger.info(f"[P:{self.process_id}] flush_to_gcs: START for {self.local_size} local positions.")
        
#         data_to_write = {k: self.local_buffer[k][:self.local_size].copy() for k in self.local_buffer}
#         current_local_size_flushing = self.local_size
#         total_written_this_flush = 0
#         files_created_this_flush = 0
#         flush_timestamp = int(time.time())
#         batch_id_suffix = f"{self.host_id}_{flush_timestamp}"
#         unique_iterations_in_local = np.unique(data_to_write['iteration'])
        
#         t_write_loop_start = time.time()
#         for iteration_val in unique_iterations_in_local:
#             iter_mask = data_to_write['iteration'] == iteration_val
#             if not np.any(iter_mask): continue
#             iter_data = {k: v[iter_mask] for k, v in data_to_write.items()}
#             file_path_on_gcs = f"{self.buffer_dir}/iteration_{iteration_val}/{batch_id_suffix}.tfrecord"
            
#             # Log _write_tfrecord une fois par appel, il a ses propres logs internes pour le temps d'upload
#             example_count = self._write_tfrecord(file_path_on_gcs, iter_data)

#             if example_count > 0:
#                 total_written_this_flush += example_count
#                 files_created_this_flush += 1
#                 if iteration_val not in self.gcs_index: self.gcs_index[iteration_val] = []
#                 self.gcs_index[iteration_val].append(file_path_on_gcs)
#                 self.gcs_file_metadata[file_path_on_gcs] = {'size': example_count, 'timestamp': flush_timestamp, 'iteration': iteration_val}
#                 self.metrics["files_added"] += 1
        
#         if files_created_this_flush > 0:
#              logger.info(f"[P:{self.process_id}] flush_to_gcs: TFRecord write loop created {files_created_this_flush} files, took {time.time() - t_write_loop_start:.2f}s.")

#         self.local_size = 0; self.position = 0
        
#         self._update_total_size() # Recalcule self.total_size à partir de gcs_file_metadata
        
#         t_update_idx_start = time.time()
#         self._update_gcs_index(force=False) 
#         logger.info(f"[P:{self.process_id}] flush_to_gcs: _update_gcs_index(force=False) took {time.time() - t_update_idx_start:.2f}s. GCS Total: {self.total_size}")
            
#         if self.total_size > self.max_buffer_size * self.buffer_cleanup_threshold:
#             logger.info(f"[P:{self.process_id}] flush_to_gcs: Cleanup needed. GCS Total ({self.total_size}) > threshold.")
#             t_cleanup_start = time.time()
#             self._cleanup_buffer()
#             logger.info(f"[P:{self.process_id}] flush_to_gcs: _cleanup_buffer took {time.time() - t_cleanup_start:.2f}s. GCS Total after cleanup: {self.total_size}")
        
#         logger.info(f"[P:{self.process_id}] flush_to_gcs: END. Flushed {total_written_this_flush} pos in {files_created_this_flush} files. Total GCSReplayBuffer.flush_to_gcs time: {time.time() - t_flush_start:.2f}s")
#         return total_written_this_flush

#     def _write_tfrecord(self, file_path_on_gcs: str, data: dict):
#         t_start_write_tf = time.time()
#         tmp_dir = "/tmp"; os.makedirs(tmp_dir, exist_ok=True)
#         temp_local_path = os.path.join(tmp_dir, os.path.basename(file_path_on_gcs))
#         example_count = len(data['board'])
#         if example_count == 0: return 0
            
#         try:
#             with tf.io.TFRecordWriter(temp_local_path) as writer: # Écriture locale
#                 for i in range(example_count):
#                     # ... (création de l'exemple TFRecord) ...
#                     example = tf.train.Example(features=tf.train.Features(feature={'board': tf.train.Feature(bytes_list=tf.train.BytesList(value=[data['board'][i].tobytes()])), 'marbles_out': tf.train.Feature(bytes_list=tf.train.BytesList(value=[data['marbles_out'][i].tobytes()])), 'policy': tf.train.Feature(bytes_list=tf.train.BytesList(value=[data['policy'][i].tobytes()])), 'outcome': tf.train.Feature(int64_list=tf.train.Int64List(value=[data['outcome'][i].item()])), 'player': tf.train.Feature(int64_list=tf.train.Int64List(value=[data['player'][i].item()])), 'game_id': tf.train.Feature(int64_list=tf.train.Int64List(value=[data['game_id'][i].item()])), 'move_num': tf.train.Feature(int64_list=tf.train.Int64List(value=[data['move_num'][i].item()])), 'iteration': tf.train.Feature(int64_list=tf.train.Int64List(value=[data['iteration'][i].item()])), 'model_version': tf.train.Feature(int64_list=tf.train.Int64List(value=[data['model_version'][i].item()]))}))
#                     writer.write(example.SerializeToString())
#         except Exception as e:
#             logger.error(f"[P:{self.process_id}] _write_tfrecord: Failed to write local {temp_local_path}: {e}", exc_info=True)
#             return 0

#         gcs_blob = self.bucket.blob(file_path_on_gcs)
#         gcs_blob.metadata = {'example_count': str(example_count)}
#         t_upload_start = time.time()
#         try:
#             gcs_blob.upload_from_filename(temp_local_path) # Upload GCS
#             logger.info(f"[P:{self.process_id}] _write_tfrecord: GCS upload for {file_path_on_gcs} ({example_count} pos) took {time.time() - t_upload_start:.2f}s.")
#         except Exception as e:
#             logger.error(f"[P:{self.process_id}] _write_tfrecord: GCS upload FAILED for {file_path_on_gcs}: {e}", exc_info=True)
#             if os.path.exists(temp_local_path): os.remove(temp_local_path)
#             return 0
#         if os.path.exists(temp_local_path): os.remove(temp_local_path)
#         # Removed overall time log for _write_tfrecord to reduce verbosity, upload time is key.
#         return example_count

#     def _update_gcs_index(self, force=False):
#         t_start_update_idx = time.time()
#         current_time = time.time()
        
#         if not force and (current_time - self.last_index_update) < self.index_update_interval:
#             # logger.info(f"[P:{self.process_id}] _update_gcs_index: Skipped (interval).") # Peut être trop fréquent
#             return True
        
#         logger.info(f"[P:{self.process_id}] _update_gcs_index: START (force={force}).")
#         updated_successfully = False
#         try:
#             prefix_to_list = f"{self.buffer_dir}/"
#             t_list_blobs_start = time.time()
#             blobs_list = list(self.bucket.list_blobs(prefix=prefix_to_list))
#             logger.info(f"[P:{self.process_id}] _update_gcs_index: list_blobs('{prefix_to_list}') found {len(blobs_list)} items, took {time.time() - t_list_blobs_start:.2f}s")
            
#             new_index_map, new_metadata_map, tfrecord_files_count, total_gcs_examples = {}, {}, 0, 0
#             # ... (logique de parsing des blobs comme avant, sans logs par blob) ...
#             for blob_item in blobs_list:
#                 path = blob_item.name;_ = blob_item # Linter fix
#                 if not path.endswith('.tfrecord'): continue
#                 tfrecord_files_count += 1
#                 try: iteration_str = next(p for p in reversed(path.split('/')) if p.startswith('iteration_')); iteration_val = int(iteration_str.replace('iteration_', ''))
#                 except: continue # Skip malformed paths
#                 if iteration_val not in new_index_map: new_index_map[iteration_val] = []
#                 new_index_map[iteration_val].append(path)
#                 try: filename_base = os.path.basename(path); timestamp_val = int(filename_base.split('_')[-1].split('.')[0])
#                 except: timestamp_val = int(blob_item.time_created.timestamp()) if blob_item.time_created else 0
#                 example_count = 0
#                 if hasattr(blob_item, 'metadata') and blob_item.metadata and 'example_count' in blob_item.metadata:
#                     try: example_count = int(blob_item.metadata['example_count'])
#                     except ValueError: example_count = 0 
#                 total_gcs_examples += example_count
#                 new_metadata_map[path] = {'size': example_count, 'timestamp': timestamp_val, 'iteration': iteration_val}

#             if tfrecord_files_count > 0 or not blobs_list : # Process if TFRecords found OR if GCS is truly empty
#                 self.gcs_index, self.gcs_file_metadata, self.total_size = new_index_map, new_metadata_map, total_gcs_examples
#                 logger.info(f"[P:{self.process_id}] _update_gcs_index: Index updated. {tfrecord_files_count} TFRecs, {len(new_index_map)} iters, {total_gcs_examples} GCS pos.")
#                 updated_successfully = True
#             else: # Blobs found, but no TFRecords
#                  logger.warning(f"[P:{self.process_id}] _update_gcs_index: No TFRecord files found among {len(blobs_list)} items. Index remains empty.")
#                  self.gcs_index, self.gcs_file_metadata, self.total_size = {}, {}, 0
#                  updated_successfully = True
#         except Exception as e:
#             logger.error(f"[P:{self.process_id}] _update_gcs_index: Error: {e}", exc_info=True)
        
#         self.last_index_update = current_time
#         logger.info(f"[P:{self.process_id}] _update_gcs_index: END (success={updated_successfully}). Total time: {time.time() - t_start_update_idx:.2f}s")
#         return updated_successfully

#     def _update_total_size(self):
#         self.total_size = sum(meta['size'] for meta in self.gcs_file_metadata.values())
#         return self.total_size

#     def _cleanup_buffer(self):
#         t_start_cleanup = time.time()
#         current_gcs_persisted_size = self._update_total_size() # Assure que self.total_size est à jour avec GCS
        
#         logger.info(f"[P:{self.process_id}] _cleanup_buffer: START. GCS size: {current_gcs_persisted_size}, Max: {self.max_buffer_size}.")
#         if current_gcs_persisted_size <= self.max_buffer_size * 0.8: return

#         self.metrics["cleanup_operations"] += 1
#         amount_to_remove = current_gcs_persisted_size - int(self.max_buffer_size * 0.8)
#         logger.info(f"[P:{self.process_id}] _cleanup_buffer: Need to remove ~{amount_to_remove} positions.")
        
#         all_files_meta = list(self.gcs_file_metadata.items())
#         if not all_files_meta: logger.warning(f"[P:{self.process_id}] _cleanup_buffer: No GCS files in metadata."); return
        
#         all_files_meta.sort(key=lambda x: x[1]['timestamp'])
#         # ... (logique de calcul des probabilités et sélection des fichiers à supprimer, comme avant, sans logs internes à la boucle) ...
#         files_with_age = []
#         if len(all_files_meta) > 1: oldest_time, newest_time = all_files_meta[0][1]['timestamp'], all_files_meta[-1][1]['timestamp']; time_span = float(max(1, newest_time - oldest_time)); norm_age_fn = lambda ts: 1.0 - ((ts - oldest_time) / time_span)
#         elif len(all_files_meta) == 1: norm_age_fn = lambda ts: 0.5
#         else: norm_age_fn = lambda ts: 0.0 # Should not happen if all_files_meta not empty
#         for path, meta in all_files_meta: files_with_age.append({'path': path, 'size': meta['size'], 'age': norm_age_fn(meta['timestamp'])})
#         probs = np.array([math.exp(f['age'] * self.cleanup_temperature) for f in files_with_age]); total_prob = np.sum(probs)
#         if total_prob <= 0: probs = np.ones(len(files_with_age)) / len(files_with_age) if len(files_with_age)>0 else np.array([])
#         else: probs /= total_prob
#         files_to_delete_paths, examples_marked_for_removal = [], 0
#         if len(probs)>0: # Ensure probs is not empty
#             sorted_candidates_indices = np.argsort([-p for p in probs]) 
#             for idx in sorted_candidates_indices:
#                 if examples_marked_for_removal >= amount_to_remove: break
#                 file_info = files_with_age[idx]; files_to_delete_paths.append(file_info['path']); examples_marked_for_removal += file_info['size']
        
#         logger.info(f"[P:{self.process_id}] _cleanup_buffer: Selected {len(files_to_delete_paths)} files ({examples_marked_for_removal} pos) for deletion.")

#         deleted_files_count, actually_removed_examples_count = 0, 0
#         t_delete_loop_start = time.time()
#         for file_path_to_delete in files_to_delete_paths:
#             try:
#                 file_metadata = self.gcs_file_metadata.get(file_path_to_delete)
#                 blob_to_delete = self.bucket.blob(file_path_to_delete); blob_to_delete.delete()
#                 if file_metadata:
#                     iteration_of_deleted = file_metadata['iteration']
#                     if iteration_of_deleted in self.gcs_index and file_path_to_delete in self.gcs_index[iteration_of_deleted]:
#                         self.gcs_index[iteration_of_deleted].remove(file_path_to_delete)
#                         if not self.gcs_index[iteration_of_deleted]: del self.gcs_index[iteration_of_deleted]
#                     del self.gcs_file_metadata[file_path_to_delete]
#                     actually_removed_examples_count += file_metadata['size']
#                 deleted_files_count += 1; self.metrics["files_removed"] += 1
#             except Exception as e: logger.warning(f"[P:{self.process_id}] _cleanup_buffer: Error deleting {file_path_to_delete}: {e}", exc_info=True)
        
#         if deleted_files_count > 0:
#             logger.info(f"[P:{self.process_id}] _cleanup_buffer: GCS file deletion loop took {time.time() - t_delete_loop_start:.2f}s for {deleted_files_count} files.")
        
#         self._update_total_size()
#         logger.info(f"[P:{self.process_id}] _cleanup_buffer: END. {deleted_files_count} files removed ({actually_removed_examples_count} pos). New GCS size: {self.total_size}. Total time: {time.time() - t_start_cleanup:.2f}s")

#     def _parse_tfrecord(self, example_proto):
#         # Pas de logs ici, appelé très fréquemment lors du sampling
#         # ... (logique de _parse_tfrecord comme avant) ...
#         feature_description = {'board': tf.io.FixedLenFeature([], tf.string),'marbles_out': tf.io.FixedLenFeature([], tf.string),'policy': tf.io.FixedLenFeature([], tf.string),'outcome': tf.io.FixedLenFeature([], tf.int64),'player': tf.io.FixedLenFeature([], tf.int64),'game_id': tf.io.FixedLenFeature([], tf.int64),'move_num': tf.io.FixedLenFeature([], tf.int64),'iteration': tf.io.FixedLenFeature([], tf.int64),'model_version': tf.io.FixedLenFeature([], tf.int64)}
#         parsed = tf.io.parse_single_example(example_proto, feature_description)
#         return {'board': tf.io.decode_raw(parsed['board'], tf.int8).numpy().reshape(self.board_size, self.board_size),'marbles_out': tf.io.decode_raw(parsed['marbles_out'], tf.int8).numpy().reshape(2),'policy': tf.io.decode_raw(parsed['policy'], tf.float32).numpy().reshape(self.action_space),'outcome': parsed['outcome'].numpy().astype(np.int8),'player': parsed['player'].numpy().astype(np.int8),'game_id': parsed['game_id'].numpy().astype(np.int32),'move_num': parsed['move_num'].numpy().astype(np.int16),'iteration': parsed['iteration'].numpy().astype(np.int32),'model_version': parsed['model_version'].numpy().astype(np.int32)}

#     def sample(self, batch_size, rng_key=None):
#         t_sample_start = time.time()
#         current_time_for_sample = time.time()
#         if current_time_for_sample - self.last_index_update > self.index_update_interval:
#             logger.info(f"[P:{self.process_id}] sample: Triggering _update_gcs_index (interval: {self.index_update_interval}s).")
#             t_idx_update_start_in_sample = time.time()
#             self._update_gcs_index() 
#             logger.info(f"[P:{self.process_id}] sample: _update_gcs_index call took {time.time() - t_idx_update_start_in_sample:.2f}s")
        
#         gcs_has_data = bool(self.gcs_index) and self.total_size > 0
#         sampled_data, source = None, "N/A"

#         if not gcs_has_data:
#             # ... (logique de fallback sur buffer local, comme avant) ...
#             logger.warning(f"[P:{self.process_id}] sample: No GCS data. Fallback to local (size: {self.local_size}).")
#             if self.local_size == 0: logger.error(f"[P:{self.process_id}] sample: Buffer COMPLETELY EMPTY."); raise ValueError("Buffer empty")
#             indices = np.random.randint(0, self.local_size, size=batch_size) if rng_key is None else np.array(jax.random.randint(rng_key, (batch_size,), 0, self.local_size))
#             sampled_data = {k: self.local_buffer[k][indices] for k in self.local_buffer}; source = "local_fallback_no_gcs"
#         else:
#             try:
#                 t_gcs_sample_start = time.time()
#                 sampled_data = self._sample_from_gcs(batch_size, rng_key)
#                 logger.info(f"[P:{self.process_id}] sample: _sample_from_gcs took {time.time() - t_gcs_sample_start:.2f}s")
#                 source = "GCS"
#             except Exception as e:
#                 logger.warning(f"[P:{self.process_id}] sample: Error sampling GCS: {e}. Fallback to local (size: {self.local_size}).", exc_info=True)
#                 if self.local_size > 0:
#                     indices = np.random.randint(0, self.local_size, size=batch_size) if rng_key is None else np.array(jax.random.randint(rng_key, (batch_size,), 0, self.local_size))
#                     sampled_data = {k: self.local_buffer[k][indices] for k in self.local_buffer}; source = "local_fallback_gcs_error"
#                 else: logger.error(f"[P:{self.process_id}] sample: GCS sample FAILED AND local empty."); raise ValueError("GCS sample fail, local empty")
        
#         if sampled_data: self.metrics["samples_served"] += len(sampled_data['board']) # Utiliser len(sampled_data['board']) pour la taille réelle du batch
#         logger.info(f"[P:{self.process_id}] sample: END. Got {len(sampled_data['board']) if sampled_data else 0} samples from {source}. Total sample() time: {time.time() - t_sample_start:.2f}s")
#         return sampled_data

#     def _sample_from_gcs(self, n_samples, rng_key=None):
#         # Les logs internes à cette fonction peuvent être nombreux si elle appelle _load_examples_from_gcs plusieurs fois.
#         # Pour l'instant, gardons les logs dans _load_examples_from_gcs.
#         # ... (logique de _sample_from_gcs comme avant, sans logs additionnels ici pour réduire la verbosité) ...
#         # ... (sauf peut-être un log au début et à la fin avec le temps total)
#         t_sfg_start = time.time()
#         iterations = sorted(list(self.gcs_index.keys()))
#         if not iterations: raise ValueError("No GCS iters")
#         if self.recency_enabled and len(iterations) > 1: min_iter, max_iter = min(iterations), max(iterations); iter_range = float(max(1, max_iter - min_iter)); weights = np.array([math.exp(((i - min_iter) / iter_range) * self.recency_temperature) for i in iterations]); probs = weights / np.sum(weights)
#         else: probs = np.ones(len(iterations)) / len(iterations)
#         num_iters_to_sample_from = min(3, len(iterations))
#         if rng_key is None: chosen_iteration_indices = np.random.choice(len(iterations), size=num_iters_to_sample_from, p=probs, replace=True); selected_iteration_values = [iterations[i] for i in chosen_iteration_indices]
#         else: rng_key, iter_choice_key = jax.random.split(rng_key); chosen_iteration_indices = jax.random.choice(iter_choice_key, np.arange(len(iterations)), shape=(num_iters_to_sample_from,), p=probs, replace=True); selected_iteration_values = [iterations[i] for i in chosen_iteration_indices] # Linter fix
#         all_loaded_examples = []
#         target_samples_per_iteration_group = n_samples // num_iters_to_sample_from + 1
#         for iter_val in selected_iteration_values:
#             if len(all_loaded_examples) >= n_samples: break
#             files_in_iter = self.gcs_index.get(iter_val, [])
#             if not files_in_iter: continue
#             num_files_to_try_from_iter = min(2, len(files_in_iter))
#             if rng_key is None: chosen_file_indices_in_iter = np.random.choice(len(files_in_iter), size=num_files_to_try_from_iter, replace=False)
#             else: rng_key, file_choice_key = jax.random.split(rng_key); chosen_file_indices_in_iter = jax.random.choice(file_choice_key, np.arange(len(files_in_iter)), shape=(num_files_to_try_from_iter,), replace=False) # Linter fix
#             selected_files_paths = [files_in_iter[i] for i in chosen_file_indices_in_iter]
#             samples_needed_from_this_iter_group = min(target_samples_per_iteration_group, n_samples - len(all_loaded_examples))
#             target_per_file = samples_needed_from_this_iter_group // num_files_to_try_from_iter + 1 if num_files_to_try_from_iter > 0 else samples_needed_from_this_iter_group
#             for file_path_to_load in selected_files_paths:
#                 if len(all_loaded_examples) >= n_samples: break
#                 try: loaded_from_file = self._load_examples_from_gcs(file_path_to_load, target_per_file); all_loaded_examples.extend(loaded_from_file)
#                 except Exception as e_load: logger.warning(f"[P:{self.process_id}] _sample_from_gcs: Error loading {file_path_to_load}: {e_load}", exc_info=False) # exc_info=False pour moins de bruit
#         if not all_loaded_examples: raise ValueError("No examples from GCS")
#         if len(all_loaded_examples) < n_samples: indices_to_dup = np.random.choice(len(all_loaded_examples), size=n_samples - len(all_loaded_examples), replace=True); all_loaded_examples.extend([all_loaded_examples[i] for i in indices_to_dup])
#         elif len(all_loaded_examples) > n_samples: all_loaded_examples = all_loaded_examples[:n_samples]
#         result_batch = {k: np.array([ex[k] for ex in all_loaded_examples]) for k in all_loaded_examples[0].keys()}
#         # logger.info(f"[P:{self.process_id}] _sample_from_gcs: END. Loaded {len(all_loaded_examples)} examples. Total time: {time.time() - t_sfg_start:.2f}s") # Peut être un peu trop
#         return result_batch

#     def _load_examples_from_gcs(self, file_path_on_gcs: str, max_examples_to_load: int):
#         t_start_load_gcs_file = time.time()
#         # Pas de log au début ici pour réduire verbosité, le log de download suffira
#         gcs_blob_to_load = self.bucket.blob(file_path_on_gcs)
#         tmp_dir = "/tmp"; os.makedirs(tmp_dir, exist_ok=True) # Linter fix
#         temp_local_download_path = os.path.join(tmp_dir, os.path.basename(file_path_on_gcs) + f"_dl_{self.process_id}_{int(t_start_load_gcs_file)}") # Unique temp file
#         examples_list = []
#         try:
#             t_download_start_gcs = time.time()
#             gcs_blob_to_load.download_to_filename(temp_local_download_path) # GCS Download
#             logger.info(f"[P:{self.process_id}] _load_examples_from_gcs: Downloaded {file_path_on_gcs} ({os.path.getsize(temp_local_download_path) if os.path.exists(temp_local_download_path) else 0} bytes) in {time.time() - t_download_start_gcs:.2f}s")
#             raw_tf_dataset = tf.data.TFRecordDataset(temp_local_download_path)
#             for loaded_count, raw_example_proto in enumerate(raw_tf_dataset): # Linter fix
#                 if loaded_count >= max_examples_to_load: break
#                 examples_list.append(self._parse_tfrecord(raw_example_proto))
#         except Exception as e: logger.error(f"[P:{self.process_id}] _load_examples_from_gcs: Error processing {file_path_on_gcs}: {e}", exc_info=False)
#         finally:
#             if os.path.exists(temp_local_download_path): os.remove(temp_local_download_path)
#         # logger.info(f"[P:{self.process_id}] _load_examples_from_gcs: Loaded {len(examples_list)} from {file_path_on_gcs}. Total time: {time.time() - t_start_load_gcs_file:.2f}s") # Peut être un peu trop
#         return examples_list
    
#     def sample_with_recency_bias(self, batch_size, temperature=None, rng_key=None):
#         original_temp = self.recency_temperature
#         if temperature is not None: self.recency_temperature = temperature
#         result = self.sample(batch_size, rng_key)
#         if temperature is not None: self.recency_temperature = original_temp
#         return result
    
#     def start_new_game(self):
#         self.current_game_id += 1
#         return self.current_game_id
    
#     def get_stats(self):
#         gcs_persisted_size = sum(meta['size'] for meta in self.gcs_file_metadata.values())
#         stats = {"total_gcs_persisted_size": gcs_persisted_size, "local_cache_size": self.local_size, "combined_current_size": gcs_persisted_size + self.local_size, "max_buffer_size_config": self.max_buffer_size, "fill_percentage": 100 * gcs_persisted_size / self.max_buffer_size if self.max_buffer_size > 0 else 0, "iterations": len(self.gcs_index), "files": sum(len(files) for files in self.gcs_index.values())}
#         stats.update(self.metrics); return stats
    
#     def close(self):
#         logger.info(f"[P:{self.process_id}] close: Closing GCSReplayBufferSync. Local cache size: {self.local_size}")
#         if self.local_size > 0:
#             logger.info(f"[P:{self.process_id}] close: Performing final flush of {self.local_size} local positions.")
#             t_final_flush_start = time.time()
#             self.flush_to_gcs() 
#             logger.info(f"[P:{self.process_id}] close: Final flush took {time.time() - t_final_flush_start:.2f}s")
#         final_gcs_size = sum(meta['size'] for meta in self.gcs_file_metadata.values())
#         logger.info(f"[P:{self.process_id}] close: GCSReplayBufferSync closed. Final GCS persisted size: {final_gcs_size}")
    
#     def __del__(self):
#         try:
#             if self.local_size > 0: self.close()
#         except: pass