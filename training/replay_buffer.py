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
                max_buffer_size: int = 20_000_000,  # Nouveau: taille maximale du buffer (positions)
                buffer_cleanup_threshold: float = 0.95,  # Nouveau: seuil de nettoyage (% de remplissage)
                board_size: int = 9,
                action_space: int = 1734,
                recency_enabled: bool = True,
                recency_temperature: float = 0.8,
                cleanup_temperature: float = 2.0):  # Nouveau: température pour l'échantillonnage de nettoyage
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
        
        # Identifiant de processus et d'hôte pour éviter les conflits
        self.process_id = jax.process_index()
        self.host_id = f"{os.uname().nodename}_{self.process_id}"
        
        # Cache local de données pour la première itération ou secours
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
        self.gcs_file_metadata = {}  # Nouveau: stocke les métadonnées des fichiers (ex: taille, timestamp)
        
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
        if hasattr(board, 'device'):  # Détecte si c'est un tableau JAX
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
        """
        Écrit synchroniquement le contenu du buffer local sur GCS.
        Cette méthode est typiquement appelée à la fin d'une itération
        pour s'assurer que toutes les données sont stockées.
        """
        if self.local_size == 0:
            return 0  # Rien à écrire
        
        logger.info(f"Début du flush vers GCS: {self.local_size} positions à écrire")
        
        # Préparer les données du buffer local
        data_to_write = {}
        for key in self.local_buffer:
            data_to_write[key] = self.local_buffer[key][:self.local_size].copy()
        
        # Obtenir les itérations uniques
        iterations = np.unique(data_to_write['iteration'])
        logger.info(f"Données de {len(iterations)} itérations différentes: {iterations}")
        
        # Compteur pour les positions écrites
        total_written = 0
        files_created = 0
        
        # Générer un ID de batch unique
        timestamp = int(time.time())
        batch_id = f"{self.host_id}_{timestamp}"
        
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
            
            logger.info(f"Écriture de {positions_in_iter} positions pour l'itération {iteration} dans {file_path}")
            
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
        
        # Pause pour permettre à GCS de finaliser les opérations
        logger.info(f"Attente de 2 secondes pour que GCS finalise les opérations d'écriture...")
        time.sleep(2)
        
        # Forcer l'actualisation de l'index après l'écriture
        logger.info(f"Mise à jour de l'index GCS après l'écriture...")
        index_updated = self._update_gcs_index()
        
        if index_updated:
            logger.info(f"Index GCS mis à jour avec succès après le flush")
        else:
            logger.warning(f"Échec de la mise à jour de l'index GCS après le flush")
            
        # Vérifier des incohérences potentielles
        if len(self.gcs_index) == 0:
            logger.warning(f"ANOMALIE: {files_created} fichiers créés mais l'index GCS est vide!")
        
        # Vérifier si nettoyage nécessaire
        if self.total_size > self.max_buffer_size * self.buffer_cleanup_threshold:
            self._cleanup_buffer()
        
        logger.info(f"Flush terminé: {total_written} positions écrites dans {files_created} fichiers")
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
            # Continuer quand même, car l'écriture du fichier pourrait fonctionner
        
        # Télécharger le fichier
        blob = self.bucket.blob(file_path)
        blob.metadata = {'example_count': str(example_count)}
        blob.upload_from_filename(temp_path)
        
        # Nettoyer
        os.remove(temp_path)
        
        return example_count
    
    def _update_gcs_index(self):
        """Met à jour l'index des fichiers disponibles sur GCS et compte précisément les exemples"""
        try:
            # Lister tous les blobs dans le dossier buffer
            prefix = f"{self.buffer_dir}/"
            logger.info(f"Tentative de mise à jour de l'index GCS pour {prefix}")
            blobs = list(self.bucket.list_blobs(prefix=prefix))
            logger.info(f"Trouvé {len(blobs)} blobs dans {prefix}")
            
            if len(blobs) == 0:
                # Vérifier si le dossier existe
                check_blob = self.bucket.blob(f"{prefix}.placeholder")
                if not check_blob.exists():
                    logger.warning(f"Le dossier {prefix} n'existe peut-être pas dans le bucket {self.bucket_name}")
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
                    
                    # Extraire les informations de timestamp du nom de fichier
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
                        logger.debug(f"Fichier {path}: {example_count} exemples selon les métadonnées")
                    else:
                        # Si pas de métadonnées, estimer (sera corrigé lors du chargement)
                        example_count = 1000
                        logger.debug(f"Fichier {path}: pas de métadonnées, estimation de 1000 exemples")
                    
                    total_examples += example_count
                    
                    # Stocker les métadonnées
                    new_metadata[path] = {
                        'size': example_count,
                        'timestamp': timestamp,
                        'iteration': iteration
                    }
            
            # Journaliser des informations supplémentaires sur les fichiers trouvés
            if tfrecord_files_found > 0:
                iterations_found = list(new_index.keys())
                logger.info(f"Fichiers TFRecord trouvés: {tfrecord_files_found}, dans les itérations: {iterations_found}")
                for iter_num, files in new_index.items():
                    logger.info(f"  - Itération {iter_num}: {len(files)} fichiers")
            else:
                logger.warning("Aucun fichier TFRecord trouvé dans le dossier buffer")
                
            # Remplacer l'index et les métadonnées
            self.gcs_index = new_index
            self.gcs_file_metadata = new_metadata
            self.total_size = total_examples
            
            logger.info(f"Index GCS mis à jour: {len(new_metadata)} fichiers, {total_examples} positions")
            
            return len(new_metadata) > 0  # Renvoie True si des fichiers ont été trouvés
            
        except Exception as e:
            logger.error(f"Erreur lors de la mise à jour de l'index GCS: {e}")
            import traceback
            logger.error(traceback.format_exc())
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
        logger.info(f"Nettoyage du buffer: {self.total_size} positions, besoin de supprimer {overflow} positions")
        
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
        
        logger.info(f"Nettoyage terminé: {removed_count} fichiers supprimés, nouvelle taille: {self.total_size} positions")
    
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
        # Log de début pour tracer chaque appel à sample
        logger.info(f"Demande d'échantillonnage de {batch_size} exemples")
        
        has_valid_data = False
        
        # Journalisation détaillée de l'état de l'index GCS
        if not has_valid_data:
            logger.warning(f"Index GCS problématique: contient {len(self.gcs_index)} itérations mais aucune avec des fichiers valides")
            
            # Tenter une actualisation de l'index
            try:
                self._update_gcs_index()
                
                # Vérifier si l'actualisation a résolu le problème
                has_valid_data = False
                for iter_num, iter_files in self.gcs_index.items():
                    if iter_files:
                        has_valid_data = True
                        break
                        
                if not has_valid_data:
                    logger.warning("L'actualisation n'a pas résolu le problème, l'index reste vide ou invalide")
            except Exception as e:
                logger.error(f"Erreur lors de la tentative d'actualisation de l'index: {e}")
                import traceback
                logger.error(traceback.format_exc())
        
        # Si toujours pas de données valides, utiliser le cache local
        if not has_valid_data:
            if self.local_size == 0:
                # Journaliser plus d'informations pour le débogage
                logger.error(f"Buffer vide, impossible d'échantillonner. État du buffer: index={bool(self.gcs_index)}, local_size={self.local_size}, total_size={self.total_size}")
                raise ValueError("Buffer vide, impossible d'échantillonner (aucune donnée locale ni sur GCS)")
            
            if self.metrics["samples_served"] == 0:
                logger.info(f"Utilisation du cache local ({self.local_size} positions) en fallback")
            
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
            logger.info(f"Échantillonnage réussi depuis le cache local: {batch_size} exemples")
            return result
        
        try:
            result = self._sample_from_gcs(batch_size, rng_key)
            self.metrics["samples_served"] += batch_size
            return result
        except Exception as e:
            logger.error(f"Erreur lors de l'échantillonnage depuis GCS: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
    def _sample_from_gcs(self, n_samples, rng_key=None):
        """
        Échantillonne des exemples depuis GCS avec biais de récence.
        
        Args:
            n_samples: Nombre d'exemples à échantillonner
            rng_key: Clé JAX pour la génération de nombres aléatoires
            
        Returns:
            Dict contenant les données échantillonnées
        """
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
                size=min(5, len(iterations)), 
                p=probs, 
                replace=True
            )
        else:
            rng_key, subkey = jax.random.split(rng_key)
            selected_iters = jax.random.choice(
                subkey, 
                np.array(iterations), 
                shape=(min(5, len(iterations)),),
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
            num_files_to_sample = min(3, len(files))
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
        
        if len(all_examples) < n_samples:
            if all_examples:  
                indices_to_duplicate = np.random.choice(
                    len(all_examples), size=n_samples-len(all_examples), replace=True)
                
                for idx in indices_to_duplicate:
                    all_examples.append(all_examples[idx])
        elif len(all_examples) > n_samples:
            all_examples = all_examples[:n_samples]
        
        # Consolider les exemples en un seul dict
        if not all_examples:
            raise ValueError("Aucun exemple n'a pu être chargé depuis GCS")
        
        result = {}
        for k in all_examples[0].keys():
            result[k] = np.array([ex[k] for ex in all_examples])
        
        return result

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
        

        if file_path in self.gcs_file_metadata:

            pass
        
        return examples
    
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
        
        logger.info(f"GCSReplayBufferSync fermé. Total dans le buffer: {self.total_size} positions")
    
    def __del__(self):
        """Destructeur pour assurer la fermeture propre"""
        try:
            self.close()
        except:
            pass