import numpy as np
import jax
import jax.numpy as jnp
import time
import os
import threading
import queue
import tensorflow as tf
from google.cloud import storage
from typing import Dict, List, Tuple, Any, Optional

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
    


class GCSReplayBuffer:
    """
    Buffer d'expérience utilisant Google Cloud Storage comme stockage principal.
    Permet le partage des expériences entre plusieurs nœuds TPU.
    """
    def __init__(self, 
                bucket_name: str,
                buffer_dir: str = 'buffer',
                max_local_size: int = 10000,
                board_size: int = 9,
                action_space: int = 1734,
                flush_interval: int = 60,
                flush_size: int = 1000,
                recency_enabled: bool = True,
                recency_temperature: float = 0.8,
                max_iterations_to_keep: int = 20):
        """
        Initialise le buffer d'expérience basé sur GCS.
        
        Args:
            bucket_name: Nom du bucket GCS
            buffer_dir: Dossier dans le bucket pour stocker les données
            max_local_size: Taille maximale du cache local
            board_size: Taille du plateau (par défaut: 9 pour Abalone 2D)
            action_space: Nombre d'actions possibles
            flush_interval: Intervalle en secondes pour écrire sur GCS
            flush_size: Nombre d'exemples avant écriture forcée
            recency_enabled: Activer l'échantillonnage avec biais de récence
            recency_temperature: Température pour le biais de récence
            max_iterations_to_keep: Nombre max d'itérations à conserver dans le buffer
        """
        self.bucket_name = bucket_name
        self.buffer_dir = buffer_dir
        self.max_local_size = max_local_size
        self.board_size = board_size
        self.action_space = action_space
        self.flush_interval = flush_interval
        self.flush_size = flush_size
        self.recency_enabled = recency_enabled
        self.recency_temperature = recency_temperature
        self.max_iterations_to_keep = max_iterations_to_keep
        
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
        self.last_flush_time = time.time()
        self.pending_writes = []
        
        # Initialiser le client GCS
        self.client = storage.Client()
        self.bucket = self.client.bucket(bucket_name)
        
        # Index des données disponibles sur GCS
        self.gcs_index = {}
        self.last_indexed_time = 0
        
        # File d'attente pour l'écriture en arrière-plan
        self.write_queue = queue.Queue()
        self.running = True
        
        # Démarrer le thread d'écriture
        self.writer_thread = threading.Thread(target=self._writer_loop)
        self.writer_thread.daemon = True
        self.writer_thread.start()
        
        # Démarrer le thread d'indexation
        self.index_thread = threading.Thread(target=self._indexer_loop)
        self.index_thread.daemon = True
        self.index_thread.start()
        
        # Initialiser l'index à partir de GCS
        self._update_gcs_index()
    
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
        
        # Vérifier si on doit écrire sur GCS
        if (len(self.pending_writes) >= self.flush_size or 
            time.time() - self.last_flush_time >= self.flush_interval):
            self._queue_writes()
    
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
    
    def _queue_writes(self):
        """Prépare les données du buffer local pour l'écriture sur GCS"""
        if self.local_size == 0:
            return
        
        # Créer une copie des données locales à écrire
        data_to_write = {}
        for key in self.local_buffer:
            if self.local_size < self.max_local_size:
                # Buffer pas encore plein, prendre seulement les données valides
                data_to_write[key] = self.local_buffer[key][:self.local_size].copy()
            else:
                # Buffer cyclique plein, réorganiser correctement
                data_to_write[key] = np.concatenate([
                    self.local_buffer[key][self.position:],
                    self.local_buffer[key][:self.position]
                ])
        
        # Ajouter à la file d'attente pour écriture
        timestamp = int(time.time())
        batch_id = f"{self.host_id}_{timestamp}"
        iterations = np.unique(data_to_write['iteration'])
        
        self.write_queue.put((batch_id, iterations, data_to_write))
        self.pending_writes.append(batch_id)
        self.last_flush_time = time.time()
    
    def _writer_loop(self):
        """Boucle principale du thread d'écriture"""
        while self.running:
            try:
                # Essayer d'obtenir un batch à écrire
                batch_id, iterations, data = self.write_queue.get(timeout=1.0)
                
                # Écrire les données sur GCS en format TFRecord
                for iteration in iterations:
                    # Filtrer les données pour cette itération
                    iter_mask = data['iteration'] == iteration
                    if not np.any(iter_mask):
                        continue
                    
                    # Créer un sous-ensemble pour cette itération
                    iter_data = {k: v[iter_mask] for k, v in data.items()}
                    
                    # Créer le chemin dans le bucket
                    iter_path = f"{self.buffer_dir}/iteration_{iteration}"
                    file_path = f"{iter_path}/{batch_id}.tfrecord"
                    
                    # Écrire en format TFRecord
                    self._write_tfrecord(file_path, iter_data)
                    
                    # Mettre à jour l'index local
                    if iteration not in self.gcs_index:
                        self.gcs_index[iteration] = []
                    self.gcs_index[iteration].append(file_path)
                
                # Marquer comme terminé
                if batch_id in self.pending_writes:
                    self.pending_writes.remove(batch_id)
                
                self.write_queue.task_done()
            
            except queue.Empty:
                # Pas de données à écrire pour le moment
                pass
            
            except Exception as e:
                print(f"Erreur dans _writer_loop: {e}")
    
    def _write_tfrecord(self, file_path: str, data: Dict[str, np.ndarray]):
        """Écrit les données en format TFRecord sur GCS"""
        # Créer un fichier TFRecord temporaire
        temp_path = f"/tmp/{os.path.basename(file_path)}"
        with tf.io.TFRecordWriter(temp_path) as writer:
            # Écrire chaque exemple
            for i in range(len(data['board'])):
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
        
        # Uploader sur GCS
        blob = self.bucket.blob(file_path)
        blob.upload_from_filename(temp_path)
        
        # Supprimer le fichier temporaire
        os.remove(temp_path)
    
    def _indexer_loop(self):
        """Boucle de mise à jour périodique de l'index GCS"""
        while self.running:
            # Mettre à jour l'index toutes les 2 minutes
            time.sleep(120)
            self._update_gcs_index()
            
            # Nettoyer les anciennes itérations si nécessaire
            self._cleanup_old_iterations()
    
    def _update_gcs_index(self):
        """Met à jour l'index des fichiers disponibles sur GCS"""
        try:
            # Lister tous les blobs dans le dossier buffer
            prefix = f"{self.buffer_dir}/"
            blobs = list(self.bucket.list_blobs(prefix=prefix))
            
            # Réinitialiser l'index
            new_index = {}
            
            # Traiter chaque blob
            for blob in blobs:
                # Extraire l'itération du chemin
                path = blob.name
                if not path.endswith('.tfrecord'):
                    continue
                
                # Format attendu: buffer_dir/iteration_X/batch_id.tfrecord
                parts = path.split('/')
                if len(parts) >= 3 and parts[-2].startswith('iteration_'):
                    iteration = int(parts[-2].replace('iteration_', ''))
                    
                    if iteration not in new_index:
                        new_index[iteration] = []
                    
                    new_index[iteration].append(path)
            
            # Mettre à jour l'index avec les nouvelles données
            self.gcs_index = new_index
            self.last_indexed_time = time.time()
            
            # Calculer le nombre total d'exemples (approximatif)
            self._estimate_total_size()
            
        except Exception as e:
            print(f"Erreur lors de la mise à jour de l'index GCS: {e}")
    
    def _estimate_total_size(self):
        """Estime la taille totale du buffer sur GCS"""
        # Cette estimation est approximative et pourrait être affinée
        # en lisant les métadonnées des fichiers TFRecord
        total_files = sum(len(files) for files in self.gcs_index.values())
        # Supposons une moyenne de 1000 exemples par fichier
        self.total_size = total_files * 1000 + self.local_size
    
    def _cleanup_old_iterations(self):
        """Supprime les données des itérations les plus anciennes"""
        if len(self.gcs_index) <= self.max_iterations_to_keep:
            return
        
        # Trier les itérations par ordre croissant
        iterations = sorted(self.gcs_index.keys())
        
        # Calculer combien d'itérations doivent être supprimées
        to_remove = len(iterations) - self.max_iterations_to_keep
        
        if to_remove <= 0:
            return
        
        # Supprimer les itérations les plus anciennes
        for i in range(to_remove):
            iter_to_remove = iterations[i]
            files_to_remove = self.gcs_index[iter_to_remove]
            
            # Supprimer chaque fichier
            for file_path in files_to_remove:
                try:
                    blob = self.bucket.blob(file_path)
                    blob.delete()
                except Exception as e:
                    print(f"Erreur lors de la suppression de {file_path}: {e}")
            
            # Supprimer de l'index
            del self.gcs_index[iter_to_remove]
            
        print(f"Nettoyage: suppression de {to_remove} anciennes itérations.")
    
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
        
        # Parser l'exemple
        parsed = tf.io.parse_single_example(example, feature_description)
        
        # Convertir en numpy avec les bonnes formes
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
        if not self.gcs_index:
            if self.local_size == 0:
                raise ValueError("Buffer vide, impossible d'échantillonner")
            
            #print("Avertissement: Aucune donnée disponible sur GCS, utilisation du cache local.")
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
        
        return self._sample_from_gcs(batch_size, rng_key)

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
                    print(f"Erreur lors du chargement des exemples de {file_path}: {e}")
            
            # Si nous avons assez d'exemples, arrêter l'échantillonnage
            if len(all_examples) >= n_samples:
                break
        
        # S'assurer d'avoir le nombre requis d'exemples
        if len(all_examples) < n_samples:
            # Pas assez d'exemples, dupliquer si nécessaire
            if all_examples:  # Vérifier que la liste n'est pas vide
                indices_to_duplicate = np.random.choice(
                    len(all_examples), size=n_samples-len(all_examples), replace=True)
                
                for idx in indices_to_duplicate:
                    all_examples.append(all_examples[idx])
        elif len(all_examples) > n_samples:
            # Trop d'exemples, réduire
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
        # Sauvegarder et restaurer la température originale si nécessaire
        original_temp = self.recency_temperature
        if temperature is not None:
            self.recency_temperature = temperature
        
        result = self.sample(batch_size, rng_key)
        
        # Restaurer la température originale
        if temperature is not None:
            self.recency_temperature = original_temp
        
        return result
    
    def _load_examples_from_gcs(self, file_path, max_examples):
        """Charge des exemples depuis un fichier TFRecord sur GCS"""
        # Télécharger dans un fichier temporaire
        blob = self.bucket.blob(file_path)
        temp_path = f"/tmp/{os.path.basename(file_path)}"
        blob.download_to_filename(temp_path)
        
        # Charger le dataset
        raw_dataset = tf.data.TFRecordDataset(temp_path)
        
        # Collecter des exemples
        examples = []
        for i, raw_example in enumerate(raw_dataset):
            if i >= max_examples:
                break
            example = self._parse_tfrecord(raw_example)
            examples.append(example)
        
        # Nettoyage
        os.remove(temp_path)
        
        return examples
    
    def start_new_game(self):
        """Incrémente l'ID de partie pour commencer une nouvelle partie"""
        self.current_game_id += 1
        return self.current_game_id
    
    def close(self):
        """Ferme proprement le buffer et assure que toutes les données sont écrites"""
        if self.local_size > 0:
            self._queue_writes()
        
        # Arrêter les threads
        self.running = False
        
        if self.writer_thread.is_alive():
            self.writer_thread.join()
        
        if self.index_thread.is_alive():
            self.index_thread.join()
        
        # Vider la file d'attente
        while not self.write_queue.empty():
            batch_id, iterations, data = self.write_queue.get()
            for iteration in iterations:
                iter_mask = data['iteration'] == iteration
                if not np.any(iter_mask):
                    continue
                
                iter_data = {k: v[iter_mask] for k, v in data.items()}
                iter_path = f"{self.buffer_dir}/iteration_{iteration}"
                file_path = f"{iter_path}/{batch_id}.tfrecord"
                self._write_tfrecord(file_path, iter_data)
            
            self.write_queue.task_done()
        
        print(f"GCSReplayBuffer fermé. Données locales sauvegardées.")
    
    def __del__(self):
        """Destructeur pour assurer la fermeture propre"""
        try:
            self.close()
        except:
            pass