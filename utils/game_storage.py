import time
import uuid
import json
import numpy as np
import jax
import threading
import queue
from typing import Dict, List, Any, Optional
from google.cloud import storage


def convert_game_to_sgf_like(game_data: Dict, env, game_id: Optional[str] = None, 
                           timestamp: Optional[int] = None, model_iteration: Optional[int] = None) -> Dict:
    """
    Convertit les données d'une partie générée en format optimisé pour analyse stratégique
    
    Args:
        game_data: Dictionnaire contenant les données de la partie générée
        env: L'environnement Abalone qui contient les informations sur les mouvements
        game_id: Identifiant unique de la partie (optionnel)
        timestamp: Horodatage de la partie (optionnel)
        model_iteration: Numéro d'itération du modèle qui a généré cette partie
        
    Returns:
        Dictionnaire formaté pour stockage et analyse future
    """
    # Extraire les informations de base
    moves_per_game = int(game_data['moves_per_game'])
    
    # Obtenir les données pour chaque mouvement
    boards = game_data['boards_2d'][:moves_per_game+1]
    players = game_data['actual_players'][:moves_per_game+1]
    black_outs = game_data['black_outs'][:moves_per_game+1]
    white_outs = game_data['white_outs'][:moves_per_game+1]
    
    # Déterminer le résultat final
    if game_data['final_black_out'] >= 6:
        result = "W+6"  # Victoire des blancs
    elif game_data['final_white_out'] >= 6:
        result = "B+6"  # Victoire des noirs
    else:
        result = "Draw"  # Match nul
    
    # Construire la séquence des mouvements
    moves_sequence = []
    for i in range(moves_per_game):
        # Trouver l'action exécutée
        action_idx = int(np.argmax(game_data['policies'][i]))
        
        # Convertir l'indice d'action en mouvement concret
        positions = env.moves_index['positions'][action_idx]
        direction_idx = env.moves_index['directions'][action_idx]
        group_size = env.moves_index['group_sizes'][action_idx]
        
        # Extraire seulement les positions valides pour ce groupe
        marble_positions = positions[:group_size].tolist()
        
        # Noms des directions pour lisibilité
        direction_names = ["NE", "E", "SE", "SW", "W", "NW"]
        direction_name = direction_names[direction_idx]
        
        move_info = {
            "move_num": i,
            "player": "B" if players[i] == 1 else "W",
            "marbles": marble_positions,
            "direction": direction_name,
            "black_out": int(black_outs[i]),
            "white_out": int(white_outs[i])
        }
        moves_sequence.append(move_info)
    
    # Construire le document final
    game_record = {
        "game_id": game_id if game_id is not None else f"game_{int(time.time())}",
        "timestamp": timestamp if timestamp is not None else int(time.time()),
        "model_iteration": model_iteration,
        "total_moves": moves_per_game,
        "result": result,
        "moves": moves_sequence,
        "boards": [board.tolist() for board in boards]
    }
    
    return game_record


def convert_games_batch(games_data: Dict, env, base_game_id: Optional[str] = None, 
                       timestamp: Optional[int] = None, model_iteration: Optional[int] = None) -> List[Dict]:
    """
    Convertit un batch de parties générées pour analyse stratégique
    
    Args:
        games_data: Données de parties provenant du générateur
        env: L'environnement Abalone
        base_game_id: Préfixe pour les IDs de parties (optionnel)
        timestamp: Horodatage commun (optionnel)
        model_iteration: Itération du modèle (optionnel)
        
    Returns:
        Liste de parties converties
    """
    if timestamp is None:
        timestamp = int(time.time())
    
    converted_games = []
    
    # Pour chaque dispositif
    for device_idx in range(len(games_data['moves_per_game'])):
        device_data = jax.tree_util.tree_map(
            lambda x: x[device_idx], 
            games_data
        )
        
        # Pour chaque partie générée sur ce dispositif
        games_per_device = len(device_data['moves_per_game'])
        for game_idx in range(games_per_device):
            game_length = int(device_data['moves_per_game'][game_idx])
            if game_length == 0:
                continue  # Ignorer les parties vides
                
            # Extraire les données spécifiques à cette partie
            game_specific_data = {
                'moves_per_game': game_length,
                'boards_2d': device_data['boards_2d'][game_idx],
                'policies': device_data['policies'][game_idx],
                'actual_players': device_data['actual_players'][game_idx],
                'black_outs': device_data['black_outs'][game_idx],
                'white_outs': device_data['white_outs'][game_idx],
                'final_black_out': device_data['final_black_out'][game_idx],
                'final_white_out': device_data['final_white_out'][game_idx]
            }
            
            # Générer un ID pour cette partie
            if base_game_id is not None:
                game_id = f"{base_game_id}_{device_idx}_{game_idx}"
            else:
                game_id = f"game_{timestamp}_{device_idx}_{game_idx}"
            
            # Convertir et ajouter à la liste
            converted_game = convert_game_to_sgf_like(
                game_specific_data, 
                env,
                game_id=game_id,
                timestamp=timestamp,
                model_iteration=model_iteration
            )
            converted_games.append(converted_game)
    
    return converted_games


class GameLogger:
    """
    Classe asynchrone pour le stockage des parties dans Google Cloud Storage
    """
    def __init__(self, bucket_name: str, buffer_size: int = 64, flush_interval: int = 300):
        """
        Initialise le logger de parties
        
        Args:
            bucket_name: Nom du bucket GCS
            buffer_size: Nombre de parties à accumuler avant de les envoyer vers GCS
            flush_interval: Intervalle entre les envois forcés (en secondes)
        """
        self.bucket_name = bucket_name
        self.buffer_size = buffer_size
        self.flush_interval = flush_interval
        self.game_queue = queue.Queue()
        self.buffer = []
        self.client = storage.Client()
        self.bucket = self.client.bucket(bucket_name)
        self.running = True
        self.thread = threading.Thread(target=self._worker_loop)
        self.thread.daemon = True
        self.thread.start()
    
    def log_game(self, game_data: Dict):
        """
        Ajoute une partie à la queue pour sauvegarde ultérieure
        
        Args:
            game_data: Dictionnaire contenant les données de la partie
        """
        self.game_queue.put(game_data)
    
    def log_games_batch(self, games_data: List[Dict]):
        """
        Ajoute un batch de parties à la queue
        
        Args:
            games_data: Liste de dictionnaires contenant les données des parties
        """
        for game in games_data:
            self.game_queue.put(game)
    
    def _worker_loop(self):
        """Boucle principale du thread travailleur"""
        last_flush_time = time.time()
        
        while self.running:
            # Essayer d'obtenir une partie de la queue sans bloquer
            try:
                game = self.game_queue.get(timeout=1.0)
                self.buffer.append(game)
                self.game_queue.task_done()
            except queue.Empty:
                pass
            
            # Vérifier si on doit flush le buffer
            time_since_flush = time.time() - last_flush_time
            if len(self.buffer) >= self.buffer_size or time_since_flush >= self.flush_interval:
                if self.buffer:
                    self._flush_buffer()
                    last_flush_time = time.time()
    
    def _flush_buffer(self):
        """Écrit toutes les parties du buffer vers GCS"""
        timestamp = int(time.time())
        batch_id = f"batch_{timestamp}"
        
        # Créer un objet blob pour cette batch
        blob = self.bucket.blob(f"games/{batch_id}.json")
        
        # Sauvegarder les parties en JSON
        blob.upload_from_string(
            json.dumps({
                "timestamp": timestamp,
                "games": self.buffer
            }),
            content_type="application/json"
        )
        
        print(f"Sauvegardé {len(self.buffer)} parties dans GCS: {batch_id}")
        self.buffer = []
    
    def stop(self):
        """Arrête proprement le thread et flush les données restantes"""
        self.running = False
        self.thread.join()
        if self.buffer:
            self._flush_buffer()


class LocalGameLogger:
    """
    Version locale du GameLogger pour le développement ou l'exécution sans GCS
    """
    def __init__(self, output_dir: str, buffer_size: int = 64, flush_interval: int = 300):
        """
        Initialise le logger de parties local
        
        Args:
            output_dir: Répertoire où stocker les fichiers de parties
            buffer_size: Nombre de parties à accumuler avant de les écrire
            flush_interval: Intervalle entre les écritures forcées (en secondes)
        """
        import os
        
        self.output_dir = output_dir
        self.buffer_size = buffer_size
        self.flush_interval = flush_interval
        
        # Créer le répertoire de sortie s'il n'existe pas
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialiser comme la version GCS
        self.game_queue = queue.Queue()
        self.buffer = []
        self.running = True
        self.thread = threading.Thread(target=self._worker_loop)
        self.thread.daemon = True
        self.thread.start()
    
    def log_game(self, game_data: Dict):
        """Ajoute une partie à la queue"""
        self.game_queue.put(game_data)
    
    def log_games_batch(self, games_data: List[Dict]):
        """Ajoute un batch de parties à la queue"""
        for game in games_data:
            self.game_queue.put(game)
    
    def _worker_loop(self):
        """Boucle principale du thread travailleur"""
        import os
        
        last_flush_time = time.time()
        
        while self.running:
            # Essayer d'obtenir une partie de la queue sans bloquer
            try:
                game = self.game_queue.get(timeout=1.0)
                self.buffer.append(game)
                self.game_queue.task_done()
            except queue.Empty:
                pass
            
            # Vérifier si on doit flush le buffer
            time_since_flush = time.time() - last_flush_time
            if len(self.buffer) >= self.buffer_size or time_since_flush >= self.flush_interval:
                if self.buffer:
                    self._flush_buffer()
                    last_flush_time = time.time()
    
    def _flush_buffer(self):
        """Écrit toutes les parties du buffer localement"""
        import os
        
        timestamp = int(time.time())
        batch_id = f"batch_{timestamp}"
        
        # Créer le chemin du fichier
        file_path = os.path.join(self.output_dir, f"{batch_id}.json")
        
        # Sauvegarder les parties en JSON
        with open(file_path, 'w') as f:
            json.dump({
                "timestamp": timestamp,
                "games": self.buffer
            }, f)
        
        print(f"Sauvegardé {len(self.buffer)} parties dans {file_path}")
        self.buffer = []
    
    def stop(self):
        """Arrête proprement le thread et flush les données restantes"""
        self.running = False
        self.thread.join()
        if self.buffer:
            self._flush_buffer()