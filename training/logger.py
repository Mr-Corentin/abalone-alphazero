import os
import json
import datetime
import subprocess
from typing import Dict, Any, Optional

class GCSLogger:
    """Logger pour envoyer des logs d'entraînement sur Google Cloud Storage - Version simple"""
    
    def __init__(self, gcs_bucket: str, training_id: Optional[str] = None):
        """
        Initialize GCS Logger
        
        Args:
            gcs_bucket: Nom du bucket GCS (ex: "my-training-bucket")
            training_id: ID unique pour cette session (si None, auto-généré)
        """
        self.gcs_bucket = gcs_bucket
        self.training_id = training_id or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Paths GCS
        self.base_path = f"gs://{gcs_bucket}/logs_training/{self.training_id}"
        self.summary_path = f"{self.base_path}/training_summary.json"
        self.detailed_path = f"{self.base_path}/detailed_logs.jsonl"
        self.readable_path = f"{self.base_path}/training_log.txt"
        
        print(f"GCS Logger initialisé - Logs seront sauvegardés dans: {self.base_path}")
    
    def log_iteration_start(self, iteration: int, total_iterations: int):
        """Log le début d'une itération"""
        msg = f"=== Début itération {iteration+1}/{total_iterations} ==="
        self._write_log("iteration_start", {
            "iteration": iteration,
            "total_iterations": total_iterations,
            "message": msg
        }, f"\n{'='*80}\n{self._timestamp()} - {msg}\n{'='*80}")
    
    def log_generation_start(self, iteration: int, num_games: int):
        """Log le début de la génération de parties"""
        self._write_log("generation_start", {
            "iteration": iteration,
            "num_games": num_games,
            "message": f"Début génération de {num_games} parties"
        })
    
    def log_generation_end(self, iteration: int, duration: float, games_generated: int):
        """Log la fin de la génération de parties"""
        games_per_sec = games_generated / duration if duration > 0 else 0
        txt_msg = f"GÉNÉRATION: {games_generated} parties en {duration:.2f}s ({games_per_sec:.1f} parties/s)"
        
        self._write_log("generation_end", {
            "iteration": iteration,
            "duration_seconds": duration,
            "games_generated": games_generated,
            "games_per_second": games_per_sec,
            "message": txt_msg
        }, txt_msg)
    
    def log_training_start(self, iteration: int, num_steps: int):
        """Log le début de l'entraînement"""
        self._write_log("training_start", {
            "iteration": iteration,
            "num_steps": num_steps,
            "message": f"Début entraînement: {num_steps} étapes"
        })
    
    def log_training_end(self, iteration: int, duration: float, metrics: Dict[str, float]):
        """Log la fin de l'entraînement avec métriques"""
        txt_lines = [
            f"ENTRAÎNEMENT: {duration:.2f}s",
            f"    Loss total: {metrics.get('total_loss', 0):.4f}",
            f"    Loss politique: {metrics.get('policy_loss', 0):.4f}",
            f"    Loss valeur: {metrics.get('value_loss', 0):.4f}",
            f"    Précision politique: {metrics.get('policy_accuracy', 0):.2f}%"
        ]
        
        self._write_log("training_end", {
            "iteration": iteration,
            "duration_seconds": duration,
            "metrics": metrics,
            "message": f"Entraînement terminé en {duration:.2f}s"
        }, "\n".join(txt_lines))
    
    def log_evaluation_start(self, iteration: int, num_models: int):
        """Log le début de l'évaluation"""
        msg = f"Début évaluation contre {num_models} modèles précédents"
        self._write_log("evaluation_start", {
            "iteration": iteration,
            "num_models": num_models,
            "message": msg
        }, f"ÉVALUATION: {msg}")
    
    def log_evaluation_model(self, iteration: int, ref_iteration: int, 
                           wins: int, losses: int, draws: int, win_rate: float):
        """Log les résultats d'évaluation contre un modèle spécifique"""
        txt_msg = f"ÉVAL vs iter {ref_iteration}: {win_rate:.1%} (V:{wins} D:{losses} N:{draws})"
        
        self._write_log("evaluation_model", {
            "iteration": iteration,
            "ref_iteration": ref_iteration,
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "total_games": wins + losses + draws,
            "win_rate": win_rate,
            "message": txt_msg
        }, txt_msg)
    
    def log_evaluation_end(self, iteration: int, duration: float, 
                          global_win_rate: float, total_wins: int, total_games: int):
        """Log la fin de l'évaluation globale"""
        txt_msg = f"ÉVAL TERMINÉE: Win rate global {global_win_rate:.1%} ({total_wins}/{total_games}) en {duration:.2f}s"
        
        self._write_log("evaluation_end", {
            "iteration": iteration,
            "duration_seconds": duration,
            "global_win_rate": global_win_rate,
            "total_wins": total_wins,
            "total_games": total_games,
            "message": txt_msg
        }, txt_msg)
    
    def log_iteration_end(self, iteration: int, total_duration: float):
        """Log la fin d'une itération"""
        txt_msg = f"FIN ITÉRATION {iteration+1} - Durée totale: {total_duration:.2f}s"
        self._write_log("iteration_end", {
            "iteration": iteration,
            "total_duration_seconds": total_duration,
            "message": txt_msg
        }, f"{txt_msg}\n{'-'*80}")
    
    def log_learning_rate_update(self, iteration: int, old_lr: float, new_lr: float):
        """Log un changement de learning rate"""
        txt_msg = f"Learning Rate: {old_lr} → {new_lr}"
        self._write_log("lr_update", {
            "iteration": iteration,
            "old_lr": old_lr,
            "new_lr": new_lr,
            "message": txt_msg
        }, txt_msg)

    def log_shaping_factor_update(self, iteration: int, old_factor: float, new_factor: float):
        """Log un changement de facteur de reward shaping"""
        txt_msg = f"Reward Shaping Factor: {old_factor:.4f} → {new_factor:.4f}"
        self._write_log("shaping_factor_update", {
            "iteration": iteration,
            "old_factor": old_factor,
            "new_factor": new_factor,
            "message": txt_msg
        }, txt_msg)
    
    def log_checkpoint_save(self, iteration: int, checkpoint_type: str, path: str):
        """Log la sauvegarde d'un checkpoint"""
        txt_msg = f"CHECKPOINT {checkpoint_type.upper()} sauvegardé"
        self._write_log("checkpoint_save", {
            "iteration": iteration,
            "checkpoint_type": checkpoint_type,
            "path": path,
            "message": txt_msg
        }, txt_msg)
    
    def log_buffer_stats(self, iteration: int, buffer_stats: Dict[str, Any]):
        """Log les statistiques du buffer"""
        self._write_log("buffer_stats", {
            "iteration": iteration,
            "buffer_stats": buffer_stats,
            "message": f"Buffer: {buffer_stats.get('size', 'N/A')} positions"
        })
    
    def log_worker_timing(self, iteration: int, process_id: int, 
                         generation_time: float, training_time: float):
        """Log les timings spécifiques à un worker"""
        txt_msg = f"Worker {process_id}: Gen={generation_time:.2f}s, Train={training_time:.2f}s"
        self._write_log("worker_timing", {
            "iteration": iteration,
            "process_id": process_id,
            "generation_time": generation_time,
            "training_time": training_time,
            "message": txt_msg
        }, txt_msg)

    def log_worker_generation(self, iteration: int, process_id: int, duration: float, 
                            games_generated: int, positions_added: int):
        """Log les détails de génération pour un worker spécifique"""
        games_per_sec = games_generated / duration if duration > 0 else 0
        txt_msg = (f"Worker {process_id}: Génération terminée en {duration:.2f}s "
                f"({games_generated} parties, {games_per_sec:.1f} parties/s, "
                f"{positions_added} positions)")
        
        self._write_log("worker_generation", {
            "iteration": iteration,
            "process_id": process_id,
            "duration_seconds": duration,
            "games_generated": games_generated,
            "positions_added": positions_added,
            "games_per_second": games_per_sec,
            "message": txt_msg
        }, txt_msg)

    def log_worker_buffer_update(self, iteration: int, process_id: int, 
                                positions_added: int, buffer_stats: dict = None):
        """Log la mise à jour du buffer pour un worker spécifique"""
        txt_msg = f"Worker {process_id}: Buffer mis à jour avec +{positions_added} positions"
        
        log_data = {
            "iteration": iteration,
            "process_id": process_id,
            "positions_added": positions_added,
            "message": txt_msg
        }
        
        if buffer_stats:
            log_data["buffer_stats"] = buffer_stats
            if "total_size" in buffer_stats:
                txt_msg += f" (total: {buffer_stats['total_size']})"
            elif "size" in buffer_stats:
                txt_msg += f" (total: {buffer_stats['size']})"
        
        self._write_log("worker_buffer_update", log_data, txt_msg)

    def log_worker_training(self, iteration: int, process_id: int, duration: float, 
                        steps_completed: int, metrics: dict = None):
        """Log les détails d'entraînement pour un worker spécifique"""
        steps_per_sec = steps_completed / duration if duration > 0 else 0
        txt_msg = (f"Worker {process_id}: Entraînement terminé en {duration:.2f}s "
                f"({steps_completed} étapes, {steps_per_sec:.1f} étapes/s)")
        
        log_data = {
            "iteration": iteration,
            "process_id": process_id,
            "duration_seconds": duration,
            "steps_completed": steps_completed,
            "steps_per_second": steps_per_sec,
            "message": txt_msg
        }
        
        if metrics:
            log_data["metrics"] = metrics
            txt_msg += f" - Loss: {metrics.get('total_loss', 0):.4f}"
        
        self._write_log("worker_training", log_data, txt_msg)
    
    def log_custom(self, log_type: str, data: Dict[str, Any], message: str = ""):
        """Log personnalisé avec données arbitraires"""
        self._write_log(log_type, {
            "message": message,
            **data
        }, message)
    
    def _write_log(self, log_type: str, data: Dict[str, Any], txt_format: str = None):
        """Écrit immédiatement un log sur GCS"""
        timestamp = self._timestamp()
        
        # Préparer l'entrée JSON
        log_entry = {
            "timestamp": timestamp,
            "type": log_type,
            **data
        }
        
        # Format texte (utilise txt_format si fourni, sinon le message)
        if txt_format is None:
            txt_format = data.get('message', f"{log_type}: {data}")
        
        try:
            # Écrire JSONL
            self._append_to_gcs(
                json.dumps(log_entry) + '\n',
                self.detailed_path
            )
            
            # Écrire TXT
            self._append_to_gcs(
                f"{timestamp} - {txt_format}\n",
                self.readable_path
            )
            
        except Exception as e:
            print(f"Erreur lors de l'écriture du log: {e}")
    
    def _append_to_gcs(self, content: str, gcs_path: str):
        """Ajoute du contenu à un fichier GCS"""
        # Créer un fichier temporaire
        temp_file = f"/tmp/log_append_{int(datetime.datetime.now().timestamp() * 1000000)}.txt"
        
        with open(temp_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        try:
            if self._gcs_file_exists(gcs_path):
                # Fichier existe: composer avec l'existant
                cmd = f"gsutil cp {temp_file} {gcs_path}.tmp && "
                cmd += f"gsutil compose {gcs_path} {gcs_path}.tmp {gcs_path} && "
                cmd += f"gsutil rm {gcs_path}.tmp"
            else:
                # Fichier n'existe pas: simple copie
                cmd = f"gsutil cp {temp_file} {gcs_path}"
            
            subprocess.run(cmd, shell=True, check=True)
            
        finally:
            # Nettoyer le fichier temporaire
            if os.path.exists(temp_file):
                os.remove(temp_file)
    
    def create_summary(self, final_stats: Dict[str, Any]):
        """Crée un fichier de résumé final de l'entraînement"""
        summary = {
            "training_id": self.training_id,
            "created_at": self._timestamp(),
            "final_stats": final_stats,
            "gcs_paths": {
                "detailed_logs": self.detailed_path,
                "readable_log": self.readable_path,
                "summary": self.summary_path
            }
        }
        
        try:
            temp_file = f"/tmp/summary_{int(datetime.datetime.now().timestamp())}.json"
            with open(temp_file, 'w') as f:
                json.dump(summary, f, indent=2)
            
            subprocess.run(f"gsutil cp {temp_file} {self.summary_path}", shell=True, check=True)
            os.remove(temp_file)
            
            print(f"Résumé d'entraînement sauvegardé: {self.summary_path}")
            
        except Exception as e:
            print(f"Erreur lors de la création du résumé: {e}")
    
    def close(self):
        """Ferme le logger (pas grand chose à faire en mode immédiat)"""
        print(f"GCS Logger fermé - Logs disponibles dans: {self.base_path}")
    
    def _timestamp(self) -> str:
        """Génère un timestamp ISO formaté"""
        return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def _gcs_file_exists(self, gcs_path: str) -> bool:
        """Vérifie si un fichier existe sur GCS"""
        try:
            result = subprocess.run(
                f"gsutil -q stat {gcs_path}",
                shell=True,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            return result.returncode == 0
        except:
            return False