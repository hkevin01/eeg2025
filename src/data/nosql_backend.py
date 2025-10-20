"""
MongoDB Backend for Experiment Tracking
========================================
Provides production-ready experiment tracking with MongoDB.
"""
from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.errors import ConnectionFailure, DuplicateKeyError
from datetime import datetime
from typing import Dict, List, Optional, Any
from bson import ObjectId
import os
import logging

logger = logging.getLogger(__name__)


class MongoExperimentTracker:
    """MongoDB-backed experiment tracking system."""
    
    def __init__(self, connection_string: str = None, database: str = 'eeg2025'):
        """
        Initialize MongoDB connection.
        
        Args:
            connection_string: MongoDB URI (default: reads from env MONGO_URI)
            database: Database name (default: 'eeg2025')
        """
        self.uri = connection_string or os.getenv(
            'MONGO_URI', 
            'mongodb://localhost:27017/'
        )
        
        try:
            self.client = MongoClient(
                self.uri,
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=10000
            )
            # Test connection
            self.client.admin.command('ping')
            logger.info(f"✅ Connected to MongoDB: {self.uri}")
        except ConnectionFailure as e:
            logger.error(f"❌ Failed to connect to MongoDB: {e}")
            raise
        
        self.db = self.client[database]
        
        # Collections
        self.experiments = self.db['experiments']
        self.epochs = self.db['epochs']
        self.checkpoints = self.db['checkpoints']
        self.subjects = self.db['subjects']
        
        self._ensure_indices()
    
    def _ensure_indices(self):
        """Create necessary indices for performance."""
        try:
            # Experiments
            self.experiments.create_index([('challenge', ASCENDING)])
            self.experiments.create_index([('status', ASCENDING)])
            self.experiments.create_index([('start_time', DESCENDING)])
            self.experiments.create_index([('tags', ASCENDING)])
            self.experiments.create_index([
                ('experiment_name', ASCENDING),
                ('start_time', DESCENDING)
            ])
            
            # Epochs - unique constraint on experiment + epoch
            self.epochs.create_index([
                ('experiment_id', ASCENDING),
                ('epoch', ASCENDING)
            ], unique=True)
            self.epochs.create_index([('timestamp', DESCENDING)])
            
            # Checkpoints
            self.checkpoints.create_index([('experiment_id', ASCENDING)])
            self.checkpoints.create_index([('is_best', ASCENDING)])
            self.checkpoints.create_index([
                ('experiment_id', ASCENDING),
                ('epoch', DESCENDING)
            ])
            
            # Subjects
            self.subjects.create_index([('releases', ASCENDING)])
            self.subjects.create_index([('age', ASCENDING)])
            
            logger.info("✅ MongoDB indices created/verified")
        except Exception as e:
            logger.warning(f"⚠️  Failed to create indices: {e}")
    
    def create_experiment(self, **kwargs) -> str:
        """
        Create new experiment and return ID.
        
        Args:
            **kwargs: Experiment metadata (experiment_name, challenge, model, config, etc.)
        
        Returns:
            str: Experiment ID
        """
        doc = {
            'start_time': datetime.utcnow(),
            'status': 'running',
            'created_by': os.getenv('USER', 'unknown'),
            **kwargs
        }
        
        try:
            result = self.experiments.insert_one(doc)
            exp_id = str(result.inserted_id)
            logger.info(f"📊 Created experiment: {exp_id}")
            return exp_id
        except Exception as e:
            logger.error(f"❌ Failed to create experiment: {e}")
            raise
    
    def log_epoch(self, experiment_id: str, epoch: int, metrics: Dict, 
                  timing: Dict = None):
        """
        Log epoch metrics.
        
        Args:
            experiment_id: Experiment ID
            epoch: Epoch number
            metrics: Dictionary of metrics (train_loss, val_loss, etc.)
            timing: Optional timing information
        """
        doc = {
            'experiment_id': ObjectId(experiment_id),
            'epoch': epoch,
            'metrics': metrics,
            'timestamp': datetime.utcnow()
        }
        
        if timing:
            doc['timing'] = timing
        
        try:
            self.epochs.insert_one(doc)
            logger.debug(f"📈 Logged epoch {epoch} for experiment {experiment_id}")
        except DuplicateKeyError:
            # Update existing epoch
            self.epochs.update_one(
                {'experiment_id': ObjectId(experiment_id), 'epoch': epoch},
                {'$set': doc}
            )
            logger.debug(f"📝 Updated epoch {epoch} for experiment {experiment_id}")
        except Exception as e:
            logger.error(f"❌ Failed to log epoch: {e}")
    
    def save_checkpoint_info(self, experiment_id: str, epoch: int, 
                            metrics: Dict, file: Dict, is_best: bool = False):
        """
        Register checkpoint in database.
        
        Args:
            experiment_id: Experiment ID
            epoch: Epoch number
            metrics: Checkpoint metrics
            file: File info (path, size_mb, checksum)
            is_best: Whether this is the best checkpoint
        """
        doc = {
            'experiment_id': ObjectId(experiment_id),
            'epoch': epoch,
            'metrics': metrics,
            'file': file,
            'is_best': is_best,
            'timestamp': datetime.utcnow()
        }
        
        try:
            # If marking as best, unmark previous best
            if is_best:
                self.checkpoints.update_many(
                    {'experiment_id': ObjectId(experiment_id), 'is_best': True},
                    {'$set': {'is_best': False}}
                )
            
            self.checkpoints.insert_one(doc)
            logger.info(f"💾 Saved checkpoint: epoch {epoch}, best={is_best}")
        except Exception as e:
            logger.error(f"❌ Failed to save checkpoint info: {e}")
    
    def update_experiment_status(self, experiment_id: str, status: str, **kwargs):
        """
        Update experiment status and metadata.
        
        Args:
            experiment_id: Experiment ID
            status: New status (running, completed, failed, stopped)
            **kwargs: Additional fields to update
        """
        update = {
            'status': status,
            'end_time': datetime.utcnow(),
            **kwargs
        }
        
        try:
            self.experiments.update_one(
                {'_id': ObjectId(experiment_id)},
                {'$set': update}
            )
            logger.info(f"✅ Updated experiment {experiment_id}: status={status}")
        except Exception as e:
            logger.error(f"❌ Failed to update experiment: {e}")
    
    def get_best_models(self, challenge: int, n: int = 10) -> List[Dict]:
        """
        Get top N models for a challenge by validation loss.
        
        Args:
            challenge: Challenge number (1 or 2)
            n: Number of top models to return
        
        Returns:
            List of experiment documents
        """
        pipeline = [
            {'$match': {'challenge': challenge, 'status': 'completed'}},
            {'$sort': {'metrics.best_val_loss': ASCENDING}},
            {'$limit': n},
            {'$project': {
                'experiment_name': 1,
                'model.name': 1,
                'metrics.best_val_loss': 1,
                'metrics.best_epoch': 1,
                'start_time': 1,
                'config': 1,
                'tags': 1
            }}
        ]
        
        try:
            results = list(self.experiments.aggregate(pipeline))
            logger.info(f"🏆 Retrieved top {len(results)} models for challenge {challenge}")
            return results
        except Exception as e:
            logger.error(f"❌ Failed to get best models: {e}")
            return []
    
    def search_experiments(self, query: Dict = None, 
                          projection: Dict = None,
                          sort: List = None,
                          limit: int = None) -> List[Dict]:
        """
        Flexible experiment search with filtering, projection, and sorting.
        
        Args:
            query: MongoDB query dict
            projection: Fields to include/exclude
            sort: List of (field, direction) tuples
            limit: Maximum number of results
        
        Returns:
            List of matching experiments
        """
        query = query or {}
        
        try:
            cursor = self.experiments.find(query, projection)
            
            if sort:
                cursor = cursor.sort(sort)
            if limit:
                cursor = cursor.limit(limit)
            
            results = list(cursor)
            logger.info(f"🔍 Found {len(results)} experiments matching query")
            return results
        except Exception as e:
            logger.error(f"❌ Search failed: {e}")
            return []
    
    def get_experiment_history(self, experiment_id: str) -> Dict:
        """
        Get complete training history for an experiment.
        
        Args:
            experiment_id: Experiment ID
        
        Returns:
            Dict with experiment, epochs, and checkpoints
        """
        try:
            exp = self.experiments.find_one({'_id': ObjectId(experiment_id)})
            if not exp:
                return None
            
            epochs = list(self.epochs.find(
                {'experiment_id': ObjectId(experiment_id)}
            ).sort('epoch', ASCENDING))
            
            checkpoints = list(self.checkpoints.find(
                {'experiment_id': ObjectId(experiment_id)}
            ).sort('epoch', ASCENDING))
            
            return {
                'experiment': exp,
                'epochs': epochs,
                'checkpoints': checkpoints
            }
        except Exception as e:
            logger.error(f"❌ Failed to get experiment history: {e}")
            return None
    
    def compare_experiments(self, experiment_ids: List[str]) -> List[Dict]:
        """
        Compare multiple experiments side-by-side.
        
        Args:
            experiment_ids: List of experiment IDs to compare
        
        Returns:
            List of experiment summaries
        """
        obj_ids = [ObjectId(eid) for eid in experiment_ids]
        
        try:
            experiments = list(self.experiments.find(
                {'_id': {'$in': obj_ids}},
                {
                    'experiment_name': 1,
                    'config': 1,
                    'metrics': 1,
                    'start_time': 1,
                    'end_time': 1,
                    'status': 1
                }
            ))
            
            logger.info(f"⚖️  Compared {len(experiments)} experiments")
            return experiments
        except Exception as e:
            logger.error(f"❌ Comparison failed: {e}")
            return []
    
    def close(self):
        """Close MongoDB connection."""
        if self.client:
            self.client.close()
            logger.info("👋 Closed MongoDB connection")


# Convenience function for quick access
def get_tracker() -> MongoExperimentTracker:
    """Get a MongoExperimentTracker instance with default settings."""
    return MongoExperimentTracker()
