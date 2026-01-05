"""Model service for managing ML models."""
import torch
from pathlib import Path
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class ModelService:
    """Service for loading and managing ML models."""
    
    def __init__(self, model_path: Path, device: str = "cuda"):
        self.model_path = model_path
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        self.models: Dict[str, torch.nn.Module] = {}
        logger.info(f"ModelService initialized with device: {self.device}")
    
    def load_model(self, model_name: str, model_class, checkpoint_path: Optional[str] = None):
        """Load a model from checkpoint."""
        try:
            if checkpoint_path:
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                model = model_class()
                model.load_state_dict(checkpoint)
                model.to(self.device)
                model.eval()
                self.models[model_name] = model
                logger.info(f"Loaded model {model_name} from {checkpoint_path}")
            else:
                logger.warning(f"No checkpoint provided for {model_name}")
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {str(e)}")
            raise
    
    def get_model(self, model_name: str) -> Optional[torch.nn.Module]:
        """Get a loaded model by name."""
        return self.models.get(model_name)
    
    def list_models(self) -> list[str]:
        """List all loaded models."""
        return list(self.models.keys())
