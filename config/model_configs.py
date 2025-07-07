"""
model_configs.py
----------------
Configuration management for solar magnetic field prediction models.
Provides easy switching between different model configurations and training setups.

Key Features:
- Model-specific configurations
- Training hyperparameters
- Data pipeline settings
- Evaluation parameters
- Environment-specific settings
"""
import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
import os

@dataclass
class ModelConfig:
    """Base configuration class for models."""
    name: str
    model_type: str  # 'deeponet', 'fno', 'pinn', 'ensemble'
    description: str = ""
    
    # Model architecture
    latent_dim: int = 128
    width: int = 256
    depth: int = 6
    
    # Training parameters
    learning_rate: float = 1e-3
    batch_size: int = 8
    n_epochs: int = 1000
    early_stopping_patience: int = 50
    
    # Loss weights
    lambda_data: float = 1.0
    lambda_physics: float = 0.1
    lambda_divergence: float = 1.0
    lambda_curl: float = 0.1
    
    # Data parameters
    grid_size: tuple = (64, 64, 32)
    magnetogram_shape: tuple = (256, 256)
    
    # Optimization
    optimizer: str = 'adam'
    scheduler: str = 'cosine'
    weight_decay: float = 1e-6
    
    # Hardware
    use_mixed_precision: bool = True
    use_distributed: bool = False
    num_gpus: int = 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)
    
    def save(self, filepath: str):
        """Save config to file."""
        with open(filepath, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'ModelConfig':
        """Load config from file."""
        with open(filepath, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)

@dataclass
class DeepONetConfig(ModelConfig):
    """Configuration for DeepONet model."""
    model_type: str = 'deeponet'
    
    # DeepONet specific
    branch_depth: int = 8
    trunk_depth: int = 6
    latent_dim: int = 128
    width: int = 256
    
    # Training
    learning_rate: float = 1e-3
    batch_size: int = 4
    n_epochs: int = 1000
    
    # Loss weights
    lambda_data: float = 1.0
    lambda_physics: float = 0.1

@dataclass
class FNOConfig(ModelConfig):
    """Configuration for FNO model."""
    model_type: str = 'fno'
    
    # FNO specific
    modes: tuple = (16, 16, 8)  # (modes_x, modes_y, modes_z)
    width: int = 64
    depth: int = 4
    grid_size: tuple = (64, 64, 32)
    
    # Training
    learning_rate: float = 1e-3
    batch_size: int = 2
    n_epochs: int = 1000
    
    # Loss weights
    lambda_data: float = 1.0
    lambda_physics: float = 0.1
    lambda_divergence: float = 1.0
    lambda_curl: float = 0.1

@dataclass
class PINNConfig(ModelConfig):
    """Configuration for PINN model."""
    model_type: str = 'pinn'
    
    # PINN specific
    hidden_layers: tuple = (256, 256, 256, 256, 256)
    activation: str = 'tanh'
    
    # Training
    learning_rate: float = 1e-3
    batch_size: int = 8
    n_epochs: int = 1000
    
    # Loss weights
    lambda_data: float = 1.0
    lambda_physics: float = 1.0

@dataclass
class EnsembleConfig(ModelConfig):
    """Configuration for ensemble model."""
    model_type: str = 'ensemble'
    
    # Ensemble specific
    models: list = None  # List of model configs
    ensemble_method: str = 'average'  # 'average', 'weighted', 'voting'
    
    def __post_init__(self):
        if self.models is None:
            self.models = [
                DeepONetConfig(name="deeponet_1"),
                FNOConfig(name="fno_1"),
                PINNConfig(name="pinn_1")
            ]

@dataclass
class DataConfig:
    """Configuration for data pipeline."""
    
    # Data sources
    data_source: str = 'synthetic'  # 'synthetic', 'sdo', 'mixed'
    sdo_resolution: str = '720s'  # '720s', '45s', '12s'
    sdo_cadence: str = '12m'  # '12m', '6m', '2m'
    
    # Data processing
    grid_size: tuple = (64, 64, 32)
    magnetogram_shape: tuple = (256, 256)
    normalize_method: str = 'robust'  # 'robust', 'standard', 'minmax'
    remove_noise: bool = True
    noise_threshold: float = 50.0
    
    # Temporal data
    sequence_length: int = 10
    temporal_stride: int = 1
    forecast_horizon: int = 5
    
    # Augmentation
    use_augmentation: bool = True
    rotation_range: float = 10.0
    scale_range: tuple = (0.9, 1.1)
    noise_level: float = 0.05
    
    # Storage
    cache_dir: str = 'data/cache'
    processed_dir: str = 'data/processed'
    batch_size: int = 8

@dataclass
class TrainingConfig:
    """Configuration for training process."""
    
    # Basic training
    n_epochs: int = 1000
    batch_size: int = 8
    learning_rate: float = 1e-3
    optimizer: str = 'adam'  # 'adam', 'adamw', 'rmsprop'
    scheduler: str = 'cosine'  # 'constant', 'cosine', 'exponential'
    
    # Advanced training
    use_mixed_precision: bool = True
    use_distributed: bool = False
    gradient_clip: float = 1.0
    weight_decay: float = 1e-6
    
    # Hyperparameter optimization
    use_hyperopt: bool = False
    n_trials: int = 50
    early_stopping_patience: int = 50
    
    # Checkpointing
    save_checkpoints: bool = True
    checkpoint_frequency: int = 100
    keep_best_only: bool = True
    
    # Logging
    log_frequency: int = 10
    use_wandb: bool = False
    wandb_project: str = 'solar-magnetic-field'

@dataclass
class EvaluationConfig:
    """Configuration for evaluation process."""
    
    # Evaluation metrics
    metrics: list = None
    physics_metrics: bool = True
    field_line_metrics: bool = True
    temporal_metrics: bool = True
    
    # Comparison
    compare_models: bool = True
    baseline_models: list = None
    
    # Visualization
    save_plots: bool = True
    plot_format: str = 'png'  # 'png', 'pdf', 'svg'
    dpi: int = 300
    
    # Reporting
    generate_report: bool = True
    report_format: str = 'html'  # 'html', 'pdf', 'markdown'
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = ['mse', 'ssim', 'psnr', 'relative_l2_error']
        if self.baseline_models is None:
            self.baseline_models = ['low_lou', 'linear_interpolation']

@dataclass
class EnvironmentConfig:
    """Configuration for environment and hardware."""
    
    # Hardware
    device: str = 'auto'  # 'auto', 'cpu', 'gpu', 'tpu'
    num_gpus: int = 1
    memory_fraction: float = 0.9
    
    # Parallel processing
    num_workers: int = 4
    use_multiprocessing: bool = True
    
    # JAX settings
    jax_platform: str = 'gpu'  # 'cpu', 'gpu', 'tpu'
    jax_debug: bool = False
    jax_traceback_filtering: str = 'off'
    
    # Reproducibility
    seed: int = 42
    deterministic: bool = True

class ConfigManager:
    """Manager for handling multiple configurations."""
    
    def __init__(self, config_dir: str = 'config'):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        
        # Default configurations
        self.default_configs = {
            'deeponet': DeepONetConfig(),
            'fno': FNOConfig(),
            'pinn': PINNConfig(),
            'ensemble': EnsembleConfig(),
            'data': DataConfig(),
            'training': TrainingConfig(),
            'evaluation': EvaluationConfig(),
            'environment': EnvironmentConfig()
        }
    
    def get_config(self, config_type: str, config_name: str = 'default') -> Any:
        """
        Get configuration by type and name.
        
        Args:
            config_type: Type of configuration ('model', 'data', 'training', etc.)
            config_name: Name of specific configuration
            
        Returns:
            Configuration object
        """
        config_file = self.config_dir / f"{config_type}_{config_name}.yaml"
        
        if config_file.exists():
            # Load from file
            if config_type == 'deeponet':
                return DeepONetConfig.load(str(config_file))
            elif config_type == 'fno':
                return FNOConfig.load(str(config_file))
            elif config_type == 'pinn':
                return PINNConfig.load(str(config_file))
            elif config_type == 'ensemble':
                return EnsembleConfig.load(str(config_file))
            elif config_type == 'data':
                return DataConfig.load(str(config_file))
            elif config_type == 'training':
                return TrainingConfig.load(str(config_file))
            elif config_type == 'evaluation':
                return EvaluationConfig.load(str(config_file))
            elif config_type == 'environment':
                return EnvironmentConfig.load(str(config_file))
        else:
            # Return default
            return self.default_configs.get(config_type, ModelConfig())
    
    def save_config(self, config: Any, config_type: str, config_name: str = 'default'):
        """
        Save configuration to file.
        
        Args:
            config: Configuration object
            config_type: Type of configuration
            config_name: Name for the configuration
        """
        config_file = self.config_dir / f"{config_type}_{config_name}.yaml"
        config.save(str(config_file))
    
    def list_configs(self, config_type: str = None) -> list:
        """
        List available configurations.
        
        Args:
            config_type: Type of configuration to list (optional)
            
        Returns:
            List of configuration names
        """
        if config_type:
            pattern = f"{config_type}_*.yaml"
        else:
            pattern = "*.yaml"
        
        configs = list(self.config_dir.glob(pattern))
        return [c.stem for c in configs]
    
    def create_experiment_config(self, experiment_name: str, **kwargs) -> Dict[str, Any]:
        """
        Create a complete experiment configuration.
        
        Args:
            experiment_name: Name of the experiment
            **kwargs: Configuration overrides
            
        Returns:
            Complete experiment configuration
        """
        # Start with defaults
        experiment_config = {
            'experiment_name': experiment_name,
            'model': self.get_config('deeponet'),
            'data': self.get_config('data'),
            'training': self.get_config('training'),
            'evaluation': self.get_config('evaluation'),
            'environment': self.get_config('environment')
        }
        
        # Apply overrides
        for key, value in kwargs.items():
            if key in experiment_config:
                if isinstance(value, dict):
                    # Update nested config
                    for k, v in value.items():
                        setattr(experiment_config[key], k, v)
                else:
                    # Direct assignment
                    experiment_config[key] = value
        
        return experiment_config
    
    def save_experiment_config(self, experiment_config: Dict[str, Any], experiment_name: str):
        """
        Save complete experiment configuration.
        
        Args:
            experiment_config: Complete experiment configuration
            experiment_name: Name of the experiment
        """
        experiment_dir = self.config_dir / experiment_name
        experiment_dir.mkdir(exist_ok=True)
        
        # Save each component
        for config_type, config in experiment_config.items():
            if hasattr(config, 'save'):
                config.save(str(experiment_dir / f"{config_type}.yaml"))
        
        # Save complete config as JSON
        complete_config = {}
        for config_type, config in experiment_config.items():
            if hasattr(config, 'to_dict'):
                complete_config[config_type] = config.to_dict()
            else:
                complete_config[config_type] = config
        
        with open(experiment_dir / 'complete_config.json', 'w') as f:
            json.dump(complete_config, f, indent=2, default=str)

# Predefined configurations
PREDEFINED_CONFIGS = {
    'deeponet_fast': DeepONetConfig(
        name="deeponet_fast",
        latent_dim=64,
        width=128,
        branch_depth=6,
        trunk_depth=4,
        n_epochs=500,
        batch_size=8
    ),
    
    'deeponet_accurate': DeepONetConfig(
        name="deeponet_accurate",
        latent_dim=256,
        width=512,
        branch_depth=10,
        trunk_depth=8,
        n_epochs=2000,
        batch_size=4
    ),
    
    'fno_fast': FNOConfig(
        name="fno_fast",
        modes=(8, 8, 4),
        width=32,
        depth=3,
        grid_size=(32, 32, 16),
        n_epochs=500,
        batch_size=4
    ),
    
    'fno_accurate': FNOConfig(
        name="fno_accurate",
        modes=(32, 32, 16),
        width=128,
        depth=6,
        grid_size=(128, 128, 64),
        n_epochs=2000,
        batch_size=2
    ),
    
    'ensemble_basic': EnsembleConfig(
        name="ensemble_basic",
        models=[
            DeepONetConfig(name="deeponet_1", latent_dim=64, width=128),
            FNOConfig(name="fno_1", modes=(8, 8, 4), width=32),
            PINNConfig(name="pinn_1", hidden_layers=(128, 128, 128))
        ]
    )
}

def main():
    """Example usage of the configuration system."""
    
    # Create config manager
    config_manager = ConfigManager()
    
    # Save predefined configs
    for name, config in PREDEFINED_CONFIGS.items():
        config_manager.save_config(config, config.model_type, name)
    
    # Create experiment config
    experiment_config = config_manager.create_experiment_config(
        'solar_magnetic_field_experiment',
        model=PREDEFINED_CONFIGS['deeponet_accurate'],
        training=TrainingConfig(n_epochs=1500, use_wandb=True),
        environment=EnvironmentConfig(num_gpus=2)
    )
    
    # Save experiment config
    config_manager.save_experiment_config(experiment_config, 'solar_magnetic_field_experiment')
    
    print("Configuration system setup completed!")
    print(f"Available configs: {config_manager.list_configs()}")

if __name__ == "__main__":
    main() 