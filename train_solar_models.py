#!/usr/bin/env python3
"""
train_solar_models.py
---------------------
Main training script for solar magnetic field prediction models.
Orchestrates the entire training pipeline using the configuration system.

Usage:
    python train_solar_models.py --config deeponet_accurate --experiment solar_experiment
    python train_solar_models.py --config fno_fast --data synthetic --epochs 500
    python train_solar_models.py --ensemble --models deeponet,fno,pinn

Features:
- Configuration-driven training
- Multiple model support (DeepONet, FNO, PINN, Ensemble)
- Hyperparameter optimization
- Distributed training
- Comprehensive evaluation
- Experiment tracking
"""
import argparse
import sys
import os
from pathlib import Path
import logging
from datetime import datetime
import json

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Import our components
from config.model_configs import ConfigManager, PREDEFINED_CONFIGS
from models.solar_deeponet_3d import SolarDeepONet, PhysicsInformedLoss, create_solar_deeponet_training_step
from models.solar_fno_3d import SolarFNO3D, PhysicsInformedFNOLoss, create_solar_fno_training_step
from training.advanced_training import AdvancedTrainer
from evaluation.comprehensive_evaluation import ComprehensiveEvaluator, generate_synthetic_test_data
from data.sdo_data_pipeline import SDOMagnetogramProcessor, SyntheticDataGenerator

import jax
import jax.numpy as jnp
import numpy as np
import optax

class SolarModelTrainer:
    """Main trainer class that orchestrates the entire training pipeline."""
    
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.logger = self._setup_logging()
        
        # Training components
        self.advanced_trainer = None
        self.evaluator = None
        self.data_processor = None
        
        # Results storage
        self.results = {}
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the main trainer."""
        logger = logging.getLogger('solar_model_trainer')
        logger.setLevel(logging.INFO)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        
        logger.addHandler(ch)
        return logger
    
    def setup_experiment(self, experiment_name: str, config_overrides: dict = None):
        """
        Setup experiment with configuration.
        
        Args:
            experiment_name: Name of the experiment
            config_overrides: Configuration overrides
        """
        self.logger.info(f"Setting up experiment: {experiment_name}")
        
        # Create experiment configuration
        self.experiment_config = self.config_manager.create_experiment_config(
            experiment_name, **(config_overrides or {})
        )
        
        # Setup components
        self._setup_components()
        
        # Create experiment directory
        self.experiment_dir = Path('experiments') / experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Save experiment configuration
        self.config_manager.save_experiment_config(
            self.experiment_config, experiment_name
        )
        
        self.logger.info(f"Experiment setup completed: {self.experiment_dir}")
    
    def _setup_components(self):
        """Setup training components based on configuration."""
        env_config = self.experiment_config['environment']
        
        # Set JAX configuration
        if env_config.jax_platform == 'gpu':
            jax.config.update('jax_platform_name', 'gpu')
        elif env_config.jax_platform == 'tpu':
            jax.config.update('jax_platform_name', 'tpu')
        
        # Set random seed
        jax.random.PRNGKey(env_config.seed)
        
        # Setup advanced trainer
        self.advanced_trainer = AdvancedTrainer(
            model_type=self.experiment_config['model'].model_type,
            experiment_name=self.experiment_config['experiment_name'],
            log_dir=str(self.experiment_dir / 'logs'),
            checkpoint_dir=str(self.experiment_dir / 'checkpoints'),
            n_trials=self.experiment_config['training'].n_trials,
            n_epochs=self.experiment_config['training'].n_epochs,
            early_stopping_patience=self.experiment_config['training'].early_stopping_patience
        )
        
        # Setup evaluator
        self.evaluator = ComprehensiveEvaluator(
            results_dir=str(self.experiment_dir / 'evaluation'),
            save_plots=self.experiment_config['evaluation'].save_plots
        )
        
        # Setup data processor
        data_config = self.experiment_config['data']
        self.data_processor = SDOMagnetogramProcessor(
            data_dir=data_config.processed_dir,
            cache_dir=data_config.cache_dir,
            resolution=data_config.sdo_resolution,
            cadence=data_config.sdo_cadence
        )
    
    def prepare_data(self, data_source: str = 'synthetic') -> dict:
        """
        Prepare training and evaluation data.
        
        Args:
            data_source: Data source ('synthetic', 'sdo', 'mixed')
            
        Returns:
            Dictionary with training and evaluation data
        """
        self.logger.info(f"Preparing data from source: {data_source}")
        
        data_config = self.experiment_config['data']
        
        if data_source == 'synthetic':
            # Generate synthetic data
            generator = SyntheticDataGenerator(grid_size=data_config.grid_size)
            
            # Training data
            train_data = generator.generate_low_lou_sequence(
                n_sequences=50,
                sequence_length=data_config.sequence_length
            )
            
            # Validation data
            val_data = generator.generate_low_lou_sequence(
                n_sequences=10,
                sequence_length=data_config.sequence_length
            )
            
            # Test data
            test_data = generate_synthetic_test_data(
                n_samples=20,
                grid_size=data_config.grid_size
            )
            
        elif data_source == 'sdo':
            # Use SDO data (placeholder - would need actual data)
            self.logger.warning("SDO data processing not fully implemented yet")
            # For now, fall back to synthetic
            return self.prepare_data('synthetic')
        
        elif data_source == 'mixed':
            # Combine synthetic and SDO data
            synthetic_data = self.prepare_data('synthetic')
            # Would add SDO data here
            return synthetic_data
        
        else:
            raise ValueError(f"Unknown data source: {data_source}")
        
        return {
            'train': train_data,
            'val': val_data,
            'test': test_data
        }
    
    def create_model(self, model_config) -> tuple:
        """
        Create model based on configuration.
        
        Args:
            model_config: Model configuration
            
        Returns:
            Tuple of (model, loss_function, training_step)
        """
        self.logger.info(f"Creating model: {model_config.name}")
        
        key = jax.random.PRNGKey(self.experiment_config['environment'].seed)
        
        if model_config.model_type == 'deeponet':
            model = SolarDeepONet(
                magnetogram_shape=model_config.magnetogram_shape,
                latent_dim=model_config.latent_dim,
                branch_depth=model_config.branch_depth,
                trunk_depth=model_config.trunk_depth,
                width=model_config.width,
                key=key
            )
            
            loss_fn = PhysicsInformedLoss(
                lambda_data=model_config.lambda_data,
                lambda_physics=model_config.lambda_physics
            )
            
            training_step = create_solar_deeponet_training_step
            
        elif model_config.model_type == 'fno':
            model = SolarFNO3D(
                input_channels=3,
                output_channels=3,
                modes=model_config.modes,
                width=model_config.width,
                depth=model_config.depth,
                grid_size=model_config.grid_size,
                key=key
            )
            
            loss_fn = PhysicsInformedFNOLoss(
                lambda_data=model_config.lambda_data,
                lambda_physics=model_config.lambda_physics,
                lambda_divergence=model_config.lambda_divergence,
                lambda_curl=model_config.lambda_curl
            )
            
            training_step = create_solar_fno_training_step
            
        else:
            raise ValueError(f"Unsupported model type: {model_config.model_type}")
        
        return model, loss_fn, training_step
    
    def train_single_model(self, model_config, data: dict) -> dict:
        """
        Train a single model.
        
        Args:
            model_config: Model configuration
            data: Training and evaluation data
            
        Returns:
            Training results
        """
        self.logger.info(f"Training model: {model_config.name}")
        
        # Create model
        model, loss_fn, training_step = self.create_model(model_config)
        
        # Prepare data for this model
        if model_config.model_type == 'deeponet':
            # Convert data format for DeepONet
            train_data = self._prepare_deeponet_data(data['train'])
            val_data = self._prepare_deeponet_data(data['val'])
        else:
            # Convert data format for FNO
            train_data = self._prepare_fno_data(data['train'])
            val_data = self._prepare_fno_data(data['val'])
        
        # Run hyperparameter optimization
        study = self.advanced_trainer.run_hyperparameter_optimization()
        
        # Train best model
        results = self.advanced_trainer.train_best_model(study)
        
        # Evaluate model
        test_data = self._prepare_test_data(data['test'], model_config.model_type)
        metrics = self.evaluator.evaluate_model(
            results['model'], model_config.name, test_data, test_data['ground_truth']
        )
        
        return {
            'model': results['model'],
            'study': results['study'],
            'metrics': metrics,
            'config': model_config
        }
    
    def _prepare_deeponet_data(self, data: dict) -> dict:
        """Prepare data for DeepONet training."""
        # DeepONet expects specific data format
        sequences = data['sequences']  # (N, T, 3, nx, ny)
        
        # Take first time step for training
        magnetogram = sequences[:, 0, :, :, :]  # (N, 3, nx, ny)
        
        # Generate 3D coordinates
        batch_size = magnetogram.shape[0]
        nx, ny = magnetogram.shape[2], magnetogram.shape[3]
        
        # Create coordinate grid
        x = np.linspace(-1, 1, nx)
        y = np.linspace(-1, 1, ny)
        z = np.linspace(0, 2, 32)  # Height dimension
        
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        coords = np.stack([X, Y, Z], axis=-1)  # (nx, ny, nz, 3)
        
        # Sample random points for training
        n_points = 1000
        coords_flat = coords.reshape(-1, 3)
        indices = np.random.choice(len(coords_flat), n_points, replace=False)
        coords_sampled = coords_flat[indices]
        
        # Repeat for batch
        coords_batch = np.tile(coords_sampled[None, :, :], (batch_size, 1, 1))
        
        # Generate synthetic ground truth
        B_true = np.random.normal(0, 1, (batch_size, n_points, 3))
        
        # Time and metadata
        time = np.random.uniform(0, 1, (batch_size,))
        metadata = np.random.normal(0, 1, (batch_size, 3))
        
        return {
            'magnetogram': magnetogram,
            'coords': coords_batch,
            'B_true': B_true,
            'time': time,
            'metadata': metadata
        }
    
    def _prepare_fno_data(self, data: dict) -> dict:
        """Prepare data for FNO training."""
        # FNO expects 3D grid data
        sequences = data['sequences']  # (N, T, 3, nx, ny)
        
        # Take first time step for training
        magnetogram = sequences[:, 0, :, :, :]  # (N, 3, nx, ny)
        
        # Create 3D coordinate grid
        batch_size = magnetogram.shape[0]
        nx, ny = magnetogram.shape[2], magnetogram.shape[3]
        nz = 32  # Height dimension
        
        x = np.linspace(-1, 1, nx)
        y = np.linspace(-1, 1, ny)
        z = np.linspace(0, 2, nz)
        
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        coords = np.stack([X, Y, Z], axis=-1)  # (nx, ny, nz, 3)
        coords = np.tile(coords[None, :, :, :, :], (batch_size, 1, 1, 1, 1))
        
        # Generate synthetic 3D ground truth
        B_true = np.random.normal(0, 1, (batch_size, nx, ny, nz, 3))
        
        # Time
        time = np.random.uniform(0, 1, (batch_size,))
        
        return {
            'magnetogram': magnetogram,
            'coords': coords,
            'B_true': B_true,
            'time': time
        }
    
    def _prepare_test_data(self, data: dict, model_type: str) -> dict:
        """Prepare test data for evaluation."""
        if model_type == 'deeponet':
            return self._prepare_deeponet_data(data)
        else:
            return self._prepare_fno_data(data)
    
    def train_ensemble(self, model_configs: list, data: dict) -> dict:
        """
        Train ensemble of models.
        
        Args:
            model_configs: List of model configurations
            data: Training and evaluation data
            
        Returns:
            Ensemble training results
        """
        self.logger.info(f"Training ensemble of {len(model_configs)} models")
        
        ensemble_results = {}
        
        for model_config in model_configs:
            try:
                result = self.train_single_model(model_config, data)
                ensemble_results[model_config.name] = result
                self.logger.info(f"Completed training for {model_config.name}")
            except Exception as e:
                self.logger.error(f"Failed to train {model_config.name}: {e}")
                continue
        
        # Evaluate ensemble
        ensemble_metrics = self._evaluate_ensemble(ensemble_results, data)
        
        return {
            'models': ensemble_results,
            'ensemble_metrics': ensemble_metrics
        }
    
    def _evaluate_ensemble(self, ensemble_results: dict, data: dict) -> dict:
        """Evaluate ensemble performance."""
        # Simple ensemble evaluation - average predictions
        test_data = self._prepare_test_data(data['test'], 'deeponet')  # Use DeepONet format
        
        # Get predictions from all models
        predictions = {}
        for model_name, result in ensemble_results.items():
            model = result['model']
            pred = model(model.parameters(), 
                        test_data['magnetogram'],
                        test_data['coords'],
                        test_data.get('time'),
                        test_data.get('metadata'))
            predictions[model_name] = pred
        
        # Average predictions
        avg_prediction = np.mean(list(predictions.values()), axis=0)
        
        # Evaluate ensemble
        ensemble_metrics = self.evaluator.evaluate_model(
            None, 'ensemble', test_data, test_data['ground_truth']
        )
        
        return ensemble_metrics
    
    def run_experiment(self, experiment_name: str, config_overrides: dict = None, 
                      data_source: str = 'synthetic', use_ensemble: bool = False) -> dict:
        """
        Run complete experiment.
        
        Args:
            experiment_name: Name of the experiment
            config_overrides: Configuration overrides
            data_source: Data source for training
            use_ensemble: Whether to train ensemble
            
        Returns:
            Experiment results
        """
        self.logger.info(f"Starting experiment: {experiment_name}")
        
        # Setup experiment
        self.setup_experiment(experiment_name, config_overrides)
        
        # Prepare data
        data = self.prepare_data(data_source)
        
        # Train models
        if use_ensemble:
            model_configs = self.experiment_config['model'].models
            results = self.train_ensemble(model_configs, data)
        else:
            model_config = self.experiment_config['model']
            results = self.train_single_model(model_config, data)
        
        # Compare models
        if use_ensemble:
            model_names = list(results['models'].keys())
        else:
            model_names = [self.experiment_config['model'].name]
        
        comparison = self.evaluator.compare_models(model_names)
        
        # Generate report
        report_path = self.evaluator.create_evaluation_report()
        
        # Save results
        self._save_experiment_results(results, comparison, report_path)
        
        self.logger.info(f"Experiment completed: {experiment_name}")
        
        return {
            'results': results,
            'comparison': comparison,
            'report_path': report_path
        }
    
    def _save_experiment_results(self, results: dict, comparison: dict, report_path: str):
        """Save experiment results."""
        # Save results to JSON
        results_file = self.experiment_dir / 'results.json'
        
        # Convert results to serializable format
        serializable_results = {}
        for key, value in results.items():
            if hasattr(value, 'to_dict'):
                serializable_results[key] = value.to_dict()
            else:
                serializable_results[key] = str(value)
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        
        # Save comparison
        if comparison is not None:
            comparison_file = self.experiment_dir / 'comparison.csv'
            comparison.to_csv(comparison_file, index=False)
        
        self.logger.info(f"Results saved to {self.experiment_dir}")

def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='Train solar magnetic field prediction models')
    
    # Experiment configuration
    parser.add_argument('--experiment', type=str, default='solar_experiment',
                       help='Experiment name')
    parser.add_argument('--config', type=str, default='deeponet_accurate',
                       help='Model configuration to use')
    parser.add_argument('--data', type=str, default='synthetic',
                       choices=['synthetic', 'sdo', 'mixed'],
                       help='Data source for training')
    
    # Training options
    parser.add_argument('--ensemble', action='store_true',
                       help='Train ensemble of models')
    parser.add_argument('--models', type=str, default='deeponet,fno,pinn',
                       help='Comma-separated list of models for ensemble')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of training epochs (overrides config)')
    parser.add_argument('--trials', type=int, default=None,
                       help='Number of hyperparameter optimization trials')
    
    # Advanced options
    parser.add_argument('--gpus', type=int, default=None,
                       help='Number of GPUs to use')
    parser.add_argument('--mixed-precision', action='store_true',
                       help='Use mixed precision training')
    
    args = parser.parse_args()
    
    # Setup configuration manager
    config_manager = ConfigManager()
    
    # Create trainer
    trainer = SolarModelTrainer(config_manager)
    
    # Prepare configuration overrides
    config_overrides = {}
    
    if args.epochs:
        config_overrides['training'] = {'n_epochs': args.epochs}
    
    if args.trials:
        config_overrides['training'] = {'n_trials': args.trials}
    
    if args.gpus:
        config_overrides['environment'] = {'num_gpus': args.gpus}
    
    if args.mixed_precision:
        config_overrides['training'] = {'use_mixed_precision': True}
    
    # Get model configuration
    if args.config in PREDEFINED_CONFIGS:
        model_config = PREDEFINED_CONFIGS[args.config]
    else:
        # Try to load from file
        model_config = config_manager.get_config(args.config.split('_')[0], args.config)
    
    config_overrides['model'] = model_config
    
    # Setup ensemble if requested
    if args.ensemble:
        model_names = args.models.split(',')
        model_configs = []
        
        for name in model_names:
            if name in PREDEFINED_CONFIGS:
                model_configs.append(PREDEFINED_CONFIGS[name])
            else:
                # Create basic config
                if name == 'deeponet':
                    model_configs.append(PREDEFINED_CONFIGS['deeponet_fast'])
                elif name == 'fno':
                    model_configs.append(PREDEFINED_CONFIGS['fno_fast'])
                elif name == 'pinn':
                    model_configs.append(PREDEFINED_CONFIGS['deeponet_fast'])  # Use DeepONet as PINN
        
        config_overrides['model'] = PREDEFINED_CONFIGS['ensemble_basic']
        config_overrides['model'].models = model_configs
    
    # Run experiment
    try:
        results = trainer.run_experiment(
            experiment_name=args.experiment,
            config_overrides=config_overrides,
            data_source=args.data,
            use_ensemble=args.ensemble
        )
        
        print(f"Experiment completed successfully!")
        print(f"Results saved to: {trainer.experiment_dir}")
        print(f"Report generated: {results['report_path']}")
        
    except Exception as e:
        print(f"Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 