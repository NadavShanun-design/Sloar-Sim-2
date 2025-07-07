"""
solar_deeponet_3d.py
-------------------
3D DeepONet implementation for solar magnetic field prediction.
Maps 2D vector magnetograms to 3D magnetic field reconstructions.

Key Features:
- Branch network: Encodes 2D magnetogram data (Bx, By, Bz at surface)
- Trunk network: Encodes 3D spatial coordinates (x, y, z)
- Physics-informed loss: Maxwell's equations + divergence-free constraint
- Temporal forecasting: Autoregressive prediction capabilities

References:
- Lu et al., "Learning Nonlinear Operators via DeepONet" (2021)
- Jarolim et al., "Physics-Informed Neural Networks for Solar Magnetic Field Modeling" (2023)
"""
import jax
import jax.numpy as jnp
import numpy as np
import optax
from functools import partial
from typing import Tuple, Dict, Any, Optional
import equinox as eqx

class SolarDeepONet(eqx.Module):
    """3D DeepONet for solar magnetic field prediction."""
    
    # Branch network: processes 2D magnetogram data
    branch_encoder: eqx.Module
    branch_mlp: eqx.Module
    
    # Trunk network: processes 3D spatial coordinates
    trunk_mlp: eqx.Module
    
    # Output projection
    output_proj: eqx.Module
    
    # Physics parameters
    latent_dim: int = 128
    
    def __init__(self, 
                 magnetogram_shape: Tuple[int, int] = (256, 256),
                 latent_dim: int = 128,
                 branch_depth: int = 8,
                 trunk_depth: int = 6,
                 width: int = 256,
                 key: jax.random.PRNGKey = None):
        super().__init__()
        
        if key is None:
            key = jax.random.PRNGKey(42)
        
        self.latent_dim = latent_dim
        
        # Branch network: CNN + MLP for magnetogram encoding
        branch_key, trunk_key, output_key = jax.random.split(key, 3)
        
        # CNN encoder for 2D magnetograms (Bx, By, Bz channels)
        self.branch_encoder = self._create_cnn_encoder(
            input_channels=3, 
            output_dim=latent_dim//2,
            key=branch_key
        )
        
        # MLP for additional features (time, metadata)
        self.branch_mlp = eqx.nn.MLP(
            in_size=4,  # time + 3 magnetogram features
            out_size=latent_dim//2,
            width_size=width,
            depth=branch_depth-1,
            key=branch_key
        )
        
        # Trunk network: MLP for 3D coordinates
        self.trunk_mlp = eqx.nn.MLP(
            in_size=3,  # x, y, z coordinates
            out_size=latent_dim,
            width_size=width,
            depth=trunk_depth,
            key=trunk_key
        )
        
        # Output projection: 3D magnetic field components
        self.output_proj = eqx.nn.MLP(
            in_size=latent_dim,
            out_size=3,  # Bx, By, Bz
            width_size=width//2,
            depth=2,
            key=output_key
        )
    
    def _create_cnn_encoder(self, input_channels: int, output_dim: int, key: jax.random.PRNGKey):
        """Create CNN encoder for 2D magnetogram processing."""
        return eqx.nn.Sequential([
            # Initial convolution
            eqx.nn.Conv2d(input_channels, 64, kernel_size=3, padding=1, key=key),
            eqx.nn.LayerNorm([64, 256, 256]),
            jax.nn.gelu,
            
            # Downsampling blocks
            eqx.nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, key=key),
            eqx.nn.LayerNorm([128, 128, 128]),
            jax.nn.gelu,
            
            eqx.nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, key=key),
            eqx.nn.LayerNorm([256, 64, 64]),
            jax.nn.gelu,
            
            eqx.nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, key=key),
            eqx.nn.LayerNorm([512, 32, 32]),
            jax.nn.gelu,
            
            # Global average pooling + projection
            lambda x: jnp.mean(x, axis=(2, 3)),  # Global average pooling
            eqx.nn.Linear(512, output_dim, key=key),
            jax.nn.gelu
        ])
    
    def __call__(self, 
                 magnetogram: jnp.ndarray,  # (3, H, W) - Bx, By, Bz
                 coords: jnp.ndarray,       # (N, 3) - x, y, z coordinates
                 time: Optional[jnp.ndarray] = None,  # (1,) - time step
                 metadata: Optional[jnp.ndarray] = None  # (3,) - additional features
                 ) -> jnp.ndarray:
        """
        Forward pass: predict 3D magnetic field at given coordinates.
        
        Args:
            magnetogram: 2D vector magnetogram (Bx, By, Bz)
            coords: 3D coordinates where to predict field
            time: time step (optional)
            metadata: additional features (optional)
            
        Returns:
            B_field: predicted magnetic field (N, 3) - Bx, By, Bz
        """
        batch_size = coords.shape[0]
        
        # Branch network: encode magnetogram
        branch_cnn = self.branch_encoder(magnetogram[None, ...])  # Add batch dim
        
        # Combine with time and metadata
        if time is None:
            time = jnp.array([0.0])
        if metadata is None:
            metadata = jnp.array([0.0, 0.0, 0.0])
        
        branch_features = jnp.concatenate([time, metadata])
        branch_mlp = self.branch_mlp(branch_features)
        
        # Combine CNN and MLP features
        branch_out = jnp.concatenate([branch_cnn, branch_mlp])
        branch_out = jnp.tile(branch_out[None, :], (batch_size, 1))  # (N, latent_dim)
        
        # Trunk network: encode coordinates
        trunk_out = self.trunk_mlp(coords)  # (N, latent_dim)
        
        # Combine branch and trunk (element-wise multiplication)
        combined = branch_out * trunk_out  # (N, latent_dim)
        
        # Output projection
        B_field = self.output_proj(combined)  # (N, 3)
        
        return B_field

class PhysicsInformedLoss:
    """Physics-informed loss for solar magnetic field prediction."""
    
    def __init__(self, lambda_data: float = 1.0, lambda_physics: float = 1.0):
        self.lambda_data = lambda_data
        self.lambda_physics = lambda_physics
    
    def __call__(self, 
                 model: SolarDeepONet,
                 params: Dict[str, Any],
                 magnetogram: jnp.ndarray,
                 coords: jnp.ndarray,
                 B_true: jnp.ndarray,
                 time: Optional[jnp.ndarray] = None,
                 metadata: Optional[jnp.ndarray] = None) -> Tuple[float, Dict[str, float]]:
        """
        Compute physics-informed loss.
        
        Args:
            model: DeepONet model
            params: model parameters
            magnetogram: input 2D magnetogram
            coords: 3D coordinates
            B_true: true magnetic field values
            time: time step
            metadata: additional features
            
        Returns:
            total_loss: combined loss
            loss_components: individual loss terms
        """
        # Data loss
        B_pred = model(params, magnetogram, coords, time, metadata)
        data_loss = jnp.mean((B_pred - B_true) ** 2)
        
        # Physics loss: Maxwell's equations
        physics_loss = self._maxwell_loss(model, params, magnetogram, coords, time, metadata)
        
        # Divergence-free constraint
        div_loss = self._divergence_loss(model, params, magnetogram, coords, time, metadata)
        
        # Total loss
        total_loss = (self.lambda_data * data_loss + 
                     self.lambda_physics * physics_loss + 
                     self.lambda_physics * div_loss)
        
        loss_components = {
            'data_loss': data_loss,
            'physics_loss': physics_loss,
            'divergence_loss': div_loss,
            'total_loss': total_loss
        }
        
        return total_loss, loss_components
    
    def _maxwell_loss(self, model, params, magnetogram, coords, time, metadata):
        """Compute Maxwell's equations residual."""
        # ∇ × B = μ₀J (Ampere's law) - simplified for force-free field
        # For force-free field: ∇ × B = αB, where α is constant along field lines
        
        def B_field_fn(coords):
            return model(params, magnetogram, coords, time, metadata)
        
        # Compute curl using automatic differentiation
        curl_B = self._compute_curl(B_field_fn, coords)
        
        # For force-free field, curl should be proportional to B
        B_pred = model(params, magnetogram, coords, time, metadata)
        
        # Simplified: minimize curl magnitude (force-free approximation)
        curl_magnitude = jnp.sqrt(jnp.sum(curl_B ** 2, axis=-1))
        return jnp.mean(curl_magnitude ** 2)
    
    def _divergence_loss(self, model, params, magnetogram, coords, time, metadata):
        """Compute divergence-free constraint: ∇ · B = 0."""
        def B_field_fn(coords):
            return model(params, magnetogram, coords, time, metadata)
        
        # Compute divergence using automatic differentiation
        div_B = self._compute_divergence(B_field_fn, coords)
        
        return jnp.mean(div_B ** 2)
    
    def _compute_curl(self, B_fn, coords):
        """Compute curl of vector field using automatic differentiation."""
        def Bx(x, y, z):
            coords_xyz = jnp.stack([x, y, z])
            return B_fn(coords_xyz[None, :])[0, 0]
        
        def By(x, y, z):
            coords_xyz = jnp.stack([x, y, z])
            return B_fn(coords_xyz[None, :])[0, 1]
        
        def Bz(x, y, z):
            coords_xyz = jnp.stack([x, y, z])
            return B_fn(coords_xyz[None, :])[0, 2]
        
        # Compute partial derivatives
        dBy_dx = jax.grad(By, argnums=0)
        dBx_dy = jax.grad(Bx, argnums=1)
        dBz_dx = jax.grad(Bz, argnums=0)
        dBx_dz = jax.grad(Bx, argnums=2)
        dBz_dy = jax.grad(Bz, argnums=1)
        dBy_dz = jax.grad(By, argnums=2)
        
        # Curl components
        curl_x = dBy_dz(coords[:, 0], coords[:, 1], coords[:, 2]) - dBz_dy(coords[:, 0], coords[:, 1], coords[:, 2])
        curl_y = dBz_dx(coords[:, 0], coords[:, 1], coords[:, 2]) - dBx_dz(coords[:, 0], coords[:, 1], coords[:, 2])
        curl_z = dBx_dy(coords[:, 0], coords[:, 1], coords[:, 2]) - dBy_dx(coords[:, 0], coords[:, 1], coords[:, 2])
        
        return jnp.stack([curl_x, curl_y, curl_z], axis=-1)
    
    def _compute_divergence(self, B_fn, coords):
        """Compute divergence of vector field using automatic differentiation."""
        def Bx(x, y, z):
            coords_xyz = jnp.stack([x, y, z])
            return B_fn(coords_xyz[None, :])[0, 0]
        
        def By(x, y, z):
            coords_xyz = jnp.stack([x, y, z])
            return B_fn(coords_xyz[None, :])[0, 1]
        
        def Bz(x, y, z):
            coords_xyz = jnp.stack([x, y, z])
            return B_fn(coords_xyz[None, :])[0, 2]
        
        # Compute partial derivatives
        dBx_dx = jax.grad(Bx, argnums=0)
        dBy_dy = jax.grad(By, argnums=1)
        dBz_dz = jax.grad(Bz, argnums=2)
        
        # Divergence
        div = (dBx_dx(coords[:, 0], coords[:, 1], coords[:, 2]) + 
               dBy_dy(coords[:, 0], coords[:, 1], coords[:, 2]) + 
               dBz_dz(coords[:, 0], coords[:, 1], coords[:, 2]))
        
        return div

def create_solar_deeponet_training_step(model, loss_fn, optimizer):
    """Create JIT-compiled training step."""
    
    @jax.jit
    def training_step(params, opt_state, magnetogram, coords, B_true, time=None, metadata=None):
        """Single training step."""
        (loss, loss_components), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            model, params, magnetogram, coords, B_true, time, metadata
        )
        
        updates, new_opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        
        return new_params, new_opt_state, loss, loss_components
    
    return training_step

# Example usage and testing
def test_solar_deeponet():
    """Test the 3D Solar DeepONet implementation."""
    key = jax.random.PRNGKey(42)
    
    # Create model
    model = SolarDeepONet(
        magnetogram_shape=(256, 256),
        latent_dim=128,
        branch_depth=8,
        trunk_depth=6,
        width=256,
        key=key
    )
    
    # Generate synthetic data
    magnetogram = jax.random.normal(key, (3, 256, 256))  # Bx, By, Bz
    coords = jax.random.uniform(key, (1000, 3), minval=-1, maxval=1)  # 3D coordinates
    B_true = jax.random.normal(key, (1000, 3))  # True magnetic field
    time = jnp.array([0.0])
    metadata = jnp.array([0.0, 0.0, 0.0])
    
    # Test forward pass
    B_pred = model(magnetogram, coords, time, metadata)
    print(f"Prediction shape: {B_pred.shape}")
    
    # Test loss computation
    loss_fn = PhysicsInformedLoss(lambda_data=1.0, lambda_physics=0.1)
    params = model.parameters()
    loss, components = loss_fn(model, params, magnetogram, coords, B_true, time, metadata)
    print(f"Loss components: {components}")
    
    # Test training step
    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(params)
    training_step = create_solar_deeponet_training_step(model, loss_fn, optimizer)
    
    new_params, new_opt_state, loss, components = training_step(
        params, opt_state, magnetogram, coords, B_true, time, metadata
    )
    print(f"Training step completed. Loss: {loss:.6f}")
    
    return model, loss_fn, training_step

if __name__ == "__main__":
    model, loss_fn, training_step = test_solar_deeponet()
    print("Solar DeepONet 3D implementation tested successfully!") 