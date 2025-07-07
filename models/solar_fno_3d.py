"""
solar_fno_3d.py
---------------
3D Fourier Neural Operator (FNO) for solar magnetic field prediction.
Specialized for spectral modeling of solar magnetic fields with temporal evolution.

Key Features:
- 3D spectral convolutions for spatial modeling
- Temporal modeling for field evolution
- Physics-informed loss functions
- Multi-scale feature extraction
- Efficient FFT-based operations

References:
- Li et al., "Fourier Neural Operator for Parametric Partial Differential Equations" (2021)
- Pathak et al., "FourCastNet: A Global Data-driven High-resolution Weather Model" (2022)
"""
import jax
import jax.numpy as jnp
import numpy as np
import optax
from functools import partial
from typing import Tuple, Dict, Any, Optional, List
import equinox as eqx

class FNO3DBlock(eqx.Module):
    """3D FNO block with spectral convolutions."""
    
    # Spectral convolution weights (complex)
    spectral_weights: jnp.ndarray
    
    # Local linear transformation
    local_weights: jnp.ndarray
    local_bias: jnp.ndarray
    
    # Layer normalization
    layer_norm: eqx.Module
    
    def __init__(self, 
                 modes: Tuple[int, int, int],  # (modes_x, modes_y, modes_z)
                 width: int,
                 key: jax.random.PRNGKey):
        super().__init__()
        
        self.modes = modes
        self.width = width
        
        # Spectral weights (complex)
        spectral_key, local_key = jax.random.split(key)
        scale = 1.0 / (width * width)
        
        # Initialize complex weights for each mode
        self.spectral_weights = (
            jax.random.normal(spectral_key, (modes[0], modes[1], modes[2], width), dtype=jnp.float32) * scale +
            1j * jax.random.normal(spectral_key, (modes[0], modes[1], modes[2], width), dtype=jnp.float32) * scale
        )
        
        # Local linear transformation
        self.local_weights = jax.random.normal(local_key, (width, width), dtype=jnp.float32) * scale
        self.local_bias = jnp.zeros((width,), dtype=jnp.float32)
        
        # Layer normalization
        self.layer_norm = eqx.nn.LayerNorm([width])
    
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Forward pass through 3D FNO block.
        
        Args:
            x: Input tensor of shape (batch, nx, ny, nz, width)
            
        Returns:
            Output tensor of same shape
        """
        # Spectral convolution
        x_ft = jnp.fft.rfftn(x, axes=(1, 2, 3))  # Real FFT for efficiency
        
        # Apply spectral weights (truncate to modes)
        x_ft_truncated = x_ft[:, :self.modes[0], :self.modes[1], :self.modes[2], :]
        x_ft_weighted = x_ft_truncated * self.spectral_weights[None, :, :, :, :]
        
        # Pad back to original size
        x_ft_padded = jnp.zeros_like(x_ft)
        x_ft_padded = x_ft_padded.at[:, :self.modes[0], :self.modes[1], :self.modes[2], :].set(x_ft_weighted)
        
        # Inverse FFT
        x_spectral = jnp.fft.irfftn(x_ft_padded, axes=(1, 2, 3), s=x.shape[1:4])
        
        # Local linear transformation
        x_local = jnp.einsum('bnxyzi,ij->bnxyzj', x, self.local_weights) + self.local_bias
        
        # Combine spectral and local
        x_combined = x_spectral + x_local
        
        # Layer normalization and activation
        x_norm = self.layer_norm(x_combined)
        x_out = jax.nn.gelu(x_norm)
        
        return x_out

class SolarFNO3D(eqx.Module):
    """3D FNO for solar magnetic field prediction."""
    
    # Input projection
    input_proj: eqx.Module
    
    # FNO blocks
    fno_blocks: List[FNO3DBlock]
    
    # Output projection
    output_proj: eqx.Module
    
    # Temporal modeling
    temporal_encoder: eqx.Module
    
    def __init__(self,
                 input_channels: int = 3,  # Bx, By, Bz
                 output_channels: int = 3,  # Bx, By, Bz
                 modes: Tuple[int, int, int] = (16, 16, 8),  # Spectral modes
                 width: int = 64,
                 depth: int = 4,
                 grid_size: Tuple[int, int, int] = (64, 64, 32),
                 key: jax.random.PRNGKey = None):
        super().__init__()
        
        if key is None:
            key = jax.random.PRNGKey(42)
        
        # Input projection: magnetogram + coordinates -> latent space
        input_key, output_key, temporal_key = jax.random.split(key, 3)
        
        self.input_proj = eqx.nn.MLP(
            in_size=input_channels + 3,  # Bx, By, Bz + x, y, z coordinates
            out_size=width,
            width_size=width,
            depth=2,
            key=input_key
        )
        
        # FNO blocks
        fno_keys = jax.random.split(key, depth)
        self.fno_blocks = [
            FNO3DBlock(modes, width, fno_keys[i]) for i in range(depth)
        ]
        
        # Output projection
        self.output_proj = eqx.nn.MLP(
            in_size=width,
            out_size=output_channels,
            width_size=width//2,
            depth=2,
            key=output_key
        )
        
        # Temporal encoder for time-dependent modeling
        self.temporal_encoder = eqx.nn.MLP(
            in_size=1,  # time
            out_size=width,
            width_size=width//2,
            depth=2,
            key=temporal_key
        )
    
    def __call__(self,
                 magnetogram: jnp.ndarray,  # (batch, 3, nx, ny) - 2D magnetogram
                 coords: jnp.ndarray,       # (batch, nx, ny, nz, 3) - 3D coordinates
                 time: Optional[jnp.ndarray] = None,  # (batch,) - time steps
                 ) -> jnp.ndarray:
        """
        Forward pass: predict 3D magnetic field evolution.
        
        Args:
            magnetogram: 2D vector magnetogram (Bx, By, Bz)
            coords: 3D coordinate grid
            time: time steps for temporal modeling
            
        Returns:
            B_field: predicted 3D magnetic field (batch, nx, ny, nz, 3)
        """
        batch_size = magnetogram.shape[0]
        nx, ny, nz = coords.shape[1:4]
        
        # Encode temporal information
        if time is None:
            time = jnp.zeros((batch_size,))
        
        temporal_features = self.temporal_encoder(time[:, None])  # (batch, width)
        temporal_features = temporal_features[:, None, None, None, :]  # (batch, 1, 1, 1, width)
        temporal_features = jnp.tile(temporal_features, (1, nx, ny, nz, 1))  # (batch, nx, ny, nz, width)
        
        # Prepare input: combine magnetogram and coordinates
        # Interpolate 2D magnetogram to 3D grid
        magnetogram_3d = magnetogram[:, :, :, :, None]  # (batch, 3, nx, ny, 1)
        magnetogram_3d = jnp.tile(magnetogram_3d, (1, 1, 1, 1, nz))  # (batch, 3, nx, ny, nz)
        magnetogram_3d = jnp.transpose(magnetogram_3d, (0, 2, 3, 4, 1))  # (batch, nx, ny, nz, 3)
        
        # Combine with coordinates
        input_features = jnp.concatenate([magnetogram_3d, coords], axis=-1)  # (batch, nx, ny, nz, 6)
        
        # Input projection
        x = self.input_proj(input_features)  # (batch, nx, ny, nz, width)
        
        # Add temporal features
        x = x + temporal_features
        
        # FNO blocks
        for fno_block in self.fno_blocks:
            x = fno_block(x)
        
        # Output projection
        B_field = self.output_proj(x)  # (batch, nx, ny, nz, 3)
        
        return B_field

class PhysicsInformedFNOLoss:
    """Physics-informed loss for 3D FNO solar magnetic field prediction."""
    
    def __init__(self, 
                 lambda_data: float = 1.0,
                 lambda_physics: float = 0.1,
                 lambda_divergence: float = 1.0,
                 lambda_curl: float = 0.1):
        self.lambda_data = lambda_data
        self.lambda_physics = lambda_physics
        self.lambda_divergence = lambda_divergence
        self.lambda_curl = lambda_curl
    
    def __call__(self,
                 model: SolarFNO3D,
                 params: Dict[str, Any],
                 magnetogram: jnp.ndarray,
                 coords: jnp.ndarray,
                 B_true: jnp.ndarray,
                 time: Optional[jnp.ndarray] = None) -> Tuple[float, Dict[str, float]]:
        """
        Compute physics-informed loss for 3D FNO.
        
        Args:
            model: FNO model
            params: model parameters
            magnetogram: input 2D magnetogram
            coords: 3D coordinate grid
            B_true: true 3D magnetic field
            time: time steps
            
        Returns:
            total_loss: combined loss
            loss_components: individual loss terms
        """
        # Data loss
        B_pred = model(params, magnetogram, coords, time)
        data_loss = jnp.mean((B_pred - B_true) ** 2)
        
        # Physics losses
        div_loss = self._divergence_loss(model, params, magnetogram, coords, time)
        curl_loss = self._curl_loss(model, params, magnetogram, coords, time)
        
        # Total loss
        total_loss = (self.lambda_data * data_loss + 
                     self.lambda_divergence * div_loss + 
                     self.lambda_curl * curl_loss)
        
        loss_components = {
            'data_loss': data_loss,
            'divergence_loss': div_loss,
            'curl_loss': curl_loss,
            'total_loss': total_loss
        }
        
        return total_loss, loss_components
    
    def _divergence_loss(self, model, params, magnetogram, coords, time):
        """Compute divergence-free constraint: ∇ · B = 0."""
        def B_field_fn(coords_batch):
            return model(params, magnetogram, coords_batch, time)
        
        # Compute divergence using finite differences
        B_pred = model(params, magnetogram, coords, time)
        
        # Finite difference approximation of divergence
        dx = coords[0, 1, 0, 0, 0] - coords[0, 0, 0, 0, 0]
        dy = coords[0, 0, 1, 0, 1] - coords[0, 0, 0, 0, 1]
        dz = coords[0, 0, 0, 1, 2] - coords[0, 0, 0, 0, 2]
        
        # Compute gradients using finite differences
        dBx_dx = (B_pred[:, 1:, :, :, 0] - B_pred[:, :-1, :, :, 0]) / dx
        dBy_dy = (B_pred[:, :, 1:, :, 1] - B_pred[:, :, :-1, :, 1]) / dy
        dBz_dz = (B_pred[:, :, :, 1:, 2] - B_pred[:, :, :, :-1, 2]) / dz
        
        # Divergence
        div = dBx_dx[:, :, :, :-1] + dBy_dy[:, :-1, :, :-1] + dBz_dz[:, :-1, :-1, :]
        
        return jnp.mean(div ** 2)
    
    def _curl_loss(self, model, params, magnetogram, coords, time):
        """Compute curl constraint for force-free field."""
        B_pred = model(params, magnetogram, coords, time)
        
        # Finite difference approximation of curl
        dx = coords[0, 1, 0, 0, 0] - coords[0, 0, 0, 0, 0]
        dy = coords[0, 0, 1, 0, 1] - coords[0, 0, 0, 0, 1]
        dz = coords[0, 0, 0, 1, 2] - coords[0, 0, 0, 0, 2]
        
        # Curl components
        curl_x = ((B_pred[:, :, 1:, :, 2] - B_pred[:, :, :-1, :, 2]) / dy[:, :-1, :-1, :] -
                  (B_pred[:, :, :, 1:, 1] - B_pred[:, :, :, :-1, 1]) / dz[:, :-1, :, :-1])
        
        curl_y = ((B_pred[:, :, :, 1:, 0] - B_pred[:, :, :, :-1, 0]) / dz[:, :-1, :, :-1] -
                  (B_pred[:, 1:, :, :, 2] - B_pred[:, :-1, :, :, 2]) / dx[:, :, :-1, :-1])
        
        curl_z = ((B_pred[:, 1:, :, :, 1] - B_pred[:, :-1, :, :, 1]) / dx[:, :, :-1, :-1] -
                  (B_pred[:, :, 1:, :, 0] - B_pred[:, :, :-1, :, 0]) / dy[:, :-1, :, :-1])
        
        # Curl magnitude
        curl_magnitude = jnp.sqrt(curl_x ** 2 + curl_y ** 2 + curl_z ** 2)
        
        return jnp.mean(curl_magnitude ** 2)

def create_solar_fno_training_step(model, loss_fn, optimizer):
    """Create JIT-compiled training step for Solar FNO."""
    
    @jax.jit
    def training_step(params, opt_state, magnetogram, coords, B_true, time=None):
        """Single training step."""
        (loss, loss_components), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            model, params, magnetogram, coords, B_true, time
        )
        
        updates, new_opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        
        return new_params, new_opt_state, loss, loss_components
    
    return training_step

def generate_solar_test_data(batch_size: int = 2,
                           grid_size: Tuple[int, int, int] = (32, 32, 16),
                           key: jax.random.PRNGKey = None):
    """Generate synthetic solar magnetic field data for testing."""
    if key is None:
        key = jax.random.PRNGKey(42)
    
    nx, ny, nz = grid_size
    
    # Generate 2D magnetogram
    mag_key, coord_key, field_key, time_key = jax.random.split(key, 4)
    magnetogram = jax.random.normal(mag_key, (batch_size, 3, nx, ny))
    
    # Generate 3D coordinate grid
    x = jnp.linspace(-1, 1, nx)
    y = jnp.linspace(-1, 1, ny)
    z = jnp.linspace(0, 2, nz)  # Height above surface
    
    X, Y, Z = jnp.meshgrid(x, y, z, indexing='ij')
    coords = jnp.stack([X, Y, Z], axis=-1)  # (nx, ny, nz, 3)
    coords = jnp.tile(coords[None, :, :, :, :], (batch_size, 1, 1, 1, 1))
    
    # Generate synthetic 3D magnetic field (Low & Lou inspired)
    B_true = generate_low_lou_field(coords, field_key)
    
    # Generate time steps
    time = jax.random.uniform(time_key, (batch_size,), minval=0, maxval=1)
    
    return magnetogram, coords, B_true, time

def generate_low_lou_field(coords: jnp.ndarray, key: jax.random.PRNGKey) -> jnp.ndarray:
    """Generate synthetic magnetic field based on Low & Lou model."""
    x, y, z = coords[..., 0], coords[..., 1], coords[..., 2]
    
    # Simplified force-free field
    r = jnp.sqrt(x**2 + y**2 + z**2) + 1e-8
    theta = jnp.arccos(z / r)
    phi = jnp.arctan2(y, x)
    
    # Force-free parameter
    alpha = 0.5
    
    # Spherical components
    Br = jnp.cos(theta) * jnp.sin(alpha * r) / r
    Btheta = jnp.sin(theta) * jnp.sin(alpha * r) / r
    Bphi = jnp.sin(alpha * r) / r
    
    # Convert to Cartesian
    Bx = (Br * jnp.sin(theta) * jnp.cos(phi) +
          Btheta * jnp.cos(theta) * jnp.cos(phi) -
          Bphi * jnp.sin(phi))
    By = (Br * jnp.sin(theta) * jnp.sin(phi) +
          Btheta * jnp.cos(theta) * jnp.sin(phi) +
          Bphi * jnp.cos(phi))
    Bz = Br * jnp.cos(theta) - Btheta * jnp.sin(theta)
    
    # Add some noise for realism
    noise_key = jax.random.split(key, 3)
    noise_scale = 0.1
    Bx += noise_scale * jax.random.normal(noise_key[0], Bx.shape)
    By += noise_scale * jax.random.normal(noise_key[1], By.shape)
    Bz += noise_scale * jax.random.normal(noise_key[2], Bz.shape)
    
    return jnp.stack([Bx, By, Bz], axis=-1)

def test_solar_fno():
    """Test the 3D Solar FNO implementation."""
    key = jax.random.PRNGKey(42)
    
    # Create model
    model = SolarFNO3D(
        input_channels=3,
        output_channels=3,
        modes=(8, 8, 4),  # Reduced for testing
        width=32,
        depth=3,
        grid_size=(16, 16, 8),  # Reduced for testing
        key=key
    )
    
    # Generate test data
    magnetogram, coords, B_true, time = generate_solar_test_data(
        batch_size=2,
        grid_size=(16, 16, 8),
        key=key
    )
    
    # Test forward pass
    B_pred = model(magnetogram, coords, time)
    print(f"Prediction shape: {B_pred.shape}")
    print(f"True field shape: {B_true.shape}")
    
    # Test loss computation
    loss_fn = PhysicsInformedFNOLoss(
        lambda_data=1.0,
        lambda_physics=0.1,
        lambda_divergence=1.0,
        lambda_curl=0.1
    )
    
    params = model.parameters()
    loss, components = loss_fn(model, params, magnetogram, coords, B_true, time)
    print(f"Loss components: {components}")
    
    # Test training step
    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(params)
    training_step = create_solar_fno_training_step(model, loss_fn, optimizer)
    
    new_params, new_opt_state, loss, components = training_step(
        params, opt_state, magnetogram, coords, B_true, time
    )
    print(f"Training step completed. Loss: {loss:.6f}")
    
    return model, loss_fn, training_step

if __name__ == "__main__":
    model, loss_fn, training_step = test_solar_fno()
    print("Solar FNO 3D implementation tested successfully!") 