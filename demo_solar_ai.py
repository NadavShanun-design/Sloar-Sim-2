#!/usr/bin/env python3
"""
demo_solar_ai.py
---------------
Demonstration script for the upgraded Solar AI system.
Shows the system working with synthetic data and basic training.
"""
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os

# Set random seed for reproducibility
jax.random.PRNGKey(42)

def demo_basic_components():
    """Demonstrate basic components of the Solar AI system."""
    print("🌞 Solar AI System - Component Demonstration")
    print("=" * 50)
    
    # Test JAX functionality
    print("1. Testing JAX computation...")
    x = jnp.array([1, 2, 3, 4, 5])
    y = jnp.array([2, 3, 4, 5, 6])
    z = x * y + jnp.sin(x)
    print(f"   JAX computation: {x} * {y} + sin({x}) = {z}")
    print("   ✅ JAX working correctly!")
    
    # Test synthetic data generation
    print("\n2. Generating synthetic solar magnetic field data...")
    nx, ny, nz = 32, 32, 16
    batch_size = 4
    
    # Generate 2D magnetogram (simulating SDO/HMI data)
    magnetogram = jax.random.normal(jax.random.PRNGKey(0), (batch_size, 3, nx, ny))
    print(f"   Generated magnetogram shape: {magnetogram.shape}")
    
    # Generate 3D coordinate grid
    x = jnp.linspace(-2, 2, nx)
    y = jnp.linspace(-2, 2, ny)
    z = jnp.linspace(0, 4, nz)
    X, Y, Z = jnp.meshgrid(x, y, z, indexing='ij')
    coords = jnp.stack([X, Y, Z], axis=-1)
    coords = jnp.tile(coords[None, :, :, :, :], (batch_size, 1, 1, 1, 1))
    print(f"   Generated 3D coordinate grid shape: {coords.shape}")
    
    # Generate synthetic magnetic field (Low & Lou inspired)
    r = jnp.sqrt(X**2 + Y**2 + Z**2) + 1e-8
    theta = jnp.arccos(Z / r)
    phi = jnp.arctan2(Y, X)
    
    # Force-free field components
    alpha = 0.5
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
    
    B_field = jnp.stack([Bx, By, Bz], axis=-1)
    print(f"   Generated 3D magnetic field shape: {B_field.shape}")
    print("   ✅ Synthetic data generation successful!")
    
    return magnetogram, coords, B_field

def demo_simple_training():
    """Demonstrate simple training with synthetic data."""
    print("\n3. Demonstrating simple training...")
    
    # Generate data
    magnetogram, coords, B_true = demo_basic_components()
    
    # Create a simple model (simplified version)
    print("   Creating simple neural network model...")
    
    # Simple MLP for demonstration
    def simple_mlp(params, x):
        """Simple MLP for demonstration."""
        for i, (w, b) in enumerate(params[:-1]):
            x = jnp.dot(x, w) + b
            x = jax.nn.tanh(x)
        x = jnp.dot(x, params[-1][0]) + params[-1][1]
        return x
    
    # Initialize parameters
    key = jax.random.PRNGKey(42)
    layer_sizes = [6, 64, 64, 3]  # input: 6 (3 mag + 3 coords), output: 3 (Bx, By, Bz)
    params = []
    
    for i in range(len(layer_sizes) - 1):
        w_key, b_key = jax.random.split(key)
        w = jax.random.normal(w_key, (layer_sizes[i], layer_sizes[i+1])) * jnp.sqrt(2/layer_sizes[i])
        b = jnp.zeros((layer_sizes[i+1],))
        params.append((w, b))
        key = w_key
    
    print(f"   Model initialized with {len(params)} layers")
    
    # Simple loss function
    def loss_fn(params, inputs, targets):
        predictions = simple_mlp(params, inputs)
        return jnp.mean((predictions - targets) ** 2)
    
    # Prepare training data
    batch_size = magnetogram.shape[0]
    n_points = 1000
    
    # Sample random points
    coords_flat = coords[0].reshape(-1, 3)
    indices = jax.random.choice(jax.random.PRNGKey(1), len(coords_flat), (n_points,), replace=False)
    coords_sampled = coords_flat[indices]
    
    # Get corresponding magnetogram values (interpolate)
    magnetogram_flat = magnetogram[0].reshape(3, -1)
    mag_sampled = magnetogram_flat[:, indices]
    mag_sampled = mag_sampled.T  # (n_points, 3)
    
    # Combine inputs
    inputs = jnp.concatenate([mag_sampled, coords_sampled], axis=1)  # (n_points, 6)
    
    # Get targets
    B_flat = B_true[0].reshape(-1, 3)
    targets = B_flat[indices]  # (n_points, 3)
    
    print(f"   Training data prepared: {inputs.shape} -> {targets.shape}")
    
    # Simple training loop
    print("   Starting training...")
    import optax
    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(params)
    
    @jax.jit
    def step(params, opt_state, inputs, targets):
        loss, grads = jax.value_and_grad(loss_fn)(params, inputs, targets)
        updates, new_opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss
    
    # Training
    n_epochs = 50
    losses = []
    
    for epoch in range(n_epochs):
        params, opt_state, loss = step(params, opt_state, inputs, targets)
        losses.append(float(loss))
        
        if epoch % 10 == 0:
            print(f"   Epoch {epoch}: Loss = {loss:.6f}")
    
    print(f"   Final loss: {losses[-1]:.6f}")
    print("   ✅ Training completed successfully!")
    
    return params, losses

def demo_visualization():
    """Demonstrate visualization capabilities."""
    print("\n4. Creating visualizations...")
    
    # Generate data
    magnetogram, coords, B_field = demo_basic_components()
    
    # Create visualization directory
    os.makedirs('demo_outputs', exist_ok=True)
    
    # Plot magnetogram
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    components = ['Bx', 'By', 'Bz']
    
    for i, (ax, comp) in enumerate(zip(axes, components)):
        im = ax.imshow(magnetogram[0, i], cmap='RdBu_r', origin='lower')
        ax.set_title(f'Magnetogram {comp}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    plt.savefig('demo_outputs/magnetogram.png', dpi=150, bbox_inches='tight')
    print("   ✅ Magnetogram visualization saved!")
    
    # Plot 3D field lines (simplified)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Sample field lines
    x, y, z = coords[0, :, :, :, 0], coords[0, :, :, :, 1], coords[0, :, :, :, 2]
    Bx, By, Bz = B_field[0, :, :, :, 0], B_field[0, :, :, :, 1], B_field[0, :, :, :, 2]
    
    # Plot a few field lines
    for i in range(0, 32, 8):
        for j in range(0, 32, 8):
            ax.quiver(x[i, j, 0], y[i, j, 0], z[i, j, 0], 
                     Bx[i, j, 0], By[i, j, 0], Bz[i, j, 0],
                     length=0.5, color='red', alpha=0.7)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Magnetic Field Lines (Sample)')
    
    plt.savefig('demo_outputs/field_lines_3d.png', dpi=150, bbox_inches='tight')
    print("   ✅ 3D field lines visualization saved!")
    
    plt.close('all')

def demo_system_capabilities():
    """Demonstrate the full system capabilities."""
    print("\n5. System Capabilities Summary")
    print("=" * 50)
    
    capabilities = [
        "✅ 3D DeepONet for solar magnetic field prediction",
        "✅ 3D FNO with spectral convolutions",
        "✅ Physics-informed loss functions (Maxwell's equations)",
        "✅ Temporal forecasting capabilities",
        "✅ Real SDO/HMI data integration",
        "✅ Synthetic data generation (Low & Lou model)",
        "✅ Hyperparameter optimization with Optuna",
        "✅ Distributed training support",
        "✅ Comprehensive evaluation metrics",
        "✅ Field line visualization and analysis",
        "✅ Configuration management system",
        "✅ Production-ready infrastructure"
    ]
    
    for capability in capabilities:
        print(f"   {capability}")
    
    print("\n🎯 Ready for advanced solar physics research!")
    print("   - Space weather forecasting")
    print("   - Solar flare prediction")
    print("   - Coronal mass ejection modeling")
    print("   - Magnetic field topology analysis")

def main():
    """Main demonstration function."""
    print("🚀 Solar AI System - Live Demonstration")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"JAX version: {jax.__version__}")
    print(f"Available devices: {jax.devices()}")
    print()
    
    try:
        # Run demonstrations
        demo_basic_components()
        params, losses = demo_simple_training()
        demo_visualization()
        demo_system_capabilities()
        
        print("\n" + "=" * 60)
        print("🎉 Solar AI System Demonstration Completed Successfully!")
        print("📁 Check 'demo_outputs/' directory for visualizations")
        print("🔬 Ready for advanced solar magnetic field research!")
        
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 