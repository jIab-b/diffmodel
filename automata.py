import numpy as np
from scipy.signal import convolve2d
from PIL import Image, ImageDraw
import torch
from torchvision import transforms
from torchvision.utils import save_image

# --- 1. Procedural Generation Methods ---

def cellular_automata(width, height, steps=50, rule='game_of_life'):
    """Generates a pattern using Cellular Automata."""
    grid = np.random.choice([0, 1], size=(height, width))
    
    if rule == 'game_of_life':
        kernel = np.array([[1, 1, 1],
                           [1, 0, 1],
                           [1, 1, 1]])
        for _ in range(steps):
            neighbors = convolve2d(grid, kernel, mode='same', boundary='wrap')
            grid = ((grid == 1) & ((neighbors == 2) | (neighbors == 3))) | \
                   ((grid == 0) & (neighbors == 3))
            grid = grid.astype(int)
            
    return torch.from_numpy(grid).float()

def perlin_noise(width, height, scale=100.0, octaves=6, persistence=0.5, lacunarity=2.0):
    """Generates Perlin noise."""
    # This is a simplified implementation. For real use, a library like 'noise' is better.
    shape = (height, width)
    noise = np.zeros(shape)
    for i in range(octaves):
        freq = lacunarity**i
        amp = persistence**i
        
        # Generate noise at a smaller resolution
        small_shape = (int(height / freq), int(width / freq))
        if small_shape[0] < 1 or small_shape[1] < 1: continue # Skip if too small
        
        # Create a PIL Image from the noise array
        small_noise = np.random.randn(*small_shape)
        img = Image.fromarray(small_noise)
        
        # Resize it to the full shape using PIL
        resized_img = img.resize(shape, Image.BICUBIC)
        
        noise += amp * np.array(resized_img)
        
    return torch.from_numpy(noise).float()


def reaction_diffusion(width, height, steps=100, f=0.055, k=0.062):
    """Generates a pattern using the Gray-Scott Reaction-Diffusion model."""
    u = np.ones((height, width))
    v = np.zeros((height, width))

    # Initial disturbance
    r = 20
    u[height//2-r:height//2+r, width//2-r:width//2+r] = 0.5
    v[height//2-r:height//2+r, width//2-r:width//2+r] = 0.25
    
    u += 0.01 * np.random.rand(height, width)
    v += 0.01 * np.random.rand(height, width)

    laplacian_kernel = np.array([[0.05, 0.2, 0.05],
                                 [0.2, -1, 0.2],
                                 [0.05, 0.2, 0.05]])

    for _ in range(steps):
        lu = convolve2d(u, laplacian_kernel, mode='same', boundary='wrap')
        lv = convolve2d(v, laplacian_kernel, mode='same', boundary='wrap')
        uvv = u * v * v
        u += (1 * lu - uvv + f * (1 - u))
        v += (0.5 * lv + uvv - (k + f) * v)
        
    return torch.from_numpy(u).float()

# --- 2. ProcGen Mixer ---

class ProcGenMixer:
    def __init__(self, configs):
        self.configs = configs

    def generate_noise(self, config_name, target_shape):
        if config_name not in self.configs:
            raise ValueError(f"Config '{config_name}' not found.")
        
        config = self.configs[config_name]
        
        final_noise = torch.zeros(target_shape)
        
        for mix_item in config['mix']:
            method = mix_item['method']
            params = mix_item['params']
            weight = mix_item['weight']
            
            # Generate a 2D pattern and flatten it
            pattern_2d = method(**params)
            pattern_flat = pattern_2d.flatten()
            
            # Resize flat pattern to match target size
            current_noise = torch.zeros(target_shape).flatten()
            n_elements = min(len(pattern_flat), len(current_noise))
            current_noise[:n_elements] = pattern_flat[:n_elements]
            current_noise = current_noise.reshape(target_shape)

            final_noise += weight * current_noise
            
        # Normalize
        final_noise = (final_noise - final_noise.mean()) / final_noise.std()
        return final_noise

# --- 3. Noise Configurations ---

NOISE_CONFIGS = {
    'ca_only': {
        'mix': [
            {'method': cellular_automata, 'params': {'width': 64, 'height': 64, 'steps': 20}, 'weight': 1.0}
        ]
    },
    'perlin_only': {
        'mix': [
            {'method': perlin_noise, 'params': {'width': 64, 'height': 64, 'octaves': 4}, 'weight': 1.0}
        ]
    },
    'rd_only': {
        'mix': [
            {'method': reaction_diffusion, 'params': {'width': 64, 'height': 64, 'steps': 50}, 'weight': 1.0}
        ]
    },
    'ca_plus_perlin': {
        'mix': [
            {'method': cellular_automata, 'params': {'width': 64, 'height': 64, 'steps': 15}, 'weight': 0.6},
            {'method': perlin_noise, 'params': {'width': 64, 'height': 64, 'octaves': 3}, 'weight': 0.4}
        ]
    },
    'rd_and_ca': {
        'mix': [
            {'method': reaction_diffusion, 'params': {'width': 64, 'height': 64, 'steps': 40, 'f': 0.035, 'k': 0.065}, 'weight': 0.7},
            {'method': cellular_automata, 'params': {'width': 64, 'height': 64, 'steps': 10}, 'weight': 0.3}
        ]
    }
}

# --- 4. Demonstration ---

if __name__ == '__main__':
    from train import q_sample, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod
    
    # --- Config ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    IMG_PATH = 'images/image_00800.png'
    LATENT_DIM = 32
    BATCH_SIZE = 16
    TIMESTEP = 500 # Example timestep
    
    # --- Load and process image ---
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])
    image = Image.open(IMG_PATH).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(DEVICE) # Add batch dimension
    
    # In a real scenario, you'd get latents from the VAE encoder
    # Here, we'll just use a random tensor as a placeholder for x_start latents
    x_start_latents = torch.randn(BATCH_SIZE, LATENT_DIM).to(DEVICE)

    # --- Generate and apply noise ---
    mixer = ProcGenMixer(NOISE_CONFIGS)
    
    print("Generating and applying different noise patterns...")
    
    for name, config in NOISE_CONFIGS.items():
        print(f"  - {name}")
        
        # Generate procedural noise with the target shape of the latents
        proc_noise = mixer.generate_noise(name, target_shape=x_start_latents.shape)
        
        # Create noisy latents using the forward process function from train.py
        t = torch.full((BATCH_SIZE,), TIMESTEP, dtype=torch.long).to(DEVICE)
        noisy_latents = q_sample(x_start_latents, t, noise=proc_noise.to(DEVICE))
        
        # For demonstration, we can't decode the latents without the VAE.
        # So, we'll visualize the noise itself applied to the original image.
        
        # Resize noise to image dimensions for visualization
        noise_for_viz = proc_noise[0].reshape(1, 1, 4, 8).repeat(1, 3, 16, 8) # Just an example reshape
        noise_for_viz = torch.nn.functional.interpolate(noise_for_viz, size=(64, 64), mode='bilinear', align_corners=False)
        noise_for_viz = (noise_for_viz - noise_for_viz.min()) / (noise_for_viz.max() - noise_for_viz.min())

        # Add noise to image
        noisy_image = torch.clamp(image_tensor + (noise_for_viz.to(DEVICE) * 0.5), 0, 1)
        
        # Save
        save_path = f"output/noisy_demo_{name}.png"
        save_image(noisy_image, save_path)

    print("\nDemonstration complete. Check the 'output' directory for noisy images.")