import os
import torch
from torchvision.utils import save_image
import math

from train import VAE, LatentDiffusionNet
from train_combined import StochasticNoiseRecommender, map_params_to_config
from automata import ProcGenMixer
from infer_aut import p_sample_loop # Re-use the sampler from the previous inference script

# --- Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = "output"
VAE_MODEL_PATH = os.path.join(OUTPUT_DIR, "vae.pth")
DIFFUSION_MODEL_PATH = os.path.join(OUTPUT_DIR, "diffusion_comp.pth")
RECOMMENDER_MODEL_PATH = os.path.join(OUTPUT_DIR, "noise_comp.pth")
LATENT_DIM = 32
TIMESTEPS = 1000
NUM_IMAGES = 4

if __name__ == "__main__":
    # --- Load Models ---
    # 1. VAE
    vae = VAE().to(DEVICE)
    vae.load_state_dict(torch.load(VAE_MODEL_PATH, map_location=DEVICE))
    vae.eval()
    print(f"Loaded VAE from {VAE_MODEL_PATH}")

    # 2. Combined Diffusion Model
    diffusion_model = LatentDiffusionNet(latent_dim=LATENT_DIM).to(DEVICE)
    diffusion_model.load_state_dict(torch.load(DIFFUSION_MODEL_PATH, map_location=DEVICE))
    diffusion_model.eval()
    print(f"Loaded Combined Diffusion model from {DIFFUSION_MODEL_PATH}")

    # 3. Combined Noise Recommender
    recommender = StochasticNoiseRecommender().to(DEVICE)
    recommender.load_state_dict(torch.load(RECOMMENDER_MODEL_PATH, map_location=DEVICE))
    recommender.eval()
    print(f"Loaded Combined Noise Recommender from {RECOMMENDER_MODEL_PATH}")

    # --- Generation ---
    print(f"Generating {NUM_IMAGES} image(s) using the combined model...")
    
    # Start with random latents
    initial_latents = torch.randn(NUM_IMAGES, LATENT_DIM, device=DEVICE)
    
    # Get noise parameters from the recommender
    with torch.no_grad():
        # For inference, we might want to use the mean of the distribution for more stable outputs
        param_dist, mu, _ = recommender(initial_latents)
        # Using mu directly instead of reparameterizing gives a deterministic output for a given z
        params = torch.sigmoid(mu) 

    # Generate the initial procedural noise tensor `x_T`
    batch_configs = map_params_to_config(params)
    initial_noise = torch.zeros_like(initial_latents)
    for i, config in enumerate(batch_configs):
        mixer = ProcGenMixer(config)
        initial_noise[i] = mixer.generate_noise('mix', initial_latents[i].shape, device=DEVICE)

    # Run the reverse diffusion process starting from the procedural noise
    generated_latents = p_sample_loop(diffusion_model, initial_noise.shape, TIMESTEPS, DEVICE)

    # Decode the latents into images
    with torch.no_grad():
        generated_images = vae.decoder(generated_latents)

    save_path = os.path.join(OUTPUT_DIR, "generated_sample_combined.png")
    save_image(generated_images, save_path, nrow=int(math.sqrt(NUM_IMAGES)))
    print(f"Generated images saved to {save_path}")
    print("Combined inference script finished.")