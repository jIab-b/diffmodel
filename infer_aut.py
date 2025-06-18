import os
import torch
from torchvision.utils import save_image
from train import VAE, LatentDiffusionNet, Decoder
from train_aut import NoiseRecommender
from automata import ProcGenMixer, NOISE_CONFIGS
import math

# --- Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = "output"
VAE_MODEL_PATH = os.path.join(OUTPUT_DIR, "vae.pth")
DIFFUSION_MODEL_PATH = os.path.join(OUTPUT_DIR, "diffusion_model.pth")
RECOMMENDER_MODEL_PATH = os.path.join(OUTPUT_DIR, "noise_recommender.pth")
LATENT_DIM = 32
TIMESTEPS = 1000
NUM_IMAGES = 4

def p_sample_loop(model, shape, timesteps, device):
    """The reverse diffusion process."""
    img = torch.randn(shape, device=device)
    for i in reversed(range(timesteps)):
        t = torch.full((shape[0],), i, device=device, dtype=torch.long)
        img = p_sample(model, img, t, i)
    return img

def p_sample(model, x, t, t_index):
    with torch.no_grad():
        betas = torch.linspace(1e-4, 0.02, TIMESTEPS, device=DEVICE)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.nn.functional.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        sqrt_recip_alphas_t = sqrt_recip_alphas[t].view(-1, 1)
        sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod[t].view(-1, 1)
        
        model_mean = sqrt_recip_alphas_t * (x - betas[t].view(-1, 1) * model(x, t) / sqrt_one_minus_alphas_cumprod_t)

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = posterior_variance[t].view(-1, 1)
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise

if __name__ == "__main__":
    # Load all three models
    vae = VAE().to(DEVICE)
    vae.load_state_dict(torch.load(VAE_MODEL_PATH, map_location=DEVICE))
    vae.eval()
    print(f"Loaded VAE from {VAE_MODEL_PATH}")

    diffusion_model = LatentDiffusionNet(latent_dim=LATENT_DIM).to(DEVICE)
    diffusion_model.load_state_dict(torch.load(DIFFUSION_MODEL_PATH, map_location=DEVICE))
    diffusion_model.eval()
    print(f"Loaded Diffusion model from {DIFFUSION_MODEL_PATH}")

    recommender = NoiseRecommender().to(DEVICE)
    if os.path.exists(RECOMMENDER_MODEL_PATH):
        recommender.load_state_dict(torch.load(RECOMMENDER_MODEL_PATH, map_location=DEVICE))
        recommender.eval()
        print(f"Loaded Noise Recommender from {RECOMMENDER_MODEL_PATH}")
    else:
        print("Noise Recommender model not found. Using random noise.")
        recommender = None

    # --- Generation ---
    print(f"Generating {NUM_IMAGES} image(s)...")
    
    # Start with random latents
    initial_latents = torch.randn(NUM_IMAGES, LATENT_DIM, device=DEVICE)
    
    if recommender:
        # Use the recommender to get the noise config
        # As in train_aut, this is a simplified stand-in for a real parameter mapping
        noise_mixer = ProcGenMixer(NOISE_CONFIGS)
        initial_noise = noise_mixer.generate_noise('ca_plus_perlin', initial_latents.shape).to(DEVICE)
    else:
        initial_noise = torch.randn_like(initial_latents)

    # The reverse diffusion process starts from the generated procedural noise
    generated_latents = p_sample_loop(diffusion_model, initial_noise.shape, TIMESTEPS, DEVICE)

    # Decode the generated latents into images
    with torch.no_grad():
        generated_images = vae.decoder(generated_latents)

    save_path = os.path.join(OUTPUT_DIR, "generated_sample_aut.png")
    save_image(generated_images, save_path, nrow=int(math.sqrt(NUM_IMAGES)))
    print(f"Generated images saved to {save_path}")
    print("Inference script finished.")