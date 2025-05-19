import os
import math
import torch
import torch.nn as nn
from torchvision.utils import save_image
from PIL import Image
from torchvision import transforms # Added for image loading and transformations

# --- Configuration (Copied from train.py, only relevant parts) ---
IMG_SIZE = 64 # VAE's native processing and output resolution
LATENT_DIM = 32
CHANNELS = 3 # Needed for VAE Decoder output shape
TIMESTEPS = 1000
BETA_START = 1e-4
BETA_END = 0.02
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = "output"
VAE_MODEL_PATH = os.path.join(OUTPUT_DIR, "vae.pth")
DIFFUSION_MODEL_PATH = os.path.join(OUTPUT_DIR, "diffusion_model.pth")

# Create output directory if it doesn't exist (good practice for standalone script)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- VAE Model (Copied from train.py) ---
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(CHANNELS, 32, kernel_size=4, stride=2, padding=1), # 64x64 -> 32x32
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), # 32x32 -> 16x16
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # 16x16 -> 8x8
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 2 * LATENT_DIM) # mu and log_var
        )

    def forward(self, x):
        x = self.model(x)
        mu, log_var = torch.chunk(x, 2, dim=1)
        return mu, log_var

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(LATENT_DIM, 128 * 8 * 8),
            nn.Unflatten(1, (128, 8, 8)),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), # 8x8 -> 16x16
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1), # 16x16 -> 32x32
            nn.ReLU(),
            nn.ConvTranspose2d(32, CHANNELS, kernel_size=4, stride=2, padding=1), # 32x32 -> 64x64
            nn.Sigmoid() # Output pixel values between 0 and 1
        )

    def forward(self, z):
        return self.model(z)

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        return self.decoder(z), mu, log_var

# --- Diffusion Model (Copied from train.py) ---
class LatentDiffusionNet(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM, time_emb_dim=128):
        super().__init__()
        self.latent_dim = latent_dim
        self.time_emb_dim = time_emb_dim

        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 2),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 2, time_emb_dim)
        )

        self.main_net = nn.Sequential(
            nn.Linear(latent_dim + time_emb_dim, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Linear(512, latent_dim) # Predicts noise
        )

    def _time_embedding(self, t):
        half_dim = self.time_emb_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=DEVICE) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=1)
        return self.time_mlp(emb)

    def forward(self, x_t, t):
        time_emb = self._time_embedding(t.float())
        x = torch.cat([x_t, time_emb], dim=1)
        return self.main_net(x)

# --- Diffusion Utilities (Copied from train.py) ---
betas = torch.linspace(BETA_START, BETA_END, TIMESTEPS, device=DEVICE)
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
# These are not directly used by the DDPM sampling logic in `sample` but kept for completeness if other sampling methods were added.
# alphas_cumprod_prev = nn.functional.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
# sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
# sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
# sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
# posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)


# --- Inference/Sampling (Moved from train.py) ---
@torch.no_grad()
def sample(diffusion_model, vae_decoder, num_images=1, img_size_latent=LATENT_DIM, target_display_size=None):
    print(f"\nGenerating {num_images} image(s)...")
    diffusion_model.eval()
    vae_decoder.eval()

    latents_t = torch.randn((num_images, img_size_latent), device=DEVICE)

    for i in reversed(range(TIMESTEPS)):
        t = torch.full((num_images,), i, device=DEVICE, dtype=torch.long)
        predicted_noise = diffusion_model(latents_t, t)
        
        alpha_t = alphas[t].view(-1, 1)
        alpha_cumprod_t = alphas_cumprod[t].view(-1, 1)
        beta_t = betas[t].view(-1,1)
        
        noise_z = torch.randn_like(latents_t) if i > 0 else torch.zeros_like(latents_t)

        term1 = (1.0 / torch.sqrt(alpha_t))
        term2_factor = (1.0 - alpha_t) / torch.sqrt(1.0 - alpha_cumprod_t)
        term2 = (latents_t - term2_factor * predicted_noise)
        latents_t = term1 * term2 + torch.sqrt(beta_t) * noise_z
        
    generated_images_vae_output = vae_decoder(latents_t) # Output at IMG_SIZE
    generated_images_cpu = generated_images_vae_output.cpu()

    images_to_save_list = []
    final_pil_images = []

    for i in range(num_images):
        img_tensor_cpu = generated_images_cpu[i]
        pil_image = transforms.ToPILImage()(img_tensor_cpu)

        if target_display_size and target_display_size != IMG_SIZE:
            pil_image = pil_image.resize((target_display_size, target_display_size), Image.LANCZOS)
        
        final_pil_images.append(pil_image)
        images_to_save_list.append(transforms.ToTensor()(pil_image)) # Convert back to tensor for save_image

    images_to_save_tensor = torch.stack(images_to_save_list)
    
    size_str = f"_resized_{target_display_size}" if target_display_size and target_display_size != IMG_SIZE else ""
    filename = f'generated_sample_steps_{TIMESTEPS}{size_str}.png'
    save_path = os.path.join(OUTPUT_DIR, filename)
    
    save_image(images_to_save_tensor, save_path, nrow=int(math.sqrt(num_images)))
    print(f"Generated images saved to {save_path}")
    
    return final_pil_images if num_images == 1 else final_pil_images # Return list of PIL images




if __name__ == "__main__":
    print(f"Using device: {DEVICE}")
    TARGET_DISPLAY_RESOLUTION = 256 # Define desired output resolution for display/saving

    # 1. Load VAE
    vae = VAE().to(DEVICE)
    if os.path.exists(VAE_MODEL_PATH):
        vae.load_state_dict(torch.load(VAE_MODEL_PATH, map_location=DEVICE))
        # vae.eval() will be called inside functions using it
        print(f"Loaded VAE from {VAE_MODEL_PATH}")
    else:
        print(f"VAE model not found at {VAE_MODEL_PATH}. Cannot proceed with inference.")
        exit()

    # 2. Load Diffusion Model
    diffusion_model = LatentDiffusionNet(latent_dim=LATENT_DIM).to(DEVICE)
    if os.path.exists(DIFFUSION_MODEL_PATH):
        diffusion_model.load_state_dict(torch.load(DIFFUSION_MODEL_PATH, map_location=DEVICE))
        # diffusion_model.eval() will be called inside functions using it
        print(f"Loaded Diffusion model from {DIFFUSION_MODEL_PATH}")
    else:
        print(f"Diffusion model not found at {DIFFUSION_MODEL_PATH}. Cannot proceed with generation.")
        # We might still want to do reconstruction if VAE is loaded
        # For now, let's exit if diffusion model is needed for generation part
        # exit() # Commenting out to allow reconstruction even if diffusion model is missing

    # 3. Generate images using Diffusion Model and VAE Decoder
    if os.path.exists(DIFFUSION_MODEL_PATH): # Only generate if diffusion model loaded
        generated_pil_images = sample(diffusion_model, vae.decoder, num_images=1, target_display_size=TARGET_DISPLAY_RESOLUTION)
    else:
        print("Skipping image generation as diffusion model was not found.")
            


    print("\nInference script finished.")
