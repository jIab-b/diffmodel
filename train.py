import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image

# --- Configuration ---
IMG_SIZE = 64
LATENT_DIM = 32 # Latent dimension for VAE
CHANNELS = 3
BATCH_SIZE = 16 # Reduce if OOM
LR = 1e-4
EPOCHS_VAE = 25 # Epochs for VAE pre-training
EPOCHS_DIFFUSION = 100 # Epochs for Diffusion model training
TIMESTEPS = 1000 # Number of diffusion timesteps
BETA_START = 1e-4
BETA_END = 0.02
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_DIR = "images"
OUTPUT_DIR = "output"
VAE_MODEL_PATH = os.path.join(OUTPUT_DIR, "vae.pth")
DIFFUSION_MODEL_PATH = os.path.join(OUTPUT_DIR, "diffusion_model.pth")

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- VAE Model ---
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

def vae_loss_function(recon_x, x, mu, log_var):
    BCE = nn.functional.binary_cross_entropy(recon_x.view(-1, IMG_SIZE * IMG_SIZE * CHANNELS),
                                             x.view(-1, IMG_SIZE * IMG_SIZE * CHANNELS), reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KLD

# --- Diffusion Model (Simple U-Net like structure for latents) ---
# For simplicity, this U-Net will operate on flattened latents + time embedding
# A proper U-Net for image latents would use Conv layers.
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

        # Simplified network for latent vector processing
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

# --- Diffusion Utilities ---
betas = torch.linspace(BETA_START, BETA_END, TIMESTEPS, device=DEVICE)
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = nn.functional.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

def q_sample(x_start, t, noise=None): # Forward process: q(x_t | x_0)
    if noise is None:
        noise = torch.randn_like(x_start)
    sqrt_alphas_cumprod_t = sqrt_alphas_cumprod[t].view(-1, 1) # For latents, ensure broadcasting
    sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod[t].view(-1, 1)
    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

# --- Dataset ---
class CustomImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.img_names = [f for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f))]
        self.transform = transform

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        try:
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a dummy tensor or skip
            return torch.zeros((CHANNELS, IMG_SIZE, IMG_SIZE))


transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # VAE uses Sigmoid, so [0,1] is fine
])

dataset = CustomImageDataset(img_dir=IMAGE_DIR, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)


# --- Training VAE ---
def train_vae():
    print("Training VAE...")
    vae = VAE().to(DEVICE)
    optimizer_vae = optim.Adam(vae.parameters(), lr=LR)

    if os.path.exists(VAE_MODEL_PATH):
        print(f"Loading pre-trained VAE from {VAE_MODEL_PATH}")
        vae.load_state_dict(torch.load(VAE_MODEL_PATH, map_location=DEVICE))
        return vae

    for epoch in range(EPOCHS_VAE):
        for i, images in enumerate(dataloader):
            images = images.to(DEVICE)
            recon_images, mu, log_var = vae(images)
            loss = vae_loss_function(recon_images, images, mu, log_var)

            optimizer_vae.zero_grad()
            loss.backward()
            optimizer_vae.step()

            if (i + 1) % 50 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS_VAE}], Step [{i+1}/{len(dataloader)}], VAE Loss: {loss.item()/images.size(0):.4f}")
        
        # Save a sample reconstruction
        with torch.no_grad():
            sample_input = next(iter(dataloader)).to(DEVICE)
            recon_sample, _, _ = vae(sample_input)
            comparison = torch.cat([sample_input[:4], recon_sample[:4]])
            save_image(comparison.cpu(), os.path.join(OUTPUT_DIR, f'vae_reconstruction_epoch_{epoch+1}.png'), nrow=4)

    torch.save(vae.state_dict(), VAE_MODEL_PATH)
    print(f"VAE training complete. Model saved to {VAE_MODEL_PATH}")
    return vae

# --- Training Diffusion Model ---
def train_diffusion(vae_model):
    print("Training Diffusion Model...")
    diffusion_model = LatentDiffusionNet(latent_dim=LATENT_DIM).to(DEVICE)
    optimizer_diffusion = optim.Adam(diffusion_model.parameters(), lr=LR)
    print('device used', DEVICE)

    if os.path.exists(DIFFUSION_MODEL_PATH):
        print(f"Loading pre-trained Diffusion model from {DIFFUSION_MODEL_PATH}")
        diffusion_model.load_state_dict(torch.load(DIFFUSION_MODEL_PATH, map_location=DEVICE))
        
        # return diffusion_model # If you want to skip training if model exists

    vae_model.eval() # VAE is used for encoding, not training here

    for epoch in range(EPOCHS_DIFFUSION):
        for i, images in enumerate(dataloader):
            images = images.to(DEVICE)
            optimizer_diffusion.zero_grad()

            with torch.no_grad():
                mu, log_var = vae_model.encoder(images)
                latents = vae_model.reparameterize(mu, log_var) # x_0 (original latents)

            # Sample timesteps
            t = torch.randint(0, TIMESTEPS, (images.size(0),), device=DEVICE).long()
            
            # Sample noise and create noisy latents (x_t)
            noise = torch.randn_like(latents)
            noisy_latents = q_sample(latents, t, noise) # x_t

            # Predict noise
            predicted_noise = diffusion_model(noisy_latents, t)
            
            loss = nn.functional.mse_loss(predicted_noise, noise)
            loss.backward()
            optimizer_diffusion.step()

            #if (i + 1) % 100 == 0:
                #print(f"Epoch [{epoch+1}/{EPOCHS_DIFFUSION}], Step [{i+1}/{len(dataloader)}], Diffusion Loss: {loss.item():.4f}")
        print(f"epcoh {epoch+1} done")
        
    torch.save(diffusion_model.state_dict(), DIFFUSION_MODEL_PATH)
    print(f"Diffusion model training complete. Model saved to {DIFFUSION_MODEL_PATH}")
    return diffusion_model

if __name__ == "__main__":
    # 1. Train or load VAE
    vae = train_vae()
    
    # 2. Train or load Diffusion Model
    # Ensure VAE is loaded if not trained in this session
    if not 'vae' in locals() or vae is None:
        vae = VAE().to(DEVICE)
        if os.path.exists(VAE_MODEL_PATH):
            vae.load_state_dict(torch.load(VAE_MODEL_PATH, map_location=DEVICE))
            print(f"Loaded VAE from {VAE_MODEL_PATH} for diffusion training.")
        else:
            print("VAE model not found. Please train VAE first or ensure VAE_MODEL_PATH is correct.")
            exit()
            
    diffusion_model = train_diffusion(vae)

    print("Training script finished. To generate images, run inference.py.")
