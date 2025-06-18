import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

from train import VAE, LatentDiffusionNet, q_sample, CustomImageDataset
from automata import ProcGenMixer, NOISE_CONFIGS

# --- Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_DIR = "images"
OUTPUT_DIR = "output"
VAE_MODEL_PATH = os.path.join(OUTPUT_DIR, "vae.pth")
DIFFUSION_MODEL_PATH = os.path.join(OUTPUT_DIR, "diffusion_model.pth")
RECOMMENDER_MODEL_PATH = os.path.join(OUTPUT_DIR, "noise_recommender.pth")
LR_META = 1e-5
EPOCHS_META = 50
BATCH_SIZE = 16
LATENT_DIM = 32
TIMESTEPS = 1000

# --- Noise Recommender Network ---
class NoiseRecommender(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM, output_param_dim=10):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_param_dim),
            nn.Sigmoid()
        )

    def forward(self, z):
        return self.model(z)

def train_recommender(vae_model, diffusion_model):
    print("Training Noise Recommender...")
    recommender = NoiseRecommender().to(DEVICE)
    optimizer = optim.Adam(recommender.parameters(), lr=LR_META)
    
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])
    dataset = CustomImageDataset(img_dir=IMAGE_DIR, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)

    vae_model.eval()
    diffusion_model.eval()

    for epoch in range(EPOCHS_META):
        for i, images in enumerate(dataloader):
            images = images.to(DEVICE)
            optimizer.zero_grad()

            with torch.no_grad():
                mu, log_var = vae_model.encoder(images)
                latents = vae_model.reparameterize(mu, log_var)

            # This is a simplified example. A real implementation would need a more
            # sophisticated way to map the recommender output to the mixer config.
            # For now, we'll just use a fixed config and pretend the recommender is choosing it.
            noise_mixer = ProcGenMixer(NOISE_CONFIGS)
            noise = noise_mixer.generate_noise('ca_plus_perlin', latents.shape).to(DEVICE)
            
            t = torch.randint(0, TIMESTEPS, (images.size(0),), device=DEVICE).long()
            noisy_latents = q_sample(latents, t, noise)
            
            predicted_noise = diffusion_model(noisy_latents, t)
            
            loss = nn.functional.mse_loss(predicted_noise, noise)
            loss.backward()
            optimizer.step()

            if (i + 1) % 50 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS_META}], Step [{i+1}/{len(dataloader)}], Recommender Loss: {loss.item():.4f}")

    torch.save(recommender.state_dict(), RECOMMENDER_MODEL_PATH)
    print(f"Noise Recommender training complete. Model saved to {RECOMMENDER_MODEL_PATH}")
    return recommender

if __name__ == "__main__":
    # Load pre-trained VAE
    vae = VAE().to(DEVICE)
    vae.load_state_dict(torch.load(VAE_MODEL_PATH, map_location=DEVICE))
    print(f"Loaded VAE from {VAE_MODEL_PATH}")

    # Load pre-trained Diffusion Model
    diffusion_model = LatentDiffusionNet(latent_dim=LATENT_DIM).to(DEVICE)
    diffusion_model.load_state_dict(torch.load(DIFFUSION_MODEL_PATH, map_location=DEVICE))
    print(f"Loaded Diffusion model from {DIFFUSION_MODEL_PATH}")

    train_recommender(vae, diffusion_model)
    print("Meta-training script finished.")