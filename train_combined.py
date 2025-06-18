import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from train import VAE, LatentDiffusionNet, q_sample, CustomImageDataset
from automata import ProcGenMixer, cellular_automata, perlin_noise, reaction_diffusion

# --- Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_DIR = "images"
OUTPUT_DIR = "output"
VAE_MODEL_PATH = os.path.join(OUTPUT_DIR, "vae.pth")
NEW_DIFFUSION_PATH = os.path.join(OUTPUT_DIR, "diffusion_comp.pth")
NEW_RECOMMENDER_PATH = os.path.join(OUTPUT_DIR, "noise_comp.pth")

# --- Training Hyperparameters ---
LR_DIFFUSION = 1e-4  # Higher LR for the "critic"
LR_RECOMMENDER = 2e-6 # Lower LR for the "generator" (TTUR)
EPOCHS = 75
BATCH_SIZE = 16
LATENT_DIM = 32
TIMESTEPS = 1000
NOISE_PENALTY_LAMBDA = 0.01 # Weight for the noise energy penalty

# --- Stochastic Noise Recommender ---
# Outputs a distribution over noise parameters
class StochasticNoiseRecommender(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM, param_dim=6):
        super().__init__()
        self.latent_dim = latent_dim
        self.param_dim = param_dim
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 2 * param_dim) # mu and log_var for each parameter
        )

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, z):
        params = self.model(z)
        mu, log_var = torch.chunk(params, 2, dim=1)
        # We sample from the distribution and then apply a sigmoid
        # to keep the base parameters in a predictable [0, 1] range.
        sampled_params = self.reparameterize(mu, log_var)
        return torch.sigmoid(sampled_params), mu, log_var

# Helper to map normalized params [0,1] to meaningful ranges
def map_params_to_config(params_tensor):
    # params_tensor is a batch of [ca_steps, ca_weight, perlin_octaves, perlin_weight, rd_steps, rd_weight]
    configs = []
    for params in params_tensor:
        # Scale parameters from [0,1] to reasonable values
        ca_s = int(5 + params[0] * 45)    # steps: 5-50
        ca_w = params[1]                  # weight: 0-1
        p_oct = int(2 + params[2] * 6)    # octaves: 2-8
        p_w = params[3]                   # weight: 0-1
        rd_s = int(20 + params[4] * 80)   # steps: 20-100
        rd_w = params[5]                  # weight: 0-1

        # Normalize weights to sum to 1
        total_w = ca_w + p_w + rd_w + 1e-8
        
        config = {
            'mix': [
                {'method': cellular_automata, 'params': {'width': 64, 'height': 64, 'steps': ca_s}, 'weight': ca_w / total_w},
                {'method': perlin_noise, 'params': {'width': 64, 'height': 64, 'octaves': p_oct}, 'weight': p_w / total_w},
                {'method': reaction_diffusion, 'params': {'width': 64, 'height': 64, 'steps': rd_s}, 'weight': rd_w / total_w}
            ]
        }
        configs.append(config)
    return configs


def train_combined(vae_model):
    print("Starting combined training...")
    
    # 1. Instantiate models
    diffusion_model = LatentDiffusionNet(latent_dim=LATENT_DIM).to(DEVICE)
    recommender = StochasticNoiseRecommender().to(DEVICE)

    # 2. Instantiate two optimizers (TTUR)
    optimizer_diffusion = optim.Adam(diffusion_model.parameters(), lr=LR_DIFFUSION)
    optimizer_recommender = optim.Adam(recommender.parameters(), lr=LR_RECOMMENDER)

    # --- Dataloader ---
    transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
    dataset = CustomImageDataset(img_dir=IMAGE_DIR, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)

    vae_model.eval() # VAE is frozen, only used for encoding

    for epoch in range(EPOCHS):
        for i, images in enumerate(dataloader):
            images = images.to(DEVICE)
            
            # Zero out both optimizers
            optimizer_diffusion.zero_grad()
            optimizer_recommender.zero_grad()

            # 3. Forward Pass through the combined graph
            with torch.no_grad():
                mu, log_var = vae_model.encoder(images)
                latents = vae_model.reparameterize(mu, log_var)

            # Get noise parameter distributions and sample them
            param_dist, _, _ = recommender(latents)
            
            # Generate a batch of noise tensors
            batch_configs = map_params_to_config(param_dist)
            procedural_noise = torch.zeros_like(latents)
            for j, config in enumerate(batch_configs):
                # Note: This loop is slow. A vectorized mixer would be much faster.
                mixer = ProcGenMixer(config)
                procedural_noise[j] = mixer.generate_noise('mix', latents[j].shape, device=DEVICE)
            
            # Add noise to latents
            t = torch.randint(0, TIMESTEPS, (images.size(0),), device=DEVICE).long()
            noisy_latents = q_sample(latents, t, noise=procedural_noise)
            
            # Predict noise with the diffusion model
            predicted_noise = diffusion_model(noisy_latents, t)
            
            # 4. Calculate Loss
            mse_loss = nn.functional.mse_loss(predicted_noise, procedural_noise)
            
            # Add noise energy penalty to prevent collapse to zero
            noise_penalty = torch.mean(procedural_noise.pow(2))
            
            # Total loss
            loss = mse_loss + NOISE_PENALTY_LAMBDA * noise_penalty

            # 5. Backpropagate through both networks
            loss.backward()
            
            # 6. Step both optimizers
            optimizer_diffusion.step()
            optimizer_recommender.step()

            if (i + 1) % 2 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}, MSE: {mse_loss.item():.4f}")

    # 7. Save the new models
    torch.save(diffusion_model.state_dict(), NEW_DIFFUSION_PATH)
    torch.save(recommender.state_dict(), NEW_RECOMMENDER_PATH)
    print(f"Combined training complete.")
    print(f"Diffusion model saved to {NEW_DIFFUSION_PATH}")
    print(f"Recommender model saved to {NEW_RECOMMENDER_PATH}")


if __name__ == "__main__":
    # Load the pre-trained VAE, which is essential and not trained here
    if not os.path.exists(VAE_MODEL_PATH):
        print(f"Error: VAE model not found at {VAE_MODEL_PATH}")
        print("Please run train.py first to train and save the VAE.")
        exit()
        
    vae = VAE().to(DEVICE)
    vae.load_state_dict(torch.load(VAE_MODEL_PATH, map_location=DEVICE))
    print(f"Loaded pre-trained VAE from {VAE_MODEL_PATH}")

    train_combined(vae)