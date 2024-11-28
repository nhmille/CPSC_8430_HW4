import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision import datasets, transforms
from torchvision.utils import save_image
from pytorch_fid import fid_score

class Generator(nn.Module):
    def __init__(self, noise_dim=50, cond_dim=10, filt=64):
        super(Generator, self).__init__()
        
        # Layer to create embedding for condition (cifar-10 class)
        # Condition vector has a dimension of 256
        self.cond_embedding = nn.Sequential(nn.Linear(cond_dim, 256), 
                                            nn.ReLU())
        
        # Full connected layer for combining noise and condition
        # self.fc = nn.Sequential(nn.Linear(noise_dim + 256, 3*filt*filt), nn.ReLU())
        self.fc = nn.Sequential(nn.Linear(noise_dim + 256, 4*8*8*filt), nn.ReLU())

        
        # Increase spatial dimensions and reduce channels
        # Starting dimensions are (4x4x512)
        self.convUp = nn.Sequential(            
            # (8x8x256) -> (16x16x128)
            nn.ConvTranspose2d(4*filt, 2*filt, kernel_size=6, stride=2, padding=2),
            nn.BatchNorm2d(2*filt),
            nn.ReLU(),
            
            # (16x16x128) -> (32x32x64)
            nn.ConvTranspose2d(2*filt, filt, kernel_size=6, stride=2, padding=2),
            nn.BatchNorm2d(filt),
            nn.ReLU(),
            
            # (32x32x64) -> (64x64x3)
            nn.ConvTranspose2d(filt, 3, kernel_size=6, stride=2, padding=2),
            
            # Use Tanh to scale pixels to [-1,1]
            nn.Tanh())


    def forward(self, noise, cond_input):
        # Embed condition
        cond_embedding = self.cond_embedding(cond_input)
        
        # Combine noise and condition -> Is this the correct way of adding noise?
        combined = torch.cat((noise, cond_embedding), dim=1)
        x = self.fc(combined)
        
        # Resize into (batch_size, feature_map_size (512), 4, 4)
        x = x.view(-1, 256, 4, 4)
        
        # Pass through convolutional layers
        x = self.convUp(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, cond_dim=10, filt=64):
        super(Discriminator, self).__init__()
        
        # Embed condition to 256 vector
        self.cond_embedding = nn.Sequential(nn.Linear(cond_dim, 256), 
                                            nn.ReLU())
        
        # Reduce spatial dimensions while increasing depth/channels
        self.convDown = nn.Sequential(
            # (64x64x3) -> (32x32x64)
            nn.Conv2d(3, filt, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2),
            
            # (32x32x64) -> (16x16x128)
            nn.Conv2d(filt, 2*filt, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(2*filt),
            nn.LeakyReLU(0.2),
            
            # (16x16x128) -> (8x8x256)
            nn.Conv2d(2*filt, 4*filt, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(4*filt),
            nn.LeakyReLU(0.2))
        
        # Combine image features and condition to make a single prediction of real/fake -> Raw # for WGAN
        self.combine = nn.Sequential(
            nn.Conv2d(256 + 256, 512, kernel_size=1),
            nn.Flatten(),
            nn.Linear(256*8*8, 1))

    def forward(self, image, cond_input):
        # Embed condition to 256
        cond_embedding = self.cond_embedding(cond_input)
        
        cond_embedding = cond_embedding.view(-1, 256, 1, 1)
        
        # Tile embedding to match spatial dimensions
        cond_embedding = cond_embedding.repeat(1, 1, 8, 8)
        
        # Extract features
        image_features = self.convDown(image)
        
        # Combine features and condition
        combined = torch.cat((image_features, cond_embedding), dim=1)
        output = self.combine(combined)
    
        return output



def train(Gen, Disc, training_data, device, epochs=1, noise_dim=50, cond_dim=10):
    Gen.to(device)
    Disc.to(device)

    D_opt = optim.RMSprop(Disc.parameters(), lr=0.0001)
    G_opt = optim.RMSprop(Gen.parameters(), lr=0.0002)

    # Binary real/fake -> Real is 0.9 for input smoothing of discriminator
    real = 0.9
    fake = 0.0
    
    G_losses = []
    D_losses = []
    
    D_loss = torch.tensor(0.0, requires_grad=True, device=device)
    G_loss = torch.tensor(0.0, requires_grad=True, device=device)
    
    for epoch in range(epochs):
        D_epoch_loss = 0.0
        G_epoch_loss = 0.0

        for i, (real_images, labels) in enumerate(tqdm(training_data, desc=f"Epoch {epoch + 1}/{epochs}")):
            real_images = real_images.to(device)
            labels = labels.to(device)
            batch_size = real_images.size(0)

            # One-hot embedding for 10 conditions in cifar-10
            cond = torch.nn.functional.one_hot(labels, num_classes=cond_dim).float().to(device)

            # Train Discriminator
            if i % 1 == 0:
                Disc.zero_grad()

                # First, train on real images
                real_images = real_images + 0.05*torch.randn_like(real_images)
                # real_labels = torch.full((batch_size, 1), real, device=device)
                output_real = Disc(real_images, cond)

                # Then, train on fake images from generator
                noise = torch.randn(batch_size, noise_dim, device=device)
                fake_images = Gen(noise, cond)
                fake_images = fake_images + 0.05*torch.randn_like(fake_images)
                
                output_fake = Disc(fake_images.detach(), cond)
                
                # Wasserstein loss with grad penatly
                gp = gradient_penalty(Disc, real_images, fake_images, cond)
                D_loss = -(torch.mean(output_real) - torch.mean(output_fake)) + 10*gp
                
                D_loss.backward()
                D_opt.step()

                D_epoch_loss += D_loss.item()
                
                # Gradient clipping if gradient penalty does not work
                for p in Disc.parameters():
                    p.data.clamp_(-0.01, 0.01)

            # Train Generator
            if i % 5 == 0:
                Gen.zero_grad()
                noise = torch.randn(batch_size, noise_dim, device=device)
                # noise = torch.full((batch_size, noise_dim), 0.0, device=device)
                
                # Perturb condition
                # cond += 0.1*torch.randn_like(cond)
                # cond = cond.clamp(0, 1)
                
                fake_images = Gen(noise, cond)
                output_fake = Disc(fake_images, cond)

                G_loss = -torch.mean(output_fake)
                G_loss.backward()
                G_opt.step()

                G_epoch_loss += G_loss.item()
        
        G_losses.append(G_epoch_loss/len(training_data))
        D_losses.append(D_epoch_loss/len(training_data))
        save_best_images(Gen, device)


        # Log average losses at the end of each epoch
        tqdm.write(
            f"Epoch [{epoch + 1}/{epochs}],  D Loss: {D_epoch_loss/len(training_data):.4f}, G Loss: {G_epoch_loss/len(training_data):.4f}")

    torch.save(Gen.state_dict(), "./models/wgan_generator.pth")
    torch.save(Disc.state_dict(), "./models/wgan_discriminator.pth")
    plot_losses(G_losses, D_losses, save_path="loss_plot.png")
    save_best_images(Gen, device)
    save_generated_images(Gen, device)
    save_real_images()
    
def gradient_penalty(critic, real_data, fake_data, cond):
    alpha = torch.rand(real_data.size(0), 1, 1, 1, device=real_data.device)
    interpolates = alpha * real_data + (1 - alpha) * fake_data
    interpolates.requires_grad_(True)
    
    output = critic(interpolates, cond)
    gradients = torch.autograd.grad(
        outputs=output,
        inputs=interpolates,
        grad_outputs=torch.ones_like(output, device=real_data.device),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_norm = gradients.norm(2, dim=1)
    return ((gradient_norm - 1) ** 2).mean()

    
    
    
def plot_losses(G_losses, D_losses, save_path=None):
    plt.figure(figsize=(10, 5))
    plt.plot(G_losses, label="Generator Loss")
    plt.plot(D_losses, label="Discriminator Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Losses")
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def save_best_images(Gen, device, noise_dim=50, cond_dim=10, epoch=None):
    # Generate noise and corresponding class labels
    noise = torch.randn(10, noise_dim, device=device)
    labels = torch.arange(0, 10, device=device)
    
    # Convert labels to one-hot vectors
    cond = torch.nn.functional.one_hot(labels, num_classes=cond_dim).float().to(device)
    cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                       'dog', 'frog', 'horse', 'ship', 'truck']
    
    # Generate images
    Gen.eval()
    with torch.no_grad():
        fake_images = Gen(noise, cond).cpu()
    Gen.train()
    
    # Rescale images from [-1, 1] to [0, 1]
    fake_images = (fake_images + 1) / 2

    # Plot and save images
    fig, axes = plt.subplots(1, 10, figsize=(20, 4))
    for i in range(10):
        axes[i].imshow(fake_images[i].permute(1, 2, 0).numpy())
        axes[i].axis('off')
        axes[i].set_title(f"{cifar10_classes[i]}")
    
    plt.tight_layout()
    if epoch is not None:
        plt.savefig(f"generated_images/best_images_epoch_{epoch}.png")
    else:
        plt.savefig("generated_images/best_images.png")
    plt.close()


def save_generated_images(Gen, device, noise_dim=50, cond_dim=10, output_folder="generated_images_fid", num_images=100):
    os.makedirs(output_folder, exist_ok=True)

    Gen.eval()
    with torch.no_grad():
        for i in range(num_images):
            # Generate a random class label
            label = torch.randint(0, cond_dim, (1,), device=device)
            
            # Generate noise and condition vector
            noise = torch.randn(1, noise_dim, device=device)
            cond = torch.nn.functional.one_hot(label, num_classes=cond_dim).float()

            # Generate image
            fake_image = Gen(noise, cond).cpu()
            
            # Rescale image from [-1, 1] to [0, 1]
            fake_image = (fake_image + 1) / 2
            
            # Save image
            class_name = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                          'dog', 'frog', 'horse', 'ship', 'truck'][label.item()]
            image_path = os.path.join(output_folder, f"{class_name}_{i}.png")
            save_image(fake_image, image_path)
    
    Gen.train()
    print(f"Saved {num_images} generated images to {output_folder}")

def save_real_images(data_dir="./real_images", num_images=100):
    os.makedirs(data_dir, exist_ok=True)

    # Transformations to normalize CIFAR-10 images to [0, 1]
    transform = transforms.Compose([
        transforms.ToTensor()  # Convert PIL image to tensor in [0, 1]
    ])

    # Load CIFAR-10 training dataset
    cifar10_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)

    # Use a DataLoader to access images
    dataloader = torch.utils.data.DataLoader(cifar10_dataset, batch_size=1, shuffle=True)

    # CIFAR-10 class names
    cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                       'dog', 'frog', 'horse', 'ship', 'truck']

    for i, (image, label) in enumerate(dataloader):
        if i >= num_images:
            break
        
        # Get class name
        class_name = cifar10_classes[label.item()]
        
        # Save image
        image_path = os.path.join(data_dir, f"{class_name}_{i}.png")
        save_image(image, image_path)

    print(f"Saved {num_images} real images to {data_dir}")


def evaluateFID(real_images_path="./real_images", generated_images_path="./generated_images_fid"):
    # Compute FID using PyTorch library tool
    fid_value = fid_score.calculate_fid_given_paths(
        [real_images_path, generated_images_path], batch_size=50, device='cuda', dims=2048)
    print(f"FID Score: {fid_value}")




