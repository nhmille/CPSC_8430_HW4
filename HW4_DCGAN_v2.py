import torch
from torch.utils.data import DataLoader
import torchvision.datasets as dset
import torchvision.transforms as transforms

from HW4_DCGAN_Supp import Generator
from HW4_DCGAN_Supp import Discriminator
from HW4_DCGAN_Supp import train
from HW4_DCGAN_Supp import evaluateFID

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    G = Generator()
    D = Discriminator()
    
    noise = torch.randn(10, 50, device=device)
    labels = torch.arange(0, 10, device=device)
    
    # Convert label integers to one-hot vectors
    cond = torch.nn.functional.one_hot(labels, num_classes=10).float().to(device)
    
    dataset = dset.CIFAR10(root="./data", download=True,
                               transform=transforms.Compose([
                                   transforms.Resize((64,64)),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),]))
    
    training_data = DataLoader(dataset, batch_size=128, shuffle=True)
    
    train(G, D, training_data, device, epochs=50, noise_dim=50, cond_dim=10)
        
    # Evaluate trained model
    evaluateFID()
    