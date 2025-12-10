import torch

import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader

from models.unet import UNet

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device}")

transform = transforms.Compose([
   transforms.ToTensor(),
   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                       download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=0)
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                      download=True, transform=transform)
testloader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=0)

model = UNet(t_emb_dim=128, ch=64, out_ch=3)
model = model.to(device)

for images, labels in trainloader:
    images = images.to(device)

    t = torch.randint(0, 100, (images.shape[0],), device=device).long()
    output = model(images, t)

    assert images.shape == output.shape, "shape not equal"
    
    print("succeed")
   #  breakpoint()