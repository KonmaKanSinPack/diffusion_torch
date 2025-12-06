import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader

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

for images, labels in trainloader:
    breakpoint()