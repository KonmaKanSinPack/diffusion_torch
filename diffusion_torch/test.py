import torch

import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader

import classifier_metrics_numpy
from models.unet import UNet
# from optim_utils import EmaUpdater
import torch.nn.functional as F
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device}")

def grad_clip(params, mode: str = "value", value: float = None, **kwargs) -> None:
    """do a gradient clipping

    Args:
        params (tensor): model params
        mode (str, optional): 'value' or 'norm'. Defaults to 'value'.
    """
    assert mode in ["value", "norm"], "mode should be @value or @norm"
    if mode == "norm":
        nn.utils.clip_grad.clip_grad_norm_(parameters=params, max_norm=value, **kwargs)
    else:  # mode == 'value'
        nn.utils.clip_grad.clip_grad_value_(parameters=params, clip_value=value)

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
# ema_updater = EmaUpdater(
#       model, deepcopy(model), decay=0.995, start_iter=20_000
#    )
opt_d = torch.optim.AdamW(model.parameters(), lr=1e-4,weight_decay=1e-4)
do_pad = True
loss_func = torch.nn.MSELoss()
for epoch in range(50_000):
   for images, labels in trainloader:
      images = images.to(device)
      b, c, h, w = images.shape
      if h != w and do_pad:
         print("do padding for this image")
         if h > w:
            diff = h - w
            images = F.pad(images, (diff//2, diff - diff//2, 0, 0), "constant", 0)
         else:
            diff = w - h
            images = F.pad(images, (0, 0, diff//2, diff - diff//2), "constant", 0)
         b, c, h, w = images.shape
         print(f"padded image shape: {images.shape}")
      
      
      images = torch.reshape(images,[b,c,-1])
      U,S,Vt = torch.linalg.svd(images)
      S = torch.diag_embed(S)
      U_inv = torch.linalg.pinv(U)
      UH = U_inv@images
      k = S.shape[1]
      res = S@Vt[:,:k,:]-UH

      t = torch.ones(images.shape[0], device=device).long()
      images = torch.reshape(images,[b,c,h,w])
      res = torch.reshape(res,[b,c,h,w])
      output = model(images, t)

      loss = loss_func(output, res)
      opt_d.zero_grad()
      loss.backward()
      grad_clip(model.parameters(), mode="norm", value=0.003)
      opt_d.step()
      # ema_updater.update(epoch)

      if epoch % 10 == 0:
         # Compute FID and Inception score
         model.eval()
         with torch.no_grad():
            for img, _ in testloader:
               img = img.to(device)
               b, c, h, w = img.shape
               if h != w and do_pad:
                  print("do padding for this image")
                  if h > w:
                     diff = h - w
                     img = F.pad(img, (diff//2, diff - diff//2, 0, 0), "constant", 0)
                  else:
                     diff = w - h
                     img = F.pad(img, (0, 0, diff//2, diff - diff//2), "constant", 0)
                  b, c, h, w = img.shape
                  print(f"padded image shape: {img.shape}")
               
               t = torch.ones(images.shape[0], device=device).long()
               
               res = model(img,t)
               img = torch.reshape(img,[b,c,-1])
               U, S, Yt = torch.linalg.svd(img)
               S = torch.diag_embed(S)
               k = S.shape[1]
               output = img - U@res
               
               print(f"max U@res: {max(U@res)}, min U@res: {min(U@res)}")

               img = torch.reshape(img,[b,c,h,w])
               output = torch.reshape(output,[b,c,h,w])
               # Inception score
               # metrics['{}/inception{}'.format(samples_key, self.num_inception_samples)] = float(
               # classifier_metrics_numpy.classifier_score_from_logits(inception_gen['logits']))

               # # FID vs training set
               # metrics['{}/trainfid{}'.format(samples_key, self.num_inception_samples)] = float(
               # classifier_metrics_numpy.frechet_classifier_distance_from_activations(
               #    cached_inception_real_train['pool_3'], inception_gen['pool_3']))

               # FID vs val set
               
               print(f"FID vs val set:{float(classifier_metrics_numpy.frechet_classifier_distance_from_activations(output, img))}")