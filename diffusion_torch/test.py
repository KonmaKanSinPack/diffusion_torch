import torch

import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader

import classifier_metrics_numpy
from models.unet import UNet
# from optim_utils import EmaUpdater
import torch.nn.functional as F
import torch.nn as nn

import numpy as np
from numpy import cov, iscomplexobj, trace
from scipy.linalg import sqrtm
from torchvision.utils import save_image
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device}")

def calculate_fid(act1, act2):
   # 计算均值和协方差矩阵
   mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
   mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)

   # 计算均值差的平方和
   ssdiff = np.sum((mu1 - mu2) ** 2.0)

   # 计算协方差矩阵的平方根
   covmean = sqrtm(sigma1.dot(sigma2))

   # 检查并修正虚数部分
   if iscomplexobj(covmean):
      covmean = covmean.real

   # 计算FID分数
   fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
   return fid

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

def get_i_th_singular_vectors(x, i):
   U, S, Vt = torch.linalg.svd(x)
   S = torch.diag_embed(S)
   out = (U[:, :, :i] @ S[:, :i, :i] @ Vt[:, :i, :])
   
   return out
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
         # print("do padding for this image")
         if h > w:
            diff = h - w
            images = F.pad(images, (diff//2, diff - diff//2, 0, 0), "constant", 0)
         else:
            diff = w - h
            images = F.pad(images, (0, 0, diff//2, diff - diff//2), "constant", 0)
         b, c, h, w = images.shape
         # print(f"padded image shape: {images.shape}")
      
      images = torch.reshape(images,[b,c,-1])
      pred_img = torch.randn_like(images)
      for i in range(3):
         pred_img = torch.reshape(pred_img,[b,c,-1])
         out_i = get_i_th_singular_vectors(images, i+1)
         U, S, Vt = torch.linalg.svd(pred_img)
         S = torch.diag_embed(S)
         # U_in, _, _ = torch.linalg.svd(pred_img)
         U_inv = torch.linalg.pinv(U)
         # breakpoint()
         UH = U_inv@out_i
         k = S.shape[1]
         res = S@Vt[:,:k,:]-UH
         t = (i+1)*torch.ones(out_i.shape[0], device=device).long()
         pred_img = torch.reshape(pred_img,[b,c,h,w])
         res = torch.reshape(res,[b,c,h,w])
         output = model(pred_img, t)

         loss = loss_func(output, res)
         # loss = loss_func(pred_U, U)
         opt_d.zero_grad()
         loss.backward()
         grad_clip(model.parameters(), mode="norm", value=0.003)
         opt_d.step()

         pred_res = torch.reshape(output,[b,c,-1]).detach()
         pred_img = torch.reshape(pred_img,[b,c,-1]).detach()
         U_out, _, _ = torch.linalg.svd(pred_img)
         S = torch.diag_embed(S)
         k = S.shape[1]
         pred_img = pred_img - U_out@pred_res
         # print(f"max U@res: {torch.max(U@res)}, min U@res: {torch.min(U@res)}")

         pred_img = torch.reshape(pred_img,[b,c,h,w])#.cpu().numpy()
         output = torch.reshape(output,[b,c,h,w])#.cpu().numpy()
      # ema_updater.update(epoch)
   print(f"epoch {epoch} completed")

   if epoch % 10 == 0:   
      images = torch.reshape(images,[b,c,h,w])
      pad_img = F.pad(images,(2,2,0,0),'constant',1)
      
      sav_img = torch.cat([pad_img,pred_img],dim=3)
      save_image(sav_img,f"result_{epoch}.jpg")
