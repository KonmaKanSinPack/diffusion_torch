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
from tqdm import tqdm
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

transform = transforms.Compose([
   transforms.ToTensor(),
   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                       download=False, transform=transform)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=0)
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                      download=False, transform=transform)
testloader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=0)

model = UNet(t_emb_dim=128, ch=64, out_ch=3)
model = model.to(device)
# ema_updater = EmaUpdater(
#       model, deepcopy(model), decay=0.995, start_iter=20_000
#    )
opt_d = torch.optim.AdamW(model.parameters(), lr=1e-4,weight_decay=1e-4)
do_pad = True
loss_func = torch.nn.MSELoss()
K_TRUNCATE = 16  # 截断K (修改为16,因为CIFAR10 reshape后最多32个奇异值)

"""
Super resolution recon:

I_H = I_L + I_res
    = U @ S_L @ Vt + U @ B
Where: B = S_res @ Vt

During Training: 
Truncate S_H to get S_L, K_TRUNCATE is set to 32

During Evaluation & Testing:
Model predicts B
"""

for epoch in tqdm(range(50000), desc="Training Epochs"):
   for images, labels in tqdm(trainloader, desc=f"Epoch {epoch}", leave=False):
      images = images.to(device)
      b, c, h, w = images.shape
      if h != w and do_pad:
         if h > w:
            diff = h - w
            images = F.pad(images, (diff//2, diff - diff//2, 0, 0), "constant", 0)
         else:
            diff = w - h
            images = F.pad(images, (0, 0, diff//2, diff - diff//2), "constant", 0)
         b, c, h, w = images.shape
      
      # Reshape成[b,c*h,w]，使SVD有足够奇异值
      images_flat = torch.reshape(images, [b, c*h, w])  # [64,96,32]
      U, S, Vt = torch.linalg.svd(images_flat)  # U:[64,96,96], S:[64,32], Vt:[64,32,32]
      
      # 构造低秩近似（人工截断奇异值） I_L
      k = min(K_TRUNCATE, S.shape[1])
      S_truncated = torch.zeros_like(S)
      S_truncated[:, :k] = S[:, :k]
      S_L_diag = torch.diag_embed(S_truncated)
      I_L_flat = U[:, :, :S.shape[1]] @ S_L_diag @ Vt
      I_L = torch.reshape(I_L_flat, [b, c, h, w])
      
      # 真实的 S_res
      S_res = S.clone()
      S_res[:, :k] = 0  # 只保留后面被截断的奇异值
      S_res_diag = torch.diag_embed(S_res)
      
      # 真实的 b = S_res @ Vt
      b_true_flat = S_res_diag @ Vt  # [b, 32, 32]
      
      # 模型输入低秩图像,预测 b
      t = torch.ones(b, device=device).long()
      b_pred = model(I_L, t)  # [b, c, h, w]
      b_pred_flat = torch.reshape(b_pred, [b, c*h, w])  # [b, 96, 32]
      
      # 投影到SVD空间: U^T @ b_pred
      b_pred_svd = torch.transpose(U, 1, 2)[:, :S.shape[1], :] @ b_pred_flat  # [64,32,32]
      
      # 损失: 预测的b vs 真实的b
      loss = loss_func(b_pred_svd, b_true_flat)
      
      opt_d.zero_grad()
      loss.backward()
      grad_clip(model.parameters(), mode="norm", value=0.003)
      opt_d.step()

   # 打印训练进度
   if epoch % 100 == 0:
      print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

   if epoch % 1000 == 0:
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
            
            # === 测试阶段 ===
            img_flat = torch.reshape(img, [b, c*h, w])
            U, S, Vt = torch.linalg.svd(img_flat)
            
            # 构造低秩输入 I_L
            k = min(K_TRUNCATE, S.shape[1])
            S_truncated = torch.zeros_like(S)
            S_truncated[:, :k] = S[:, :k]
            S_L_diag = torch.diag_embed(S_truncated)
            I_L_flat = U[:, :, :S.shape[1]] @ S_L_diag @ Vt
            I_L = torch.reshape(I_L_flat, [b, c, h, w])
            
            # 预测 b
            t = torch.ones(b, device=device).long()
            b_pred = model(I_L, t)
            b_pred_flat = torch.reshape(b_pred, [b, c*h, w])
            
            # 重建: I_H = I_L + U @ (U^T @ b_pred)
            b_pred_svd = torch.transpose(U, 1, 2)[:, :S.shape[1], :] @ b_pred_flat
            reconstructed_flat = I_L_flat + U[:, :, :S.shape[1]] @ b_pred_svd
            reconstructed = torch.reshape(reconstructed_flat, [b, c, h, w])
            
            # 可视化: 原图 | 低秩图像 | 重建结果
            pad_img = F.pad(img, (2, 2, 0, 0), 'constant', 1)
            pad_low = F.pad(I_L, (2, 2, 0, 0), 'constant', 1)
            pad_recon = F.pad(reconstructed, (2, 2, 0, 0), 'constant', 1)
            
            sav_img = torch.cat([pad_img, pad_low, pad_recon], dim=3)
            save_image(sav_img, f"result_{epoch}.jpg", normalize=True)
            break
      model.train()