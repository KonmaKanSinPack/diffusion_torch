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
N_ITERATIONS = 3  # 牛顿迭代次数

"""
Super resolution recon with Newton-like Iterative Refinement:

I_H = I_L + I_res
    = U @ S_L @ Vt + U @ B
Where: B = S_res @ Vt

Iterative refinement (类似牛顿法):
I_0 = I_L (初始低秩近似)
I_{n+1} = I_n + U @ b_n (迭代修正)

During Training: 
在每次迭代中,模型预测当前状态到目标的修正项
时间步 t 条件化迭代步数

During Evaluation & Testing:
多步迭代refinement,逐步逼近高质量重建
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
      
      images_flat = torch.reshape(images, [b, c*h, w])
      
      # === 牛顿迭代式训练 ===
      current_img = images.clone()
      total_loss = 0
      
      for iter_step in range(N_ITERATIONS):
         current_flat = torch.reshape(current_img, [b, c*h, w])
         U, S, Vt = torch.linalg.svd(current_flat)
         
         # 计算当前残差: 目标 - 当前状态
         residual_flat = images_flat - current_flat
         
         # 真实的修正项: b_true = U^T @ residual
         b_true_flat = torch.transpose(U, 1, 2)[:, :S.shape[1], :] @ residual_flat
         
         # 模型预测修正项 (条件化在迭代步数上)
         t = torch.ones(b, device=device).long() * iter_step
         b_pred = model(current_img, t)
         b_pred_flat = torch.reshape(b_pred, [b, c*h, w])
         
         # 投影到SVD空间
         b_pred_svd = torch.transpose(U, 1, 2)[:, :S.shape[1], :] @ b_pred_flat
         
         # 累积损失
         loss = loss_func(b_pred_svd, b_true_flat)
         total_loss += loss
         
         # 更新当前图像(用于下一次迭代)
         with torch.no_grad():
            delta_flat = U[:, :, :S.shape[1]] @ b_pred_svd
            current_flat = current_flat + delta_flat
            current_img = torch.reshape(current_flat, [b, c, h, w])
      
      opt_d.zero_grad()
      total_loss.backward()
      grad_clip(model.parameters(), mode="norm", value=0.003)
      opt_d.step()

   # 打印训练进度
   if epoch % 100 == 0:
      print(f"Epoch {epoch}, Loss: {total_loss.item():.6f}")

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
            
            # === 测试阶段: 牛顿迭代refinement ===
            img_flat = torch.reshape(img, [b, c*h, w])
            U_orig, S_orig, Vt_orig = torch.linalg.svd(img_flat)
            
            # 初始化: 低秩近似
            k = min(K_TRUNCATE, S_orig.shape[1])
            S_truncated = torch.zeros_like(S_orig)
            S_truncated[:, :k] = S_orig[:, :k]
            S_L_diag = torch.diag_embed(S_truncated)
            current_flat = U_orig[:, :, :S_orig.shape[1]] @ S_L_diag @ Vt_orig
            current_img = torch.reshape(current_flat, [b, c, h, w])
            
            # 保存中间结果
            intermediate_imgs = [current_img.clone()]
            
            # 迭代refinement
            for iter_step in range(N_ITERATIONS):
               current_flat = torch.reshape(current_img, [b, c*h, w])
               U, S, Vt = torch.linalg.svd(current_flat)
               
               # 预测修正
               t = torch.ones(b, device=device).long() * iter_step
               b_pred = model(current_img, t)
               b_pred_flat = torch.reshape(b_pred, [b, c*h, w])
               
               # 投影并应用修正
               b_pred_svd = torch.transpose(U, 1, 2)[:, :S.shape[1], :] @ b_pred_flat
               delta_flat = U[:, :, :S.shape[1]] @ b_pred_svd
               current_flat = current_flat + delta_flat
               current_img = torch.reshape(current_flat, [b, c, h, w])
               
               intermediate_imgs.append(current_img.clone())
            
            # 可视化: 原图 | 低秩初始 | 迭代1 | 迭代2 | ... | 最终
            viz_imgs = [F.pad(img, (2, 2, 0, 0), 'constant', 1)]
            for inter_img in intermediate_imgs:
               viz_imgs.append(F.pad(inter_img, (2, 2, 0, 0), 'constant', 1))
            
            sav_img = torch.cat(viz_imgs, dim=3)
            save_image(sav_img, f"recon_iter_result_{epoch}.jpg", normalize=True)
            break
      model.train()