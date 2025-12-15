import torch
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torch.nn.functional as F
import torch.nn as nn
from torchvision.utils import save_image
from tqdm import tqdm
import numpy as np
from numpy import cov, iscomplexobj, trace
from scipy.linalg import sqrtm

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device}")

def calculate_fid(act1, act2):
   mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
   mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
   ssdiff = np.sum((mu1 - mu2) ** 2.0)
   covmean = sqrtm(sigma1.dot(sigma2))
   if iscomplexobj(covmean):
      covmean = covmean.real
   fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
   return fid

def grad_clip(params, mode: str = "value", value: float = None, **kwargs) -> None:
    assert mode in ["value", "norm"], "mode should be @value or @norm"
    if mode == "norm":
        nn.utils.clip_grad.clip_grad_norm_(parameters=params, max_norm=value, **kwargs)
    else:
        nn.utils.clip_grad.clip_grad_value_(parameters=params, clip_value=value)

# ===== 下载ImageNet-100 (如果不存在) =====
import os
if not os.path.exists('./data/imagenet100'):
    print("正在下载ImageNet-100 (约13GB)...")
    os.makedirs('./data', exist_ok=True)
    
    import kagglehub

    # Download latest version
    path = kagglehub.dataset_download("ambityga/imagenet100")

    print("Path to dataset files:", path)

# ===== 数据加载 =====
transform_train = transforms.Compose([
   transforms.Resize(256),
   transforms.CenterCrop(256),
   transforms.RandomHorizontalFlip(),
   transforms.ToTensor(),
   transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

transform_test = transforms.Compose([
   transforms.Resize(256),
   transforms.CenterCrop(256),
   transforms.ToTensor(),
   transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# 使用ImageFolder直接加载
trainset = ImageFolder(root='./data/imagenet100/train', transform=transform_train)
trainloader = DataLoader(trainset, batch_size=32, shuffle=True, num_workers=8, pin_memory=True)

testset = ImageFolder(root='./data/imagenet100/val', transform=transform_test)
testloader = DataLoader(testset, batch_size=32, shuffle=False, num_workers=8, pin_memory=True)

print(f"Training samples: {len(trainset)}")
print(f"Validation samples: {len(testset)}")
print(f"Number of classes: {len(trainset.classes)}")

from models.unet import UNet

model = UNet(t_emb_dim=128, ch=64, out_ch=3)
model = model.to(device)
opt_d = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
do_pad = False  # 256x256是正方形,不需要padding
loss_func = torch.nn.MSELoss()

# 256x256图像: reshape为[b, c*h, w] = [b, 768, 256]
# SVD后: U[b,768,768], S[b,256], Vt[b,256,256]
K_TRUNCATE = 64  # 截断到64个奇异值 (可调整为32/128)

"""
Super resolution recon on ImageNet-1k (256x256):

I_H = I_L + I_res
    = U @ S_L @ Vt + U @ B
Where: B = S_res @ Vt

During Training: 
Truncate S_H to get S_L, K_TRUNCATE is set to 64

During Evaluation & Testing:
Model predicts B
"""

print("Starting training on ImageNet-1k...")

for epoch in tqdm(range(50000), desc="Training Epochs"):
   for images, labels in tqdm(trainloader, desc=f"Epoch {epoch}", leave=False):
      images = images.to(device)
      b, c, h, w = images.shape  # [32, 3, 256, 256]
      
      # Reshape成[b,c*h,w]，使SVD有足够奇异值
      images_flat = torch.reshape(images, [b, c*h, w])  # [32, 768, 256]
      U, S, Vt = torch.linalg.svd(images_flat)  # U:[32,768,768], S:[32,256], Vt:[32,256,256]
      
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
      b_true_flat = S_res_diag @ Vt  # [32, 256, 256]
      
      # 模型输入低秩图像,预测 b
      t = torch.ones(b, device=device).long()
      b_pred = model(I_L, t)  # [32, 3, 256, 256]
      b_pred_flat = torch.reshape(b_pred, [b, c*h, w])  # [32, 768, 256]
      
      # 投影到SVD空间: U^T @ b_pred
      b_pred_svd = torch.transpose(U, 1, 2)[:, :S.shape[1], :] @ b_pred_flat  # [32,256,256]
      
      # 损失: 预测的b vs 真实的b
      loss = loss_func(b_pred_svd, b_true_flat)
      
      opt_d.zero_grad()
      loss.backward()
      grad_clip(model.parameters(), mode="norm", value=0.003)
      opt_d.step()

   # 打印训练进度
   if epoch % 100 == 0:
      print(f"\nEpoch {epoch}, Loss: {loss.item():.6f}")

   if epoch % 1000 == 0:
      print(f"\n=== Epoch {epoch}: Evaluation ===")
      model.eval()
      with torch.no_grad():
         for img, _ in testloader:
            img = img.to(device)
            b, c, h, w = img.shape
            
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
            
            # 反归一化以便可视化
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
            img_vis = img * std + mean
            I_L_vis = I_L * std + mean
            recon_vis = reconstructed * std + mean
            
            # 可视化: 原图 | 低秩图像 | 重建结果
            pad_img = F.pad(img_vis, (4, 4, 0, 0), 'constant', 1)
            pad_low = F.pad(I_L_vis, (4, 4, 0, 0), 'constant', 1)
            pad_recon = F.pad(recon_vis, (4, 4, 0, 0), 'constant', 1)
            
            sav_img = torch.cat([pad_img, pad_low, pad_recon], dim=3)
            save_image(sav_img, f"imagenet_recon_{epoch}.jpg")
            print(f"Saved imagenet_recon_{epoch}.jpg")
            break
      model.train()
      
      # 保存检查点
      if epoch % 5000 == 0:
         torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': opt_d.state_dict(),
            'loss': loss.item(),
         }, f"checkpoint_epoch_{epoch}.pth")
         print(f"Checkpoint saved: checkpoint_epoch_{epoch}.pth")

print("Training complete!")