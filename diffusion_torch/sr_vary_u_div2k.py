import torch
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
from pathlib import Path
import requests
from tqdm import tqdm
import zipfile

from models.unet import UNet
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
        nn.utils.clip_grad.clip_grad_norm_(
            parameters=params, max_norm=value, **kwargs)
    else:  # mode == 'value'
        nn.utils.clip_grad.clip_grad_value_(
            parameters=params, clip_value=value)


def download_file(url, save_path):
    """下载文件并显示进度条"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(save_path, 'wb') as file, tqdm(
        desc=save_path.name,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            pbar.update(size)


def download_div2k(data_root='./data/DIV2K'):
    """下载DIV2K数据集"""
    data_root = Path(data_root)
    data_root.mkdir(parents=True, exist_ok=True)
    
    # DIV2K训练集和验证集的URL
    urls = {
        'train_hr': 'http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip',
        'valid_hr': 'http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip',
    }
    
    for name, url in urls.items():
        zip_path = data_root / f'{name}.zip'
        extract_path = data_root / name.replace('_hr', '')
        
        # 检查是否已经下载并解压
        if extract_path.exists() and any(extract_path.iterdir()):
            print(f"{name} already exists, skipping download.")
            continue
        
        # 下载
        if not zip_path.exists():
            print(f"Downloading {name}...")
            try:
                download_file(url, zip_path)
            except Exception as e:
                print(f"Error downloading {name}: {e}")
                print("Please download manually from: https://data.vision.ee.ethz.ch/cvl/DIV2K/")
                continue
        
        # 解压
        print(f"Extracting {name}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_root)
        
        print(f"{name} ready!")


class DIV2KDataset(Dataset):
    """DIV2K数据集加载器"""
    def __init__(self, root_dir, split='train', patch_size=128, transform=None):
        """
        Args:
            root_dir: DIV2K数据集根目录
            split: 'train' or 'valid'
            patch_size: 训练时裁剪的patch大小
            transform: 数据增强变换
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.patch_size = patch_size
        self.transform = transform
        
        # 构建图像路径列表
        if split == 'train':
            self.img_dir = self.root_dir / 'DIV2K_train_HR'
        else:
            self.img_dir = self.root_dir / 'DIV2K_valid_HR'
        
        if not self.img_dir.exists():
            raise RuntimeError(f"Dataset not found at {self.img_dir}. Please run download first.")
        
        self.image_files = sorted(list(self.img_dir.glob('*.png')))
        print(f"Found {len(self.image_files)} images in {split} set")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        
        # 随机裁剪patch
        if self.split == 'train':
            w, h = image.size
            if w >= self.patch_size and h >= self.patch_size:
                x = np.random.randint(0, w - self.patch_size + 1)
                y = np.random.randint(0, h - self.patch_size + 1)
                image = image.crop((x, y, x + self.patch_size, y + self.patch_size))
            else:
                # 如果图像太小，resize到patch_size
                image = image.resize((self.patch_size, self.patch_size), Image.BICUBIC)
        else:
            # 验证集：resize到固定大小或裁剪中心
            w, h = image.size
            if w > self.patch_size and h > self.patch_size:
                x = (w - self.patch_size) // 2
                y = (h - self.patch_size) // 2
                image = image.crop((x, y, x + self.patch_size, y + self.patch_size))
            else:
                image = image.resize((self.patch_size, self.patch_size), Image.BICUBIC)
        
        if self.transform:
            image = self.transform(image)
        
        return image, 0  # 返回0作为标签以保持接口一致


# === 数据集准备 ===
print("Preparing DIV2K dataset...")
data_root = './data/DIV2K'
download_div2k(data_root)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

PATCH_SIZE = 128  # DIV2K图像较大，使用128x128的patch进行训练
BATCH_SIZE = 16   # DIV2K图像质量高，batch_size可以小一些

trainset = DIV2KDataset(root_dir=data_root, split='train', patch_size=PATCH_SIZE, transform=transform)
trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

validset = DIV2KDataset(root_dir=data_root, split='valid', patch_size=PATCH_SIZE, transform=transform)
validloader = DataLoader(validset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

model = UNet(t_emb_dim=128, ch=64, out_ch=3)
model = model.to(device)

opt_d = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
do_pad = True
loss_func = torch.nn.MSELoss()

# === 超分辨率参数 ===
SCALE_FACTOR = 4  # 超分倍数 (4x: 32×32 -> 128×128)
N_ITERATIONS = 3  # 迭代次数

"""
Super Resolution with SVD-based Newton-like Iterative Refinement on DIV2K:

核心创新: 在SVD分解空间进行迭代修正

I_H (高分辨率) = 原始128×128图像
I_L (低分辨率) = 32×32下采样后bicubic上采样回128×128

SVD分解:
I_H_flat = U @ S_H @ Vt (目标)
I_L_flat = U @ S_L @ Vt (初始状态)

迭代修正公式:
I_{n+1} = I_n + U @ b_n

其中 b_n 由模型预测，在SVD子空间进行约束
"""

print("=" * 60)
print("Super Resolution Training on DIV2K Dataset")
print(f"Patch Size: {PATCH_SIZE}x{PATCH_SIZE}")
print(f"Scale Factor: {SCALE_FACTOR}x")
print(f"Iterations: {N_ITERATIONS}")
print(f"Batch Size: {BATCH_SIZE}")
print(f"Device: {device}")
print("=" * 60)

num_epochs = 200

epoch_pbar = tqdm(range(num_epochs), desc="Training Epochs")
for epoch in epoch_pbar:
    batch_pbar = tqdm(trainloader, desc=f"Epoch {epoch}", leave=False)
    for images, _ in batch_pbar:
        images = images.to(device)  # I_H: 高分辨率目标
        b, c, h, w = images.shape

        # 处理非正方形图像
        if h != w and do_pad:
            if h > w:
                diff = h - w
                images = F.pad(
                    images, (diff//2, diff - diff//2, 0, 0), "constant", 0)
            else:
                diff = w - h
                images = F.pad(
                    images, (0, 0, diff//2, diff - diff//2), "constant", 0)
            b, c, h, w = images.shape

        # === 生成低分辨率I_L (模拟超分任务) ===
        # 1. 下采样到低分辨率
        img_lr = F.interpolate(
            images, scale_factor=1/SCALE_FACTOR, mode='bicubic', align_corners=False)
        # 2. 上采样回原尺寸 (模糊的初始估计)
        img_lr_upscaled = F.interpolate(img_lr, size=(
            h, w), mode='bicubic', align_corners=False)

        # 展平用于SVD
        images_flat = torch.reshape(images, [b, c*h, w])

        # === SVD-based 迭代式训练 ===
        current_img = img_lr_upscaled.clone()  # 从低分辨率上采样开始
        total_loss = 0

        for iter_step in range(N_ITERATIONS):
            current_flat = torch.reshape(current_img, [b, c*h, w])

            # SVD分解当前状态
            U, S, Vt = torch.linalg.svd(current_flat)

            # 计算残差: 目标(高分辨率) - 当前状态
            residual_flat = images_flat - current_flat

            # 真实的修正项: b_true = U^T @ residual
            # 这是在SVD子空间中的表示
            b_true_flat = torch.transpose(
                U, 1, 2)[:, :S.shape[1], :] @ residual_flat

            # 模型预测修正项 (条件化在迭代步数上)
            t = torch.ones(b, device=device).long() * iter_step
            b_pred = model(current_img, t)
            b_pred_flat = torch.reshape(b_pred, [b, c*h, w])

            # 投影到SVD空间 (核心创新: 在SVD子空间约束)
            b_pred_svd = torch.transpose(
                U, 1, 2)[:, :S.shape[1], :] @ b_pred_flat

            # 计算损失
            loss = loss_func(b_pred_svd, b_true_flat)
            total_loss += loss

            # 更新当前图像 (用于下一次迭代)
            with torch.no_grad():
                delta_flat = U[:, :, :S.shape[1]] @ b_pred_svd
                current_flat = current_flat + delta_flat
                current_img = torch.reshape(current_flat, [b, c, h, w])

        # 反向传播
        opt_d.zero_grad()
        total_loss.backward()
        grad_clip(model.parameters(), mode="norm", value=0.003)
        opt_d.step()

        # 更新batch进度条的loss信息
        batch_pbar.set_postfix({"loss": f"{total_loss.item():.6f}"})

    # 更新epoch进度条的loss信息
    epoch_pbar.set_postfix({"loss": f"{total_loss.item():.6f}"})

    # 定期评估和可视化
    if epoch % 10 == 0:
        model.eval()
        with torch.no_grad():
            for img, _ in validloader:
                img = img.to(device)  # I_H: 高分辨率真实图像
                b, c, h, w = img.shape

                # 处理非正方形图像
                if h != w and do_pad:
                    print("do padding for this image")
                    if h > w:
                        diff = h - w
                        img = F.pad(
                            img, (diff//2, diff - diff//2, 0, 0), "constant", 0)
                    else:
                        diff = w - h
                        img = F.pad(
                            img, (0, 0, diff//2, diff - diff//2), "constant", 0)
                    b, c, h, w = img.shape
                    print(f"padded image shape: {img.shape}")

                # === 测试阶段: SVD-based 超分辨率迭代refinement ===
                # 1. 生成低分辨率初始化
                img_lr = F.interpolate(
                    img, scale_factor=1/SCALE_FACTOR, mode='bicubic', align_corners=False)
                current_img = F.interpolate(img_lr, size=(
                    h, w), mode='bicubic', align_corners=False)

                # 保存中间结果
                # LR原图 + Bicubic上采样
                intermediate_imgs = [img_lr, current_img.clone()]

                # 迭代refinement
                for iter_step in range(N_ITERATIONS):
                    current_flat = torch.reshape(current_img, [b, c*h, w])

                    # SVD分解
                    U, S, Vt = torch.linalg.svd(current_flat)

                    # 预测修正
                    t = torch.ones(b, device=device).long() * iter_step
                    b_pred = model(current_img, t)
                    b_pred_flat = torch.reshape(b_pred, [b, c*h, w])

                    # 投影到SVD空间并应用修正
                    b_pred_svd = torch.transpose(
                        U, 1, 2)[:, :S.shape[1], :] @ b_pred_flat
                    delta_flat = U[:, :, :S.shape[1]] @ b_pred_svd
                    current_flat = current_flat + delta_flat
                    current_img = torch.reshape(current_flat, [b, c, h, w])

                    intermediate_imgs.append(current_img.clone())

                # 可视化: 原图(HR) | LR | Bicubic | SVD-Iter1 | SVD-Iter2 | ... | 最终SR
                viz_imgs = [F.pad(img, (2, 2, 0, 0), 'constant', 1)]  # 高分辨率真值
                for inter_img in intermediate_imgs:
                    # 对低分辨率图像进行上采样用于可视化
                    if inter_img.shape[-1] != w:
                        inter_img = F.interpolate(
                            inter_img, size=(h, w), mode='nearest')
                    viz_imgs.append(
                        F.pad(inter_img, (2, 2, 0, 0), 'constant', 1))

                sav_img = torch.cat(viz_imgs, dim=3)
                save_image(sav_img, f"sr_div2k_result_{epoch}.jpg", normalize=True)

                # 计算PSNR
                mse = F.mse_loss(current_img, img)
                psnr = 10 * torch.log10(4.0 / mse)
                
                # Bicubic baseline
                bicubic_img = F.interpolate(img_lr, size=(h, w), mode='bicubic', align_corners=False)
                mse_bicubic = F.mse_loss(bicubic_img, img)
                psnr_bicubic = 10 * torch.log10(4.0 / mse_bicubic)

                print(f"\nEpoch {epoch} - DIV2K Evaluation:")
                print(f"  - Image size: {img.shape}")
                print(f"  - PSNR: {psnr.item():.2f} dB")
                print(f"  - Bicubic PSNR: {psnr_bicubic.item():.2f} dB")
                print(f"  - Improvement: {(psnr - psnr_bicubic).item():.2f} dB\n")
                break
        
        model.train()
        
        # 保存模型
        if epoch % 50 == 0 and epoch > 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt_d.state_dict(),
                'loss': total_loss.item(),
            }, f'checkpoint_div2k_epoch_{epoch}.pth')
            print(f"Model saved at epoch {epoch}")

print("\n" + "=" * 60)
print("Training completed!")
print("=" * 60)
