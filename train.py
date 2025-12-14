import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import glob
import random
from tqdm import tqdm
from models.networks import UnetGenerator, NLayerDiscriminator
from models.losses import GANLoss, VGGLoss, OCRPerceptualLoss

# --- 配置 ---
BATCH_SIZE = 4
LR = 0.0002
EPOCHS = 100  
WARMUP_EPOCHS = 20 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "datasets/patches"

def get_noisy_input(x, sigma):
    noise = torch.randn_like(x) * sigma
    return x + noise

class DeblurDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.A_paths = sorted(glob.glob(os.path.join(root_dir, 'train_A', '*.png')))
        self.B_paths = sorted(glob.glob(os.path.join(root_dir, 'train_B', '*.png')))
        self.base_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def __len__(self):
        return len(self.A_paths)

    def __getitem__(self, idx):
        img_A = Image.open(self.A_paths[idx]).convert('L')
        img_B = Image.open(self.B_paths[idx]).convert('L')
        
        angle = random.uniform(-10, 10)
        img_A = img_A.rotate(angle)
        img_B = img_B.rotate(angle)
        
        if random.random() < 0.5:
            img_A = img_A.transpose(Image.FLIP_LEFT_RIGHT)
            img_B = img_B.transpose(Image.FLIP_LEFT_RIGHT)
            
        return self.base_transform(img_A), self.base_transform(img_B)

def train():
    print(f"当前配置: 恒定噪声(0.05) | 判别器降采样(0.5x) | D-Lazy(1:3)")
    
    netG = UnetGenerator(input_nc=1, output_nc=1, num_downs=8, ngf=64).to(DEVICE)
    netD = NLayerDiscriminator(input_nc=2).to(DEVICE)
    
    # 权重配置
    base_lambda_adv = 0.05 
    lambda_pixel = 100.0
    lambda_perceptual = 10.0
    lambda_ocr = 50.0
    
    criterionGAN = GANLoss().to(DEVICE)
    criterionL1 = torch.nn.L1Loss().to(DEVICE)
    criterionVGG = VGGLoss().to(DEVICE)
    criterionOCR = OCRPerceptualLoss().to(DEVICE)
    
    optimizer_G = torch.optim.Adam(netG.parameters(), lr=LR, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(netD.parameters(), lr=1e-5, betas=(0.5, 0.999))
    
    dataset = DeblurDataset(DATA_DIR)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    
    CONSTANT_SIGMA = 0.05 
    
    step_counter = 0

    for epoch in range(EPOCHS):
        is_warmup = epoch < WARMUP_EPOCHS
        curr_lambda_adv = 0.0 if is_warmup else base_lambda_adv
        
        loop = tqdm(dataloader, leave=True)
        for i, (real_A, real_B) in enumerate(loop):
            real_A = real_A.to(DEVICE)
            real_B = real_B.to(DEVICE)
            step_counter += 1
            
            # --- 1. 训练生成器 ---
            optimizer_G.zero_grad()
            fake_B = netG(real_A)
            
            # GAN Loss
            if not is_warmup:
                real_A_small = F.interpolate(real_A, scale_factor=0.5, mode='bilinear', align_corners=False)
                fake_B_small = F.interpolate(fake_B, scale_factor=0.5, mode='bilinear', align_corners=False)
                
                noisy_real_A = get_noisy_input(real_A_small, CONSTANT_SIGMA)
                noisy_fake_B = get_noisy_input(fake_B_small, CONSTANT_SIGMA)
                
                pred_fake = netD(torch.cat((noisy_real_A, noisy_fake_B), 1))
                loss_G_GAN = criterionGAN(pred_fake, True) * curr_lambda_adv
            else:
                loss_G_GAN = torch.tensor(0.0).to(DEVICE)
            
            loss_G_L1 = criterionL1(fake_B, real_B) * lambda_pixel
            
            fake_B_3c = fake_B.repeat(1, 3, 1, 1)
            real_B_3c = real_B.repeat(1, 3, 1, 1)
            loss_G_VGG = criterionVGG(fake_B_3c, real_B_3c) * lambda_perceptual
            loss_G_OCR = criterionOCR(fake_B_3c, real_B_3c) * lambda_ocr
            
            loss_G = loss_G_GAN + loss_G_L1 + loss_G_VGG + loss_G_OCR
            loss_G.backward()
            optimizer_G.step()
            
            # --- 2. 训练判别器 ---
            loss_D = torch.tensor(0.0).to(DEVICE)

            if not is_warmup and (step_counter % 3 == 0):
                optimizer_D.zero_grad()
                
                # 同样的降采样 + 噪声逻辑
                real_A_small = F.interpolate(real_A, scale_factor=0.5, mode='bilinear', align_corners=False)
                fake_B_small = F.interpolate(fake_B.detach(), scale_factor=0.5, mode='bilinear', align_corners=False)
                real_B_small = F.interpolate(real_B, scale_factor=0.5, mode='bilinear', align_corners=False)
                
                noisy_real_A = get_noisy_input(real_A_small, CONSTANT_SIGMA)
                noisy_fake_B = get_noisy_input(fake_B_small, CONSTANT_SIGMA)
                noisy_real_B = get_noisy_input(real_B_small, CONSTANT_SIGMA)
                
                # Fake
                pred_fake = netD(torch.cat((noisy_real_A, noisy_fake_B), 1))
                loss_D_fake = criterionGAN(pred_fake, False)
                
                # Real
                pred_real = netD(torch.cat((noisy_real_A, noisy_real_B), 1))
                target_real = torch.tensor(0.9).to(DEVICE).expand_as(pred_real)
                loss_D_real = torch.nn.MSELoss()(pred_real, target_real)
                
                loss_D = (loss_D_real + loss_D_fake) * 0.5
                loss_D.backward()
                optimizer_D.step()
            
            # 更新进度条
            status_str = f"Noise:0.05" if not is_warmup else "WARM-UP"
            loop.set_description(f"Ep [{epoch+1}/{EPOCHS}] {status_str}")
            loop.set_postfix(
                G_L1=f"{loss_G_L1.item():.2f}",
                D_loss=f"{loss_D.item():.2f}" 
            )
            
        if (epoch + 1) % 10 == 0:
            torch.save(netG.state_dict(), f"models/netG_epoch_{epoch+1}.pth")

if __name__ == '__main__':
    train()