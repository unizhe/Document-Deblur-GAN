import os
import cv2
import glob
import numpy as np
import random
from tqdm import tqdm

GT_DIR = "datasets/origin_gt"
OUTPUT_DIR = "datasets/patches"

PATCH_SIZE = 512

NUM_PATCHES = 3000  

# 模糊半径范围：根据之前的估计，覆盖轻微失焦到严重失焦
BLUR_KERNEL_RANGE = range(5, 90, 2) 

def generate_disk_kernel(kernel_size):
    """生成圆盘散焦核"""
    kernel = np.zeros((kernel_size, kernel_size), np.float32)
    center = kernel_size // 2
    radius = kernel_size // 2
    cv2.circle(kernel, (center, center), radius, (1.0), -1)
    kernel /= np.sum(kernel)
    return kernel

def apply_random_degradation(patch):
    """
    应用随机的物理退化：随机尺寸的散焦模糊 + 辉光 + 噪声
    """
    # 1. 随机选择一个模糊核大小
    k_size = random.choice(BLUR_KERNEL_RANGE)
    kernel = generate_disk_kernel(k_size)
    
    # 2. 卷积 (物理模糊)
    blurred = cv2.filter2D(patch, -1, kernel, borderType=cv2.BORDER_REFLECT)
    
    # 3. 模拟辉光 (Bloom) 
    blurred = cv2.GaussianBlur(blurred, (5, 5), 0)
    
    # 4. 注入噪声 (模糊越严重，信噪比越低，噪声应越大)
    max_k = BLUR_KERNEL_RANGE[-1]
    noise_level = 5 + (k_size / max_k) * 20
    sigma = random.uniform(5, noise_level)
    
    noise = np.random.normal(0, sigma, blurred.shape).astype(np.float32)
    blurred_noisy = blurred + noise
    
    blurred_noisy = np.clip(blurred_noisy, 0, 255).astype(np.uint8)
    return blurred_noisy

def is_valid_patch(patch, threshold=10):
    """简单过滤掉纯白背景，避免训练无效数据"""
    if np.std(patch) < threshold:
        return False
    return True

def prepare():
    # 清理并创建目录
    os.makedirs(os.path.join(OUTPUT_DIR, 'train_A'), exist_ok=True) # Input (Blur)
    os.makedirs(os.path.join(OUTPUT_DIR, 'train_B'), exist_ok=True) # Target (Sharp)
    
    gt_paths = sorted(glob.glob(os.path.join(GT_DIR, "*.png")))
    if not gt_paths:
        print(f"错误：在 {GT_DIR} 没有找到 .png 图片！")
        return

    print(f"找到 {len(gt_paths)} 张高清基准图，开始生成数据集...")
    
    gt_images = [cv2.imread(p) for p in gt_paths]
    
    count = 0
    pbar = tqdm(total=NUM_PATCHES)
    
    while count < NUM_PATCHES:
        # 1. 随机选一张图
        img_idx = random.randint(0, len(gt_images) - 1)
        src_img = gt_images[img_idx]
        h, w, _ = src_img.shape
        
        if h < PATCH_SIZE or w < PATCH_SIZE:
            print(f"警告：图片尺寸小于 PATCH_SIZE，跳过。")
            continue

        # 2. 随机坐标裁剪 (不翻转，不旋转)
        y = random.randint(0, h - PATCH_SIZE)
        x = random.randint(0, w - PATCH_SIZE)
        
        patch_B = src_img[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
        
        # 3. 过滤背景
        if not is_valid_patch(patch_B):
            continue
            
        # 4. 生成对应的模糊图 (Sim-to-Real)
        patch_A = apply_random_degradation(patch_B)
        
        # 5. 保存
        cv2.imwrite(f"{OUTPUT_DIR}/train_A/{count}.png", patch_A)
        cv2.imwrite(f"{OUTPUT_DIR}/train_B/{count}.png", patch_B)
        
        count += 1
        pbar.update(1)
        
    pbar.close()
    print(f"数据准备完成！生成了 {count} 对严格对齐方向的训练数据。")

if __name__ == '__main__':
    prepare()