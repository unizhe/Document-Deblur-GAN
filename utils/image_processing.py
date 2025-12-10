import cv2
import numpy as np
import random

def generate_disk_kernel(kernel_size):
    """
    生成圆盘散焦核
    """
    kernel = np.zeros((kernel_size, kernel_size), np.float32)
    center = kernel_size // 2
    radius = kernel_size // 2
    
    # 绘制白色圆盘
    cv2.circle(kernel, (center, center), radius, (1.0), -1)
    
    # 归一化，保证能量守恒 (关键：否则图片会变暗或变亮)
    kernel /= np.sum(kernel)
    return kernel

def apply_defocus_blur(image_patch):
    """
    应用强物理退化模型：超大圆盘模糊 + 辉光 + 噪声
    """
    # --- 修改点 1: 大幅增加模糊核范围 ---
    # 之前的 5-25 太小，对于大字体无法造成破坏性模糊
    # 现在提升到 31 - 71，这将导致严重的失焦
    k_size = random.choice(range(31, 71, 2)) 
    
    kernel = generate_disk_kernel(k_size)
    
    # 1. 物理圆盘卷积
    blurred = cv2.filter2D(image_patch, -1, kernel, borderType=cv2.BORDER_REFLECT)
    
    # --- 修改点 2: 添加轻微的高斯辉光 (Bloom Effect) ---
    # 真实的大失焦往往伴随着光线的溢出，不仅仅是硬边缘的圆盘
    # 这有助于消除圆盘核带来的人造“振铃”感
    blurred = cv2.GaussianBlur(blurred, (5, 5), 0)
    
    # --- 修改点 3: 噪声注入 ---
    # 模糊越严重，原本的高频信号越少，信噪比越低
    # 提高噪声上限，迫使模型学会从噪声中提取结构
    sigma = random.uniform(5, 20) # 之前是 0-15
    noise = np.random.normal(0, sigma, blurred.shape).astype(np.float32)
    blurred_noisy = blurred + noise
    
    # 截断并转回uint8
    blurred_noisy = np.clip(blurred_noisy, 0, 255).astype(np.uint8)
    
    return blurred_noisy

def is_valid_patch(patch, threshold=5):
    """
    背景过滤：剔除纯白或无信息的背景块
    """
    # 计算标准差
    std = np.std(patch)
    # 稍微降低阈值，防止因为模糊导致方差变小而被错误剔除
    if std < threshold:
        return False
    return True