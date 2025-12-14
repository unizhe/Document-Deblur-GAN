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
    
    # 归一化，保证能量守恒 
    kernel /= np.sum(kernel)
    return kernel

def apply_defocus_blur(image_patch):
    """
    应用强物理退化模型：超大圆盘模糊 + 辉光 + 噪声
    """
    k_size = random.choice(range(31, 71, 2)) 
    
    kernel = generate_disk_kernel(k_size)
    
    # 1. 物理圆盘卷积
    blurred = cv2.filter2D(image_patch, -1, kernel, borderType=cv2.BORDER_REFLECT)
    
    # 添加轻微的高斯辉光 (Bloom Effect) ---
    blurred = cv2.GaussianBlur(blurred, (5, 5), 0)
    
    # 噪声注入
    sigma = random.uniform(5, 20) 
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