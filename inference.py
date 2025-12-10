import torch
import cv2
import numpy as np
from models.networks import UnetGenerator
from torchvision import transforms
from PIL import Image

# --- 配置 ---
MODEL_PATH = "models/netG_epoch_100.pth"
INPUT_IMAGE = "datasets/origin_blur/6.jpg"
OUTPUT_IMAGE = "final_restored_clean_paper.png" # 最终完美版
PATCH_SIZE = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_window(size):
    hann = np.hanning(size)
    window_2d = np.outer(hann, hann)
    return window_2d

def normalize_background(img_gray):
    """
    核心去脏功能：背景归一化
    原理：原图 / 估算的背景 = 平整的图
    """
    # 1. 估算背景：使用膨胀操作 (Dilate) 移除黑色文字，只保留亮色背景
    # 核的大小 (kernel_size) 要大于最大的文字笔画粗细，但小于整体光照变化的范围
    # 35x35 通常足够覆盖大多数标题字
    dilated_bg = cv2.dilate(img_gray, np.ones((35, 35), np.uint8))
    
    # 2. 高斯模糊：让背景光照图更平滑
    bg_blur = cv2.GaussianBlur(dilated_bg, (15, 15), 0)
    
    # 3. 归一化：(原图 / 背景) * 255
    # 这一步会把背景变成纯白 (255)，同时保留文字的相对灰度
    # 避免除以0，加上一个极小值
    norm_img = img_gray.astype(np.float32) / (bg_blur.astype(np.float32) + 1e-5)
    
    # 缩放回 0-255 范围
    norm_img = norm_img * 255.0
    norm_img = np.clip(norm_img, 0, 255).astype(np.uint8)
    
    return norm_img

def post_process_smart(img_gray):
    # --- 第一步：修复文字断裂/中空 (GAN 伪影修复) ---
    # 反转图像，让字变白
    inv_img = 255 - img_gray
    # 闭运算：填补文字内部的黑洞，连接断裂
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    repaired_inv = cv2.morphologyEx(inv_img, cv2.MORPH_CLOSE, kernel_close)
    # 反转回来
    repaired = 255 - repaired_inv

    # --- 第二步：背景归一化 (解决背景不均一的核心) ---
    # 这步之后，背景会变得非常干净、均匀
    flat_img = normalize_background(repaired)
    
    # --- 第三步：双边滤波 (去噪) ---
    # 在背景变平滑后，去除残留的细微颗粒
    denoised = cv2.bilateralFilter(flat_img, d=5, sigmaColor=50, sigmaSpace=50)

    # --- 第四步：最终色阶微调 ---
    # 现在背景已经是纯白了，我们只需要稍微压暗一下文字，增加对比度
    # 既然背景已经是255了，highlight 不需要设太低
    # shadow 设为 80，让深灰色文字变黑
    table = np.arange(256, dtype=np.float32)
    shadow = 60
    highlight = 245 # 稍微留一点余地
    table = (table - shadow) * 255 / (highlight - shadow + 1e-5)
    table = np.clip(table, 0, 255).astype(np.uint8)
    
    final = cv2.LUT(denoised, table)
    
    return final

def inference_optimized():
    # 1. 加载模型
    netG = UnetGenerator(input_nc=1, output_nc=1, num_downs=8, ngf=64).to(DEVICE)
    netG.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    netG.eval()
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # 2. 读取图像
    img_origin = cv2.imread(INPUT_IMAGE)
    if img_origin is None: raise ValueError(f"无法读取: {INPUT_IMAGE}")
    img_gray = cv2.cvtColor(img_origin, cv2.COLOR_BGR2GRAY)
    h, w = img_gray.shape
    
    # 3. 初始化 (平滑拼接)
    canvas = np.zeros((h, w), dtype=np.float32)
    weight_map = np.zeros((h, w), dtype=np.float32)
    patch_weight = get_window(PATCH_SIZE)
    base_stride = PATCH_SIZE // 2
    
    print("正在进行分块推理...")
    
    with torch.no_grad():
        for y in range(0, h, base_stride):
            for x in range(0, w, base_stride):
                y_start = min(y, h - PATCH_SIZE)
                x_start = min(x, w - PATCH_SIZE)
                y_start = max(0, y_start)
                x_start = max(0, x_start)
                y_end = y_start + PATCH_SIZE
                x_end = x_start + PATCH_SIZE
                
                patch = img_gray[y_start:y_end, x_start:x_end]
                patch_pil = Image.fromarray(patch)
                patch_tensor = transform(patch_pil).unsqueeze(0).to(DEVICE)
                
                fake_patch = netG(patch_tensor)
                fake_patch = fake_patch.squeeze().cpu().numpy()
                fake_patch = (fake_patch * 0.5 + 0.5) * 255.0
                
                canvas[y_start:y_end, x_start:x_end] += fake_patch * patch_weight
                weight_map[y_start:y_end, x_start:x_end] += patch_weight
                
    # 原始融合结果
    raw_result = canvas / (weight_map + 1e-8)
    raw_result = np.clip(raw_result, 0, 255).astype(np.uint8)
    
    print("正在进行背景归一化和智能修复...")
    
    # --- 调用新的智能处理 ---
    final_result = post_process_smart(raw_result)
    
    cv2.imwrite(OUTPUT_IMAGE, final_result)
    print(f"处理完成！请查看: {OUTPUT_IMAGE}")

if __name__ == '__main__':
    inference_optimized()