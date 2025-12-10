import torch
import cv2
import numpy as np
from models.networks import UnetGenerator  # [修改点1] 导入 U-Net
from torchvision import transforms
from PIL import Image

# --- 配置 ---
MODEL_PATH = "models/netG_epoch_100.pth" # 记得改成你实际训练的轮次
INPUT_IMAGE = "datasets/origin_blur/6.jpg"
MASK_IMAGE = "blur_mask.png" 
OUTPUT_IMAGE = "final_restored_gray_2.png"
PATCH_SIZE = 512 # 推理时可以用大一点的 Patch，显存允许的话
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def inference_with_mask():
    # 1. 加载模型 [修改点2] 参数必须与 train.py 一致
    # input_nc=1, output_nc=1, num_downs=8
    netG = UnetGenerator(input_nc=1, output_nc=1, num_downs=8, ngf=64).to(DEVICE)
    netG.load_state_dict(torch.load(MODEL_PATH))
    netG.eval()
    
    # [修改点3] 灰度标准化
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # 2. 读取图像 (转为灰度)
    # 即使原图是彩色的，我们也只处理亮度信息，这样效果最锐利
    img_origin = cv2.imread(INPUT_IMAGE)
    img_gray = cv2.cvtColor(img_origin, cv2.COLOR_BGR2GRAY)
    h, w = img_gray.shape
    
    # 读取 Mask (用于判断模糊程度)
    mask = cv2.imread(MASK_IMAGE, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print("未找到 Mask，默认全图处理")
        mask = np.zeros((h, w), dtype=np.uint8)
    else:
        mask = cv2.resize(mask, (w, h))

    mask_weight = mask.astype(np.float32) / 255.0
    
    # 初始化画布 (单通道)
    canvas = np.zeros((h, w), dtype=np.float32)
    count_map = np.zeros((h, w), dtype=np.float32)
    
    # 3. 动态步长策略
    base_stride = 256
    
    print("开始分块推理 (Grayscale U-Net)...")
    
    with torch.no_grad():
        for y in range(0, h, base_stride):
            for x in range(0, w, base_stride):
                # 边界处理
                y_end = min(y + PATCH_SIZE, h)
                x_end = min(x + PATCH_SIZE, w)
                y_start = y_end - PATCH_SIZE
                x_start = x_end - PATCH_SIZE
                
                # 提取 Patch
                patch = img_gray[y_start:y_end, x_start:x_end]
                patch_pil = Image.fromarray(patch) # 自动识别为 'L' 模式
                
                # 增加 batch 维度: (1, 1, H, W)
                patch_tensor = transform(patch_pil).unsqueeze(0).to(DEVICE)
                
                # 推理
                fake_patch = netG(patch_tensor)
                
                # 后处理: (1, 1, H, W) -> (H, W)
                fake_patch = fake_patch.squeeze().cpu().numpy()
                fake_patch = (fake_patch * 0.5 + 0.5) * 255.0
                
                # --- 区域优化策略 ---
                # 获取当前块的模糊度
                patch_mask = mask_weight[y_start:y_end, x_start:x_end]
                avg_blur = np.mean(patch_mask) if patch_mask.size > 0 else 0
                
                # 如果是极度模糊区域，稍微抑制一下可能产生的高频噪点
                if avg_blur > 0.7:
                    fake_patch = cv2.GaussianBlur(fake_patch, (3, 3), 0.5)
                
                canvas[y_start:y_end, x_start:x_end] += fake_patch
                count_map[y_start:y_end, x_start:x_end] += 1
                
    # 4. 平均融合
    result = canvas / (count_map + 1e-8)
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    # --- 最终文档增强 (针对灰度结果) ---
    
    # A. 锐化 (Unsharp Masking) - 让笔画更清晰
    gaussian = cv2.GaussianBlur(result, (0, 0), 3.0)
    result = cv2.addWeighted(result, 1.5, gaussian, -0.5, 0)
    
    # B. 对比度拉伸 (CLAHE) - 让文字更黑，纸张更白
    #clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    #result = clahe.apply(result)
    
    cv2.imwrite(OUTPUT_IMAGE, result)
    print(f"推理完成，结果已保存至 {OUTPUT_IMAGE}")

if __name__ == '__main__':
    inference_with_mask()