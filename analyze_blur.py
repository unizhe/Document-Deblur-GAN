import cv2
import numpy as np
import matplotlib.pyplot as plt

def generate_blur_map(image_path, save_mask_path="blur_mask.png"):
    # 1. 读取图像并转灰度
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # 2. 计算局部拉普拉斯方差 (Local Laplacian Variance)
    # 这种方法对边缘敏感，边缘越少（越模糊），值越低
    # 使用较大的核来平滑噪声
    laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
    
    # 3. 计算局部区域的能量 (方差)
    # 使用均值模糊来模拟局部窗口操作
    win_size = 51 
    abs_lap = np.abs(laplacian)
    local_activity = cv2.blur(abs_lap, (win_size, win_size))
    
    # 4. 归一化并反转 (让模糊区域的值变大，清晰区域变小)
    # 注意：通常方差大=清晰，方差小=模糊。
    # 我们希望 Mask 中：白色(255) = 需要强力修复(模糊)，黑色(0) = 轻微修复
    norm_activity = cv2.normalize(local_activity, None, 0, 255, cv2.NORM_MINMAX)
    blur_map = 255 - norm_activity # 反转
    
    # 5. 二值化/阈值处理以分割区域
    # 这一步是为了生成明确的 "重度模糊区" 和 "轻度模糊区"
    # 阈值需要根据实际情况微调，这里设为自动 Otsu 或者经验值
    _, binary_mask = cv2.threshold(blur_map.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 6. 形态学操作平滑 Mask (去除噪点)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 40))
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    
    # 还需要做一个高斯模糊，让过渡区域平滑，方便后续 Alpha Blending
    smooth_mask = cv2.GaussianBlur(binary_mask, (101, 101), 0)

    # 保存
    cv2.imwrite(save_mask_path, smooth_mask)
    
    # 可视化
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1), plt.title("Original"), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.subplot(1, 3, 2), plt.title("Blur Activity Map"), plt.imshow(blur_map, cmap='jet')
    plt.subplot(1, 3, 3), plt.title("Generated Mask (White=Heavy Blur)"), plt.imshow(smooth_mask, cmap='gray')
    plt.show()
    
    print(f"Mask 已生成: {save_mask_path} (白色区域代表重度模糊)")

if __name__ == "__main__":
    generate_blur_map("datasets/origin_blur/6.jpg")