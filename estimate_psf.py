import cv2
import numpy as np
import matplotlib.pyplot as plt

def estimate_blur_radius(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    h, w = img.shape
    
    # 将图像分为四个区域 (左上, 右上, 左下, 右下)
    # 用户提到：左下模糊大，右上模糊小
    regions = {
        "Top-Right (Light Blur)": img[0:h//2, w//2:w],
        "Bottom-Left (Heavy Blur)": img[h//2:h, 0:w//2],
        "Center": img[h//4:h*3//4, w//4:w*3//4]
    }
    
    print("--- 模糊半径粗略估计 ---")
    results = {}
    
    for name, patch in regions.items():
        # 使用拉普拉斯算子的方差来衡量清晰度（方差越小越模糊）
        variance = cv2.Laplacian(patch, cv2.CV_64F).var()
        
        # 这种映射是经验性的，通常方差与模糊半径成反比
        # 这里仅作为相对参考，用于设定数据生成的上下限
        # 假设清晰图方差 > 500
        estimated_r = max(1, int(1000 / (variance + 1)))
        
        # 修正：对于极端模糊，方差极小，半径估计会很大
        if estimated_r > 80: estimated_r = 80
        
        results[name] = estimated_r
        print(f"区域 [{name}]: Laplacian Variance = {variance:.2f}, 建议生成半径范围 ≈ {estimated_r-10} 到 {estimated_r+10}")

    return results

if __name__ == "__main__":
    # 请替换为你的实际路径
    estimate_blur_radius("datasets/origin_blur/6.jpg")