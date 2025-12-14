import torch
import torch.nn as nn
from torchvision import models
import torchvision.transforms as transforms

class GANLoss(nn.Module):
    """标准对抗损失"""
    def __init__(self, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.loss = nn.MSELoss() # LSGAN

    def get_target_tensor(self, prediction, target_is_real):
        target_tensor = self.real_label if target_is_real else self.fake_label
        return target_tensor.expand_as(prediction)

    def forward(self, prediction, target_is_real):
        target_tensor = self.get_target_tensor(prediction, target_is_real)
        return self.loss(prediction, target_tensor)

class VGGLoss(nn.Module):
    """VGG 感知损失"""
    def __init__(self):
        super(VGGLoss, self).__init__()
        # 使用新版 weights 参数消除警告
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
        
        self.loss_network = nn.Sequential(*list(vgg)[:35]).eval()
        
        for param in self.loss_network.parameters():
            param.requires_grad = False
        self.criterion = nn.L1Loss()

    def forward(self, x, y):
        return self.criterion(self.loss_network(x), self.loss_network(y))

class OCRPerceptualLoss(nn.Module):
    """
    OCR 感知损失
    """
    def __init__(self):
        super(OCRPerceptualLoss, self).__init__()
        # 使用 ResNet50 的深层特征作为 OCR 语义的代理
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        
        # 提取 Layer3 的输出，包含较高级的语义/形状信息
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-3]).eval()
        
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
            
        self.criterion = nn.MSELoss()

        self.transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                            std=[0.229, 0.224, 0.225])

    def forward(self, generated, target):
        # 假设输入已经在 [-1, 1]，需要转换到 [0, 1] 然后 Normalize
        gen_norm = self.transform((generated + 1) / 2)
        tgt_norm = self.transform((target + 1) / 2)
        
        gen_feats = self.feature_extractor(gen_norm)
        tgt_feats = self.feature_extractor(tgt_norm)
        
        return self.criterion(gen_feats, tgt_feats)