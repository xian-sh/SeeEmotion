import torch
import torch.nn as nn
import torch.nn.functional as F

class EEGNet(torch.nn.Module):
    def __init__(self,
                 n_classes=5,
                 Chans=30,
                 Samples=500,
                 dropoutRate=0.5,
                 kernLength=300,
                 F1=8,
                 D=2,
                 F2=16):
        super().__init__()

        # 第一段：temporal conv + depthwise conv
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, kernLength), padding=(0, kernLength//2), bias=False),
            nn.BatchNorm2d(F1),
            # depthwise
            nn.Conv2d(F1, F1*D, (Chans, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1*D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropoutRate)
        )

        # 第二段：separable conv
        self.block2 = nn.Sequential(
            nn.Conv2d(F1*D, F1*D, (1, 16), groups=F1*D, padding=(0, 8), bias=False),
            nn.Conv2d(F1*D, F2, (1, 1), bias=False),   # pointwise
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropoutRate)
        )

        # 分类头
        self.flatten = nn.Flatten()
        # 计算一下 flatten 后的维度：方便后面写 Linear
        with torch.no_grad():
            faux = torch.zeros(1, 1, Chans, Samples)
            faux = self.block1(faux)
            faux = self.block2(faux)
            flat_dim = faux.numel()

        self.dense = nn.Linear(flat_dim, n_classes)

    def forward(self, x, return_feat=False):
        # x: (B, 1, Chans, Samples)
        x = x.unsqueeze(1)
        x = self.block1(x)
        x = self.block2(x)
        feat = x
        # print(x.shape)  # 32,16,1,15
        x = self.flatten(x)
        x = self.dense(x)
        if return_feat:
            return x, feat
        return x

if __name__ == '__main__':

    sample = torch.ones(32,30,500)
    model = EEGNet(n_classes=5, Chans=30, Samples=500)
    out = model(sample)
    print(out.shape)  # torch.Size([32, 5])
    from torchinfo import summary
    summary(model, (32, 30, 500), device='cpu')


# ==========================================================================================
# Layer (type:depth-idx)                   Output Shape              Param #
# ==========================================================================================
# EEGNet                                   [32, 5]                   --
# ├─Sequential: 1-1                        [32, 16, 1, 125]          --
# │    └─Conv2d: 2-1                       [32, 8, 30, 501]          2,400
# │    └─BatchNorm2d: 2-2                  [32, 8, 30, 501]          16
# │    └─Conv2d: 2-3                       [32, 16, 1, 501]          480
# │    └─BatchNorm2d: 2-4                  [32, 16, 1, 501]          32
# │    └─ELU: 2-5                          [32, 16, 1, 501]          --
# │    └─AvgPool2d: 2-6                    [32, 16, 1, 125]          --
# │    └─Dropout: 2-7                      [32, 16, 1, 125]          --
# ├─Sequential: 1-2                        [32, 16, 1, 15]           --
# │    └─Conv2d: 2-8                       [32, 16, 1, 126]          256
# │    └─Conv2d: 2-9                       [32, 16, 1, 126]          256
# │    └─BatchNorm2d: 2-10                 [32, 16, 1, 126]          32
# │    └─ELU: 2-11                         [32, 16, 1, 126]          --
# │    └─AvgPool2d: 2-12                   [32, 16, 1, 15]           --
# │    └─Dropout: 2-13                     [32, 16, 1, 15]           --
# ├─Flatten: 1-3                           [32, 240]                 --
# ├─Linear: 1-4                            [32, 5]                   1,205
# ==========================================================================================
# Total params: 4,677
# Trainable params: 4,677
# Non-trainable params: 0
# Total mult-adds (Units.GIGABYTES): 1.16
# ==========================================================================================
# Input size (MB): 1.92
# Forward/backward pass size (MB): 67.22
# Params size (MB): 0.02
# Estimated Total Size (MB): 69.16
# ==========================================================================================