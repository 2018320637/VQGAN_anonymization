from torch.autograd import Function
import torch.nn as nn

class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None

class DomainDiscriminator(nn.Module):
    def __init__(self, input_dim, domain_num):
        super(DomainDiscriminator, self).__init__()
        self.domain_fc = nn.Linear(input_dim, domain_num)
        self.temporal_pool = nn.AdaptiveAvgPool3d((1, None, None))
        self.spatial_pool = nn.AdaptiveAvgPool2d((1, 1))
        
    def forward(self, x, alpha=0.5):
        # Pooling: B,C,T,H,W -> B,C,H,W
        x = self.temporal_pool(x).squeeze(2) # [b, c, h, w]
        x = self.spatial_pool(x)
        # Flatten: B,C,H,W -> B,C*H*W
        x = x.view(x.size(0), -1)
        reverse_feature = ReverseLayerF.apply(x, alpha)
        output = self.domain_fc(reverse_feature)
        return output
    
class DomainDiscriminator_MLP(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=1024, domain_num=2):
        super(DomainDiscriminator_MLP, self).__init__()
        
        # 三层MLP
        self.domain_fc = nn.Sequential(
            # 第一层: input_dim -> hidden_dim
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(True),
            nn.Dropout(0.5),  # 添加dropout防止过拟合
            
            # 第二层: hidden_dim -> hidden_dim
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),
            nn.Dropout(0.5),
            
            # 第三层: hidden_dim -> domain_num
            nn.Linear(hidden_dim, domain_num)
        )

    def forward(self, x, alpha=0.5):
        # Pooling: B,C,T,H,W -> B,C,H,W
        x = self.temporal_pool(x).squeeze(2) # [b, c, h, w]
        x = self.spatial_pool(x)
        # Flatten: B,C,H,W -> B,C*H*W
        x = x.view(x.size(0), -1)
        reverse_feature = ReverseLayerF.apply(x, alpha)
        output = self.domain_fc(reverse_feature)
        return output
