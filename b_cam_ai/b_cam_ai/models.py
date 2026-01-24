import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
# 1. BUILDING BLOCKS (Basic & Optimized)
# ==============================================================================

class HardSwish(nn.Module):
    """
    Activation: HardSwish (MobileNetV3).
    Nhanh, nhẹ, hiệu quả hơn ReLU cho mạng sâu.
    """
    def __init__(self, inplace=True):
        super().__init__()
        self.relu6 = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return x * self.relu6(x + 3) / 6

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation: Channel Attention.
    Bias=True cho các lớp Conv nội bộ để học Global Descriptor tốt hơn.
    """
    def __init__(self, in_ch, reduction=4):
        super().__init__()
        reduced_ch = max(8, in_ch // reduction)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, reduced_ch, 1, bias=True), 
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_ch, in_ch, 1, bias=True), 
            nn.Hardsigmoid(inplace=True) 
        )

    def forward(self, x):
        return x * self.se(x)

class ConvBnAct(nn.Module):
    """
    Standard Block: Conv + BN + Act.
    """
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, groups=1, use_hs=True):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = HardSwish(inplace=True) if use_hs else nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class InvertedResidual(nn.Module):
    """
    MobileNet Block: Expand -> Depthwise -> SE -> Project.
    """
    def __init__(self, in_ch, out_ch, stride, expand_ratio=4, use_se=True):
        super().__init__()
        self.stride = stride
        hidden_dim = int(round(in_ch * expand_ratio))
        self.use_res_connect = self.stride == 1 and in_ch == out_ch

        layers = []
        if expand_ratio != 1:
            layers.append(ConvBnAct(in_ch, hidden_dim, kernel_size=1, padding=0, use_hs=True))
        
        layers.append(
            ConvBnAct(hidden_dim, hidden_dim, kernel_size=3, stride=stride, groups=hidden_dim, padding=1, use_hs=True)
        )
        
        if use_se:
            layers.append(SEBlock(hidden_dim))

        layers.extend([
            nn.Conv2d(hidden_dim, out_ch, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_ch),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class AuxHead(nn.Module):
    """
    CẬP NHẬT: Auxiliary Head mạnh mẽ hơn cho Deep Supervision.
    Thay vì Conv1x1 đơn lẻ, dùng Conv3x3 -> BN -> Act -> Conv1x1.
    Giúp gradient lan truyền về các tầng giữa ổn định hơn.
    """
    def __init__(self, in_ch, num_classes):
        super().__init__()
        self.conv = ConvBnAct(in_ch, in_ch // 2, kernel_size=3, padding=1, use_hs=True)
        self.dropout = nn.Dropout2d(p=0.1)
        self.classifier = nn.Conv2d(in_ch // 2, num_classes, kernel_size=1)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.dropout(x)
        x = self.classifier(x)
        return x

# ==============================================================================
# 2. ENCODERS (Image & Text)
# ==============================================================================

class CustomCNNBackbone(nn.Module):
    def __init__(self, dropout_p=0.1):
        super().__init__()
        self.dropout = nn.Dropout2d(p=dropout_p) if dropout_p > 0 else nn.Identity()
        
        # Input channel = 5 (3 RGB + 2 Coord)
        self.stem = nn.Sequential(
            ConvBnAct(5, 32, stride=2, use_hs=True),
            ConvBnAct(32, 32, stride=1, use_hs=True)
        )
        
        # Stage 1: Low (/4)
        self.stage1 = nn.Sequential(
            InvertedResidual(32, 16, stride=1, expand_ratio=1, use_se=False),
            InvertedResidual(16, 24, stride=2, expand_ratio=4, use_se=False), 
            InvertedResidual(24, 24, stride=1, expand_ratio=4, use_se=False)
        )
        
        # Stage 2: Mid (/8)
        self.stage2 = nn.Sequential(
            InvertedResidual(24, 32, stride=2, expand_ratio=4, use_se=True), 
            InvertedResidual(32, 32, stride=1, expand_ratio=4, use_se=True),
            InvertedResidual(32, 32, stride=1, expand_ratio=4, use_se=True)
        )
        
        # Stage 3+4: High (/32)
        self.stage3_4 = nn.Sequential(
            InvertedResidual(32, 64, stride=2, expand_ratio=4, use_se=True),
            InvertedResidual(64, 64, stride=1, expand_ratio=4, use_se=True),
            InvertedResidual(64, 96, stride=1, expand_ratio=4, use_se=True),
            InvertedResidual(96, 160, stride=2, expand_ratio=6, use_se=True),
            InvertedResidual(160, 320, stride=1, expand_ratio=6, use_se=True)
        )
        self.dims = [24, 32, 320] 

    def forward(self, x):
        x = self.stem(x)
        
        c1 = self.stage1(x)
        c1 = self.dropout(c1)
        
        c2 = self.stage2(c1)
        c2 = self.dropout(c2)
        
        c4 = self.stage3_4(c2)
        c4 = self.dropout(c4)
        
        return c1, c2, c4

class CustomTextEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, dropout_p=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.dropout = nn.Dropout(p=dropout_p)
        self.rnn = nn.GRU(input_size=embed_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.out_dim = hidden_dim * 2 
        self.bn = nn.BatchNorm1d(self.out_dim)

    def forward(self, x):
        x = self.embedding(x)
        x = self.dropout(x) 
        output, _ = self.rnn(x)
        sent_feat, _ = torch.max(output, dim=1) 
        sent_feat = self.bn(sent_feat)
        return sent_feat

# ==============================================================================
# 3. FUSION & MAIN MODEL
# ==============================================================================

class FusionModule(nn.Module):
    def __init__(self, img_ch, text_ch):
        super().__init__()
        self.text_proj = nn.Linear(text_ch, img_ch)
        self.conv = ConvBnAct(img_ch, img_ch, kernel_size=1, padding=0, use_hs=True)

    def forward(self, img_feat, text_feat):
        t = self.text_proj(text_feat) 
        t = t.unsqueeze(-1).unsqueeze(-1)
        x = img_feat * t
        return self.conv(x)

class CustomRefSeg(nn.Module):
    """
    Final Optimized Architecture:
    + CoordConv (Optimized Cache)
    + Deep Supervision (Enhanced AuxHead)
    + Full Regularization
    """
    def __init__(self, vocab_size, num_classes=1):
        super().__init__()
        
        self.image_encoder = CustomCNNBackbone(dropout_p=0.1) 
        self.text_encoder = CustomTextEncoder(vocab_size=vocab_size, dropout_p=0.3)
        
        img_dims = self.image_encoder.dims 
        text_dim = self.text_encoder.out_dim 
        
        # Fusion
        self.fuse_low = FusionModule(img_dims[0], text_dim)
        self.fuse_mid = FusionModule(img_dims[1], text_dim)
        self.fuse_high = FusionModule(img_dims[2], text_dim)
        
        # Decoder Upsampling
        self.up_high = ConvBnAct(img_dims[2], 128, use_hs=True)
        self.up_mid = ConvBnAct(128 + img_dims[1], 64, use_hs=True)
        self.up_low = ConvBnAct(64 + img_dims[0], 32, use_hs=True)
        
        self.decoder_dropout = nn.Dropout2d(p=0.1)
        
        # Main Head
        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1)
        
        # --- Deep Supervision Heads (Enhanced) ---
        self.aux_head_mid = AuxHead(64, num_classes)
        self.aux_head_low = AuxHead(32, num_classes)
        
        self._init_weights()
        
        # Buffer cho CoordConv
        self.register_buffer('grid_x', None)
        self.register_buffer('grid_y', None)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)

    def _get_coord_grid(self, x):
        """
        Tạo hoặc lấy grid từ cache.
        Cải tiến: Không cache theo Batch Size, chỉ cache theo Spatial Size (H, W).
        Dùng .expand() để khớp với Batch Size -> Tiết kiệm bộ nhớ & nhanh hơn.
        """
        B, _, H, W = x.shape
        # Chỉ tạo lại nếu kích thước không gian thay đổi
        if self.grid_x is None or self.grid_x.shape[2:] != (H, W):
            y_coords = torch.linspace(-1, 1, H, device=x.device).view(1, 1, H, 1)
            x_coords = torch.linspace(-1, 1, W, device=x.device).view(1, 1, 1, W)
            self.grid_y = y_coords
            self.grid_x = x_coords
        
        # Expand ra B dimension (Zero-copy, rất nhanh)
        grid_x = self.grid_x.expand(B, 1, H, W)
        grid_y = self.grid_y.expand(B, 1, H, W)
        
        return grid_x, grid_y

    def forward(self, img, text_ids):
        img_size = img.shape[-2:]
        
        # 0. CoordConv
        grid_x, grid_y = self._get_coord_grid(img)
        img_coord = torch.cat([img, grid_x, grid_y], dim=1)
        
        # A. Encode
        c1, c2, c4 = self.image_encoder(img_coord) 
        text_feat = self.text_encoder(text_ids)
        
        # B. Decode & Fuse
        # 1. High Level
        x = self.fuse_high(c4, text_feat)
        x = self.up_high(x)
        x = F.interpolate(x, size=c2.shape[-2:], mode='bilinear', align_corners=False)
        
        # 2. Mid Level
        f_mid = self.fuse_mid(c2, text_feat)
        x = torch.cat([x, f_mid], dim=1)
        x = self.up_mid(x)
        
        # --- Deep Supervision (Mid) ---
        aux_mid = None
        if self.training:
            aux_mid = self.aux_head_mid(x)
            aux_mid = F.interpolate(aux_mid, size=img_size, mode='bilinear', align_corners=False)
            
        x = F.interpolate(x, size=c1.shape[-2:], mode='bilinear', align_corners=False)
        
        # 3. Low Level
        f_low = self.fuse_low(c1, text_feat)
        x = torch.cat([x, f_low], dim=1)
        x = self.up_low(x)
        
        # --- Deep Supervision (Low) ---
        aux_low = None
        if self.training:
            aux_low = self.aux_head_low(x)
            aux_low = F.interpolate(aux_low, size=img_size, mode='bilinear', align_corners=False)
        
        # C. Final Prediction
        x = self.decoder_dropout(x)
        x = self.final_conv(x)
        x = F.interpolate(x, size=img_size, mode='bilinear', align_corners=False)
        
        if self.training:
            return {"main": x, "aux_mid": aux_mid, "aux_low": aux_low}
        else:
            return x

def build_model(vocab_size, device='cuda'):
    model = CustomRefSeg(vocab_size=vocab_size)
    params = sum(p.numel() for p in model.parameters())
    print(f"[Model Info] CustomRefSeg FINAL v3 (Perfect CoordConv Cache).")
    print(f"[Model Info] Total Parameters: {params/1e6:.2f}M")
    return model.to(device)

if __name__ == "__main__":
    model = build_model(vocab_size=3000, device='cpu')
    dummy_img = torch.randn(2, 3, 320, 320)
    dummy_txt = torch.randint(0, 3000, (2, 10))
    model.train()
    out = model(dummy_img, dummy_txt)
    print(f"Training Outputs: {out.keys()}")