import torch
from torch.nn import functional as F
from torch import nn
import math
from mmengine.registry import MODELS
class FeatureNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
    def forward(self, x):
        B, C, W, H = x.shape
        x = x.view(B, C, W*H).transpose(1, 2)
        x = F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)
        x = x.transpose(1, 2).view(B, C, W, H)
        return x

# class CAttention(nn.Module):
#     def __init__(self, in_ch, out_ch, head_n, attn_shape, kernel_size=3, stride=1, padding=1) -> None:
#         super().__init__()
#         assert out_ch % head_n == 0
#         self.out_ch = out_ch
#         self.head_n = head_n
#         self.c_attn = nn.Conv2d(in_ch, out_ch*3, kernel_size, stride, padding)
#         self.c_proj = nn.Conv2d(out_ch, out_ch, kernel_size, stride, padding)
#         HW = attn_shape[0]*attn_shape[1]
#         self.register_buffer('bias', torch.tril(
#             torch.ones(HW, HW)).view(1, 1, HW, HW))
        
#     def forward(self, x):
#         B, C, W, H = x.shape # batch, embed, (width, height)->sequence length
#         q, k, v = self.c_attn(x).split(self.out_ch, dim=1)
#         k = k.view(B, self.head_n, self.out_ch//self.head_n, W*H).transpose(-1, -2)
#         q = q.view(B, self.head_n, self.out_ch//self.head_n, W*H).transpose(-1, -2)
#         v = v.view(B, self.head_n, self.out_ch//self.head_n, W*H).transpose(-1, -2)
        
#         attn = (q @ k.transpose(-1, -2)) * (1.0 / math.sqrt(k.size(-1)))

#         attn = attn.masked_fill(self.bias[:,:,:H*W, :H*W]==0, float('-inf'))
        
#         attn = F.softmax(attn, dim=-1)
#         y = attn @ v # (B, head_n, W*H, W*H) @ (B, head_n, W*H, C//head_n) -> (B, head_n, W*H, C//head_n)
#         y = y.transpose(-1, -2).contiguous().view(B, self.out_ch, W, H)
#         # TODO: ADD channel drop
#         return y

class CAttention(nn.Module):
    def __init__(self, in_ch, out_ch, head_n, attn_shape) -> None:
        super().__init__()
        assert out_ch % head_n == 0
        self.out_ch = out_ch
        self.head_n = head_n
        
        self.c_attn = nn.ModuleList()
        for i in range(head_n):
            self.c_attn.append(nn.Conv2d(in_ch, 3*out_ch//self.head_n, (i%6)*2+1, 1, i%6))
        self.c_drop = nn.Dropout2d(0.1)
        self.attn_drop = nn.Dropout2d(0.1)
        self.c_proj = nn.Conv2d(out_ch, out_ch, 1, 1, 0)
        HW = attn_shape[0]*attn_shape[1]
        self.register_buffer('bias', torch.tril(
            torch.ones(HW, HW)).view(1, 1, HW, HW))
        
    def forward(self, x):
        B, C, W, H = x.shape # batch, embed, (width, height)->sequence length
        q, k, v = None, None, None
        for i in range(self.head_n):
            qt, kt, vt = self.c_attn[i](x).split(self.out_ch//self.head_n, dim=1)
            if q is None:
                q, k, v = qt, kt, vt
                continue
            q = torch.cat((q, qt), dim=1)
            k = torch.cat((k, kt), dim=1)
            v = torch.cat((v, vt), dim=1)
        k = k.view(B, self.head_n, self.out_ch//self.head_n, W*H).transpose(-1, -2)
        q = q.view(B, self.head_n, self.out_ch//self.head_n, W*H).transpose(-1, -2)
        v = v.view(B, self.head_n, self.out_ch//self.head_n, W*H).transpose(-1, -2)
        
        attn = (q @ k.transpose(-1, -2)) * (1.0 / math.sqrt(k.size(-1)))

        # attn = attn.masked_fill(self.bias[:,:,:H*W, :H*W]==0, float('-inf'))
        # attn = self.attn_drop(attn)
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        y = attn @ v # (B, head_n, W*H, W*H) @ (B, head_n, W*H, C//head_n) -> (B, head_n, W*H, C//head_n)
        y = y.transpose(-1, -2).contiguous().view(B, self.out_ch, W, H)
        y = self.c_proj(self.c_drop(y))
        return y


# class CAttention(nn.Module):
#     def __init__(self, in_ch, out_ch, head_n, attn_shape) -> None:
#         super().__init__()
#         assert out_ch % head_n == 0
#         self.out_ch = out_ch
#         self.head_n = head_n
#         self.c_attn = nn.Sequential(nn.Conv2d(in_ch, 3*out_ch//self.head_n, 1, 1, 0))
#         for i in range(head_n):
#             self.c_attn.add_module(f'head_conv_{i}', nn.Conv2d(out_ch*3, out_ch*3, (i%6)*2+1, 1, i%6))
#         self.c_proj = nn.Conv2d(out_ch, out_ch, 1, 1, 0)
#         HW = attn_shape[0]*attn_shape[1]
#         self.register_buffer('bias', torch.tril(
#             torch.ones(HW, HW)).view(1, 1, HW, HW))
        
#     def forward(self, x):
#         B, C, W, H = x.shape # batch, embed, (width, height)->sequence length
#         q, k, v = self.c_attn(x).split(self.out_ch, dim=1)
#         k = k.view(B, self.head_n, self.out_ch//self.head_n, W*H).transpose(-1, -2)
#         q = q.view(B, self.head_n, self.out_ch//self.head_n, W*H).transpose(-1, -2)
#         v = v.view(B, self.head_n, self.out_ch//self.head_n, W*H).transpose(-1, -2)
        
#         attn = (q @ k.transpose(-1, -2)) * (1.0 / math.sqrt(k.size(-1)))

#         attn = attn.masked_fill(self.bias[:,:,:H*W, :H*W]==0, float('-inf'))
        
#         attn = F.softmax(attn, dim=-1)
#         y = attn @ v # (B, head_n, W*H, W*H) @ (B, head_n, W*H, C//head_n) -> (B, head_n, W*H, C//head_n)
#         y = y.transpose(-1, -2).contiguous().view(B, self.out_ch, W, H)
#         # TODO: ADD channel drop
#         y = self.c_proj(y)
#         return y

class MLP(nn.Module):
    def __init__(self, in_ch, out_ch, is_bais=False):
        super().__init__()
        self.conv_1 = nn.Conv2d(in_ch, out_ch*4, 3, 1, 1)
        self.gelu = nn.GELU()
        self.c_proj = nn.Conv2d(out_ch*4, out_ch, 3, 1, 1)
    def forward(self, x):
        x = self.conv_1(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class CBlock(nn.Module):
    def __init__(self, in_ch, out_ch, attn_shape, head_n=1, kernel_size=3, stride=1, padding=1, bias=False) -> None:
        super().__init__()
        self.ln_1 = FeatureNorm(in_ch, bias)
        # self.attn = CAttention(in_ch, out_ch, head_n, attn_shape, kernel_size, stride, padding)
        self.attn = CAttention(in_ch, out_ch, head_n, attn_shape)
        self.ln_2 = FeatureNorm(out_ch, bias)
        self.mlp = MLP(out_ch, out_ch)
        self.bn = nn.BatchNorm2d(out_ch)
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        # x = self.bn(x)
        return x

class CEmbadding_Conv(nn.Module):
    def __init__(self, in_ch, out_ch, in_shape, out_shape, scale_num=1):
        super().__init__()
        self.scale_convs = nn.Sequential()
        if scale_num == 0:
            self.scale_convs.add_module('embding_conv_0', nn.Conv2d(in_ch, out_ch, 3, 1, 1))
            
        scale_shape = [x//(2**(2*scale_num)) for x in in_shape]
        self.embadding = nn.Sequential(
                            nn.AdaptiveAvgPool2d(scale_shape)   
                            , nn.Flatten(start_dim=-2)
                            , nn.Linear(scale_shape[0]*scale_shape[1]
                                        , out_shape[0]*out_shape[1])
                            , nn.Dropout(0.1)
                            )
        self.out_shape = out_shape  
        for i in range(1, scale_num+1):
            temp_in_ch = in_ch if i == 1 else (out_ch//scale_num)*i
            temp_out_ch = out_ch if i == scale_num else (out_ch//scale_num)*(i+1)
            self.scale_convs.add_module(f'embding_conv_{i}', nn.Conv2d(temp_in_ch, temp_out_ch, 3, 2, 1))
            self.scale_convs.add_module(f'embding_actv_{i}', nn.GELU())

    def forward(self, x):
        out = self.scale_convs(x)
        out = self.embadding(out)
        return out.view(out.shape[0], -1, *self.out_shape)
    
class CEmbadding(nn.Module):
    def __init__(self, in_ch, out_ch, in_shape, out_shape, scale_num=1):
        super().__init__()
        self.out_shape = out_shape
        self.scale_convs = nn.Sequential()
        self.scale_num = scale_num
        self.voc_size = 1024
        self.embedding = nn.Embedding(self.voc_size, out_ch)
        if scale_num == 0:
            self.scale_convs.add_module('embding_conv_0', nn.Conv2d(in_ch, self.voc_size, 3, 1, 1))
            self.scale_convs.add_module('embding_conv_0', nn.AdaptiveAvgPool2d(out_shape))
        else:
            self.create_embadding(in_ch, out_ch, out_shape, scale_num)

    def create_embadding(self,  in_ch, out_ch, out_shape, scale_num=1):
        for i in range(1, scale_num+1):
            temp_in_ch = in_ch if i == 1 else (self.voc_size//scale_num)*i
            temp_out_ch = self.voc_size if i == scale_num else (self.voc_size//scale_num)*(i+1)
            self.scale_convs.add_module(f'embding_conv_{i}', nn.Conv2d(temp_in_ch, temp_out_ch, 3, 2, 1))
            self.scale_convs.add_module(f'embding_actv_{i}', nn.ReLU())
        self.scale_convs.add_module('embding_conv_0', nn.AdaptiveAvgPool2d(out_shape))

    def forward(self, x):
        B, _, _, _ = x.shape
        out = self.scale_convs(x)
        out = F.softmax(out, dim=1)
        indices = out.argmax(dim=1, keepdim=True)
        indices = indices.view(B, -1) # (B, outshape[0]*outshape[1])
        embd = self.embedding(indices)
        return embd.transpose(-1, -2).view(B, -1, *self.out_shape)

class PositionEmbadding(nn.Module):
    def __init__(self, in_shape) -> None:
        super().__init__()
        self.pos = nn.Parameter(torch.randn(1, 1, *in_shape)*0.1)
    def forward(self, x):
        return self.pos

@MODELS.register_module()
class CNNTransformer(nn.Module):
    def __init__(self, in_ch, in_shape, attn_ch, attn_shape=None, scale_num=4, head_n=1, attn_times=3, embad_type='conv') -> None:
        super().__init__()
        if embad_type == 'conv':
            ipe = CEmbadding_Conv(in_ch, attn_ch, in_shape, attn_shape, scale_num)
        else:
            ipe = CEmbadding(in_ch, attn_ch, in_shape, attn_shape, scale_num)
        self.transformer = nn.ModuleDict(dict(
            ipe = ipe
            , wpe = PositionEmbadding(in_shape)
            , h = nn.ModuleList([
                CBlock(attn_ch, attn_ch, attn_shape, head_n, kernel_size=((i%6)*2+1), padding=(i%6)) for i in range(attn_times)
            ])
            , ln_f = nn.LayerNorm(attn_shape)
        ))
        # init all weights
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * attn_times))
        
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    def forward(self, x, targets=None):
        x = x + self.transformer['wpe'](x)
        o = self.transformer['ipe'](x)
        
        for block in self.transformer['h']:
            o = block(o)
        o = self.transformer['ln_f'](o)
        return o


if __name__ == "__main__":
    x = torch.randn(2, 3, 32, 32)
    cnn = CNNTransformer(3, (640, 640), 256, (16, 16), 2, 4, 3)
    # cnn(x).to('cuda')
    cnn.to('cuda')
    from torchsummary import summary
    from ptflops import get_model_complexity_info
    print(summary(cnn, input_size=(3, 640, 640)))  # 假设输入尺寸为 (3, 32, 32))
    macs, params = get_model_complexity_info(cnn, (3, 640, 640), as_strings=True, print_per_layer_stat=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))