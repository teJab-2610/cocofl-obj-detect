import torch
import math
from nets_quantized.utils.training import Conv2d, BatchNorm2d, Add, Cat


class Conv(torch.nn.Module):
    persistant_buffers = True
    track_running_stats = True
    
    def __init__(self, in_ch, out_ch, k=1, s=1, p=None, d=1, g=1, is_first=False):
        super().__init__()
        
        if p is None:
            if d > 1:
                k = d * (k - 1) + 1
            p = k // 2
        
        # Use our training-specific Conv2d and BatchNorm2d implementations
        self.conv = Conv2d(in_ch, out_ch, k, s, p, dilation=d, groups=g, bias=False,
                          persistant_buffers=self.persistant_buffers)
                          
        if is_first:
            self.norm = torch.nn.BatchNorm2d(out_ch, 0.001, 0.03, track_running_stats=False)
        else:
            self.norm = BatchNorm2d(out_ch, 0.001, 0.03, 
                                    track_running_stats=self.track_running_stats,
                                    persistant_buffers=self.persistant_buffers)
        
        # Using SiLU activation as in original model
        self.relu = torch.nn.SiLU(inplace=False)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        return x


class Residual(torch.nn.Module):
    persistant_buffers = True
    
    def __init__(self, ch, add=True, is_first=False):
        super().__init__()
        self.add_m = add
        self.res_m = torch.nn.Sequential(
            Conv(ch, ch, 3, is_first=is_first),
            Conv(ch, ch, 3)
        )
        if add:
            self.add = Add(persistant_buffers=self.persistant_buffers)
    
    def forward(self, x):
        out = self.res_m(x)
        if self.add_m:
            out = self.add(out, x)
        return out


class CSP(torch.nn.Module):
    persistant_buffers = True
    def __init__(self, in_ch, out_ch, n=1, add=True, is_first=False):
        super().__init__()
        
        self.conv1 = Conv(in_ch, out_ch // 2, is_first=is_first)
        self.conv2 = Conv(in_ch, out_ch // 2)
        self.conv3 = Conv((2 + n) * out_ch // 2, out_ch)
        
        # Create a list of residual blocks
        self.res_m = torch.nn.ModuleList()
        for i in range(n):
            self.res_m.append(Residual(out_ch // 2, add))
        
        self.n = n
        self.cat = Cat(persistant_buffers=self.persistant_buffers)
    
    def forward(self, x):
        y1 = self.conv1(x)
        y2 = self.conv2(x)
        
        outputs = [y1, y2]
        
        # Apply all residual blocks to the last output
        last_out = y2
        for i in range(self.n):
            last_out = self.res_m[i](last_out)
            outputs.append(last_out)
        
        # Concatenate all outputs and apply final convolution
        combined = self.cat(outputs, dim=1)
        return self.conv3(combined)


class SPP(torch.nn.Module):
    persistant_buffers = True
    
    def __init__(self, in_ch, out_ch, k=5, is_first=False):
        super().__init__()
        
        self.conv1 = Conv(in_ch, in_ch // 2, is_first=is_first)
        self.conv2 = Conv(in_ch * 2, out_ch)
        self.res_m = torch.nn.MaxPool2d(k, 1, k // 2)
        self.cat = Cat(persistant_buffers=self.persistant_buffers)
        self.k = k
    
    def forward(self, x):
        x = self.conv1(x)
        y1 = self.res_m(x)
        y2 = self.res_m(y1)
        y3 = self.res_m(y2)
        
        combined = self.cat([x, y1, y2, y3], dim=1)
        return self.conv2(combined)


class DarkNet(torch.nn.Module):
    def __init__(self, width, depth, is_first=False):
        super().__init__()
        
        p1 = [Conv(width[0], width[1], 3, 2, is_first=is_first)]
        p2 = [Conv(width[1], width[2], 3, 2),
              CSP(width[2], width[2], depth[0])]
        p3 = [Conv(width[2], width[3], 3, 2),
              CSP(width[3], width[3], depth[1])]
        p4 = [Conv(width[3], width[4], 3, 2),
              CSP(width[4], width[4], depth[2])]
        p5 = [Conv(width[4], width[5], 3, 2),
              CSP(width[5], width[5], depth[0]),
              SPP(width[5], width[5])]

        self.p1 = torch.nn.Sequential(*p1)
        self.p2 = torch.nn.Sequential(*p2)
        self.p3 = torch.nn.Sequential(*p3)
        self.p4 = torch.nn.Sequential(*p4)
        self.p5 = torch.nn.Sequential(*p5)
    
    def forward(self, x):
        p1 = self.p1(x)
        p2 = self.p2(p1)
        p3 = self.p3(p2)
        p4 = self.p4(p3)
        p5 = self.p5(p4)
        return p3, p4, p5


class DarkFPN(torch.nn.Module):
    persistant_buffers = True
    
    def __init__(self, width, depth, is_first=False):
        super().__init__()
        
        self.up = torch.nn.Upsample(None, 2)
        self.h1 = CSP(width[4] + width[5], width[4], depth[0], False, is_first=is_first)
        self.h2 = CSP(width[3] + width[4], width[3], depth[0], False)
        self.h3 = Conv(width[3], width[3], 3, 2)
        self.h4 = CSP(width[3] + width[4], width[4], depth[0], False)
        self.h5 = Conv(width[4], width[4], 3, 2)
        self.h6 = CSP(width[4] + width[5], width[5], depth[0], False)
        
        self.cat1 = Cat(persistant_buffers=self.persistant_buffers)
        self.cat2 = Cat(persistant_buffers=self.persistant_buffers)
        self.cat3 = Cat(persistant_buffers=self.persistant_buffers)
        self.cat4 = Cat(persistant_buffers=self.persistant_buffers)
    
    def forward(self, x):
        p3, p4, p5 = x
        
        p5_up = self.up(p5)
        cat1 = self.cat1([p5_up, p4], dim=1)
        h1 = self.h1(cat1)
        
        h1_up = self.up(h1)
        cat2 = self.cat2([h1_up, p3], dim=1)
        h2 = self.h2(cat2)
        
        h3 = self.h3(h2)
        cat3 = self.cat3([h3, h1], dim=1)
        h4 = self.h4(cat3)
        
        h5 = self.h5(h4)
        cat4 = self.cat4([h5, p5], dim=1)
        h6 = self.h6(cat4)
        return h2, h4, h6


class DFL(torch.nn.Module):
    # Integral module of Distribution Focal Loss (DFL)
    def __init__(self, ch=16, is_first=False):
        super().__init__()
        self.ch = ch
        self.conv = torch.nn.Conv2d(ch, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(ch, dtype=torch.float).view(1, ch, 1, 1)
        self.conv.weight.data[:] = torch.nn.Parameter(x)
    
    def forward(self, x):
        b, c, a = x.shape
        x = x.view(b, 4, self.ch, a).transpose(2, 1)
        return self.conv(x.softmax(1)).view(b, 4, a)


class Head(torch.nn.Module):

    ##TODO
    # anchors = torch.empty(0)
    # strides = torch.empty(0)
    
    def __init__(self, nc=80, filters=(), is_first=False):
        super().__init__()
        print('Trained Head', filters)
        self.ch = 16  # DFL channels
        self.nc = nc  # number of classes
        self.nl = len(filters)  # number of detection layers
        self.no = nc + self.ch * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build
        
        c1 = max(filters[0], self.nc)
        c2 = max((filters[0] // 4, self.ch * 4))
        
        self.dfl = DFL(self.ch)
        self.cls = torch.nn.ModuleList(torch.nn.Sequential(Conv(x, c1, 3),
                                                           Conv(c1, c1, 3),
                                                           torch.nn.Conv2d(c1, self.nc, 1)) for x in filters)
        self.box = torch.nn.ModuleList(torch.nn.Sequential(Conv(x, c2, 3),
                                                           Conv(c2, c2, 3),
                                                           torch.nn.Conv2d(c2, 4 * self.ch, 1)) for x in filters)

    
    def forward(self, x):
        for i in range(self.nl):
            x[i] = torch.cat((self.box[i](x[i]), self.cls[i](x[i])), 1)
            
        if self.training:
            return x
            
        # For inference
        self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
        
        x = torch.cat([i.view(x[0].shape[0], self.no, -1) for i in x], 2)
        box, cls = x.split((self.ch * 4, self.nc), 1)
        a, b = torch.split(self.dfl(box), 2, 1)
        a = self.anchors.unsqueeze(0) - a
        b = self.anchors.unsqueeze(0) + b
        box = torch.cat(((a + b) / 2, b - a), 1)
        return torch.cat((box * self.strides, cls.sigmoid()), 1)
    
    def initialize_biases(self):
        # Initialize biases
        for a, b, s in zip(self.box, self.cls, self.stride):
            a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[:self.nc] = math.log(5 / self.nc / (640 / s) ** 2)  # cls

def make_anchors(feats, strides, grid_cell_offset=0.5):
    """Generate anchors from features."""
    anchor_points, stride_tensor = [], []
    dtype, device = feats[0].dtype, feats[0].device
    for i, stride in enumerate(strides):
        _, _, h, w = feats[i].shape
        sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset  # shift x
        sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset  # shift y
        sy, sx = torch.meshgrid(sy, sx, indexing='ij')
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)

