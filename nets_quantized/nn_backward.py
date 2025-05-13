import torch
import math
from nets_quantized.utils.utils import filter_state_dict_keys
from nets_quantized.utils.backwards import QBWConv2dBN, QBWConv2d

class QBWConv(torch.nn.Module):
    def __init__(self, in_ch, out_ch, k=1, s=1, p=None, d=1, g=1):
        super().__init__()
        
        self._register_load_state_dict_pre_hook(self.sd_hook)
        
        if p is None:
            p = k // 2 if d == 1 else d * (k - 1) // 2
        
        self.convbn = QBWConv2dBN(in_ch, out_ch, kernel_size=k, stride=s, padding=p, groups=g)
        for parameter in self.parameters():
            parameter.requires_grad = False
    
    def sd_hook(self, state_dict, prefix, *_):
        self.convbn.init(
            filter_state_dict_keys(state_dict, prefix + 'conv.weight'),
            filter_state_dict_keys(state_dict, prefix + 'norm.weight'),
            filter_state_dict_keys(state_dict, prefix + 'norm.bias'),
            filter_state_dict_keys(state_dict, prefix + 'norm.running_mean'),
            filter_state_dict_keys(state_dict, prefix + 'norm.running_var'),
            # filter_state_dict_keys(state_dict, prefix + 'norm.op_scale'),
            # filter_state_dict_keys(state_dict, prefix + 'norm.op_scale_bw')
        )
    
    def forward(self, x):
        # print("QBWConv forward shape:", type(x))
        x = self.convbn(x)
        # Apply SiLU activation
        return torch.nn.functional.silu(x)


class QBWResidual(torch.nn.Module):
    def __init__(self, ch, add=True):
        super().__init__()
        
        self._register_load_state_dict_pre_hook(self.sd_hook)
        
        self.add_m = add
        self.res_m = torch.nn.Sequential(
            QBWConv(ch, ch, 3),
            QBWConv(ch, ch, 3)
        )
        
        for parameter in self.parameters():
            parameter.requires_grad = False
    
    def sd_hook(self, state_dict, prefix, *_):
        # Handle possible nested module path structures
        # No additional initialization needed beyond what's done in the QBWConv modules
        pass
    
    def forward(self, x):
        if self.add_m:
            return self.res_m(x) + x
        else:
            return self.res_m(x)

class QBWCSP(torch.nn.Module):
    def __init__(self, in_ch, out_ch, n=1, add=True):
        super().__init__()
        
        self._register_load_state_dict_pre_hook(self.sd_hook)
        
        self.conv1 = QBWConv(in_ch, out_ch // 2)
        self.conv2 = QBWConv(in_ch, out_ch // 2)
        self.conv3 = QBWConv((2 + n) * out_ch // 2, out_ch)
        self.res_m = torch.nn.ModuleList(QBWResidual(out_ch // 2, add) for _ in range(n))
        self.n = n
        
        for parameter in self.parameters():
            parameter.requires_grad = False
    
    def sd_hook(self, state_dict, prefix, *_):
        pass
    
    def forward(self, x):
        y1 = self.conv1(x)
        y2 = self.conv2(x)
        outputs = [y1, y2]

        last_output = y2
        for i in range(self.n):
            last_output = self.res_m[i](last_output)
            outputs.append(last_output)
        
        return self.conv3(torch.cat(outputs, dim=1))


class QBWSPP(torch.nn.Module):
    def __init__(self, in_ch, out_ch, k=5):
        super().__init__()
        
        self._register_load_state_dict_pre_hook(self.sd_hook)
        
        self.conv1 = QBWConv(in_ch, in_ch // 2)
        self.conv2 = QBWConv(in_ch * 2, out_ch)
        self.k = k
        
        for parameter in self.parameters():
            parameter.requires_grad = False
    
    def sd_hook(self, state_dict, prefix, *_):
        pass
    
    def forward(self, x):
        x = self.conv1(x)
        y1 = torch.nn.functional.max_pool2d(x, self.k, 1, self.k // 2)
        y2 = torch.nn.functional.max_pool2d(y1, self.k, 1, self.k // 2)
        y3 = torch.nn.functional.max_pool2d(y2, self.k, 1, self.k // 2)
        return self.conv2(torch.cat([x, y1, y2, y3], 1))


class QBWDarkNet(torch.nn.Module):
    def __init__(self, width, depth):
        super().__init__()
        
        self._register_load_state_dict_pre_hook(self.sd_hook)
        
        p1 = [QBWConv(width[0], width[1], 3, 2)]
        p2 = [QBWConv(width[1], width[2], 3, 2),
              QBWCSP(width[2], width[2], depth[0])]
        p3 = [QBWConv(width[2], width[3], 3, 2),
              QBWCSP(width[3], width[3], depth[1])]
        p4 = [QBWConv(width[3], width[4], 3, 2),
              QBWCSP(width[4], width[4], depth[2])]
        p5 = [QBWConv(width[4], width[5], 3, 2),
              QBWCSP(width[5], width[5], depth[0]),
              QBWSPP(width[5], width[5])]

        self.p1 = torch.nn.Sequential(*p1)
        self.p2 = torch.nn.Sequential(*p2)
        self.p3 = torch.nn.Sequential(*p3)
        self.p4 = torch.nn.Sequential(*p4)
        self.p5 = torch.nn.Sequential(*p5)
        
        for parameter in self.parameters():
            parameter.requires_grad = False
    
    def sd_hook(self, state_dict, prefix, *_):
        pass
    
    def forward(self, x):
        p1 = self.p1(x)
        p2 = self.p2(p1)
        p3 = self.p3(p2)
        p4 = self.p4(p3)
        p5 = self.p5(p4)
        return p3, p4, p5


class QBWDarkFPN(torch.nn.Module):
    def __init__(self, width, depth):
        super().__init__()
        
        self._register_load_state_dict_pre_hook(self.sd_hook)
        
        self.up = torch.nn.Upsample(None, 2)
        self.h1 = QBWCSP(width[4] + width[5], width[4], depth[0], False)
        self.h2 = QBWCSP(width[3] + width[4], width[3], depth[0], False)
        self.h3 = QBWConv(width[3], width[3], 3, 2)
        self.h4 = QBWCSP(width[3] + width[4], width[4], depth[0], False)
        self.h5 = QBWConv(width[4], width[4], 3, 2)
        self.h6 = QBWCSP(width[4] + width[5], width[5], depth[0], False)
        
        for parameter in self.parameters():
            parameter.requires_grad = False
    
    def sd_hook(self, state_dict, prefix, *_):
        pass
    
    def forward(self, x):
        p3, p4, p5 = x
        
        h1 = self.h1(torch.cat([self.up(p5), p4], 1))
        h2 = self.h2(torch.cat([self.up(h1), p3], 1))
        
        h3 = self.h3(h2)
        h4 = self.h4(torch.cat([h3, h1], 1))
        h5 = self.h5(h4)
        h6 = self.h6(torch.cat([h5, p5], 1))
        
        return h2, h4, h6


class QBWDFL(torch.nn.Module):
    def __init__(self, ch=16):
        super().__init__()
        
        self._register_load_state_dict_pre_hook(self.sd_hook)
        
        self.ch = ch
        # for backward compatibility
        self.conv = torch.nn.Conv2d(ch, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(ch, dtype=torch.float).view(1, ch, 1, 1)
        self.conv.weight.data[:] = torch.nn.Parameter(x)
        
        for parameter in self.parameters():
            parameter.requires_grad = False
    
    def sd_hook(self, state_dict, prefix, *_):
        key = prefix + 'conv.weight'
        if key in state_dict:
            self.conv.weight.copy_(state_dict[key])
    
    def forward(self, x):
        b, c, a = x.shape
        x = x.view(b, 4, self.ch, a).transpose(2, 1)
        return self.conv(x.softmax(1)).view(b, 4, a)

class QBWHead(torch.nn.Module):
    anchors = torch.empty(0)
    strides = torch.empty(0)
    
    def __init__(self, nc=80, filters=()):
        super().__init__()
        
        self._register_load_state_dict_pre_hook(self.sd_hook)
        
        self.ch = 16  # DFL channels
        self.nc = nc  # number of classes
        self.nl = len(filters)  # number of detection layers
        self.no = nc + self.ch * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build

        c1 = max(filters[0], self.nc)
        c2 = max((filters[0] // 4, self.ch * 4))
        
        self.dfl = QBWDFL(self.ch)

        self.box = []
        self.cls = []
        
        for i, x in enumerate(filters):
            self.box.append([
                QBWConv(x, c2, 3),
                QBWConv(c2, c2, 3),
                QBWConv2d(c2, 4 * self.ch, kernel_size=1, stride=1, padding=0)
            ])
            
            self.cls.append([
                QBWConv(x, c1, 3),
                QBWConv(c1, c1, 3),
                QBWConv2d(c1, self.nc, kernel_size=1, stride=1, padding=0)
            ])

        
        for parameter in self.parameters():
            parameter.requires_grad = False
    
    def sd_hook(self, state_dict, prefix, *_):
        for i in range(self.nl):
            # Initialize box branch components
            self.box[i][0].convbn.init(
                filter_state_dict_keys(state_dict, f'{prefix}box.{i}.0.conv.weight'),
                filter_state_dict_keys(state_dict, f'{prefix}box.{i}.0.norm.weight'),
                filter_state_dict_keys(state_dict, f'{prefix}box.{i}.0.norm.bias'),
                filter_state_dict_keys(state_dict, f'{prefix}box.{i}.0.norm.running_mean'),
                filter_state_dict_keys(state_dict, f'{prefix}box.{i}.0.norm.running_var'),
                # filter_state_dict_keys(state_dict, f'{prefix}box.{i}.0.norm.op_scale'),
                # filter_state_dict_keys(state_dict, f'{prefix}box.{i}.0.norm.op_scale_bw')
            )
            
            self.box[i][1].convbn.init(
                filter_state_dict_keys(state_dict, f'{prefix}box.{i}.1.conv.weight'),
                filter_state_dict_keys(state_dict, f'{prefix}box.{i}.1.norm.weight'),
                filter_state_dict_keys(state_dict, f'{prefix}box.{i}.1.norm.bias'),
                filter_state_dict_keys(state_dict, f'{prefix}box.{i}.1.norm.running_mean'),
                filter_state_dict_keys(state_dict, f'{prefix}box.{i}.1.norm.running_var'),
                # filter_state_dict_keys(state_dict, f'{prefix}box.{i}.1.norm.op_scale'),
                # filter_state_dict_keys(state_dict, f'{prefix}box.{i}.1.norm.op_scale_bw')
            )
            
            self.box[i][2].init(
                filter_state_dict_keys(state_dict, f'{prefix}box.{i}.2.weight'),
                # filter_state_dict_keys(state_dict, f'{prefix}box.{i}.2.bias'),
                # filter_state_dict_keys(state_dict, f'{prefix}box.{i}.2.op_scale'),
                # filter_state_dict_keys(state_dict, f'{prefix}box.{i}.2.op_scale_bw')
            )
            
            # Initialize cls branch components
            self.cls[i][0].convbn.init(
                filter_state_dict_keys(state_dict, f'{prefix}cls.{i}.0.conv.weight'),
                filter_state_dict_keys(state_dict, f'{prefix}cls.{i}.0.norm.weight'),
                filter_state_dict_keys(state_dict, f'{prefix}cls.{i}.0.norm.bias'),
                filter_state_dict_keys(state_dict, f'{prefix}cls.{i}.0.norm.running_mean'),
                filter_state_dict_keys(state_dict, f'{prefix}cls.{i}.0.norm.running_var'),
                # filter_state_dict_keys(state_dict, f'{prefix}cls.{i}.0.norm.op_scale'),
                # filter_state_dict_keys(state_dict, f'{prefix}cls.{i}.0.norm.op_scale_bw')
            )
            
            self.cls[i][1].convbn.init(
                filter_state_dict_keys(state_dict, f'{prefix}cls.{i}.1.conv.weight'),
                filter_state_dict_keys(state_dict, f'{prefix}cls.{i}.1.norm.weight'),
                filter_state_dict_keys(state_dict, f'{prefix}cls.{i}.1.norm.bias'),
                filter_state_dict_keys(state_dict, f'{prefix}cls.{i}.1.norm.running_mean'),
                filter_state_dict_keys(state_dict, f'{prefix}cls.{i}.1.norm.running_var'),
                # filter_state_dict_keys(state_dict, f'{prefix}cls.{i}.1.norm.op_scale'),
                # filter_state_dict_keys(state_dict, f'{prefix}cls.{i}.1.norm.op_scale_bw')
            )
            
            self.cls[i][2].init(
                filter_state_dict_keys(state_dict, f'{prefix}cls.{i}.2.weight'),
                # filter_state_dict_keys(state_dict, f'{prefix}cls.{i}.2.bias'),
                # filter_state_dict_keys(state_dict, f'{prefix}cls.{i}.2.op_scale'),
                # filter_state_dict_keys(state_dict, f'{prefix}cls.{i}.2.op_scale_bw')
            )
    
    def forward(self, x):
        for i in range(self.nl):
            box_out = x[i]
            cls_out = x[i]
            for j in range(len(self.box[i])):
                # print("QBW HEAD BOX OUT SHAPE", type(box_out), type(x[i]))
                box_out = self.box[i][j](box_out)
                
            for j in range(len(self.cls[i])):
                # print("QBW HEAD CLS OUT SHAPE", type(cls_out))
                cls_out = self.cls[i][j](cls_out)
            x[i] = torch.cat((box_out, cls_out), 1)
        if self.training:
            return x
        # For inference...
        self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))

        x = torch.cat([i.view(x[0].shape[0], self.no, -1) for i in x], 2)
        box, cls = x.split((self.ch * 4, self.nc), 1)
        a, b = torch.split(self.dfl(box), 2, 1)
        a = self.anchors.unsqueeze(0) - a
        b = self.anchors.unsqueeze(0) + b
        box = torch.cat(((a + b) / 2, b - a), 1)
        return torch.cat((box * self.strides, cls.sigmoid()), 1)
    
    def initialize_biases(self):
        for a, b, s in zip(self.box, self.cls, self.stride):
            if hasattr(a[2], 'bias') and a[2].bias is not None:
                a[2].bias.data[:] = 1.0
            if hasattr(b[2], 'bias') and b[2].bias is not None:
                b[2].bias.data[:self.nc] = math.log(5 / self.nc / (640 / s) ** 2)

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