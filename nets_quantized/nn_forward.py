import torch
from torch import nn
import math
from nets_quantized.utils.utils import filter_state_dict_keys, tensor_scale
from nets_quantized.utils.forward import QConv2dBN, QAdd, QLinear, QCat, QBatchNorm2drelu, QConv2dBNRelu

class QFWConv(torch.nn.Module):
    def __init__(self, in_ch, out_ch, k=1, s=1, p=None, d=1, g=1, is_transition=False):
        super().__init__()
        
        self._register_load_state_dict_pre_hook(self.sd_hook)
        
        if p is None:
            p = k // 2 if d == 1 else d * (k - 1) // 2
        
        self.convbnrelu = QConv2dBNRelu(in_ch, out_ch, kernel_size=k, stride=s, padding=p, groups=g)
        self.is_transition = is_transition
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

        for param in self.parameters():
            param.requires_grad = False
    
    def sd_hook(self, state_dict, prefix, *_):
        self.convbnrelu.init(
            filter_state_dict_keys(state_dict, prefix + 'conv.weight'),
            filter_state_dict_keys(state_dict, prefix + 'norm.weight'),
            filter_state_dict_keys(state_dict, prefix + 'norm.bias'),
            filter_state_dict_keys(state_dict, prefix + 'norm.running_mean'),
            filter_state_dict_keys(state_dict, prefix + 'norm.running_var'),
            # filter_state_dict_keys(state_dict, prefix + 'norm.op_scale')
        )
    
    def forward(self, x):
        if not x.is_quantized:
            x = torch.quantize_per_tensor(x, tensor_scale(x), 64, dtype=torch.quint8)
            # x = self.quant(x)
        out = self.convbnrelu(x)

        # Apply SiLU activation (need to dequantize-quantize as there's no quantized SiLU in PyTorch)
        out = torch.dequantize(out)
        out = torch.nn.functional.silu(out)

        if not self.is_transition:
            out = torch.quantize_per_tensor(out, tensor_scale(out), 64, dtype=torch.quint8)
            # out = self.dequant(out)

        return out

class QFWResidual(torch.nn.Module):
    def __init__(self, ch, add=True, is_transition=False):
        super().__init__()
        
        self._register_load_state_dict_pre_hook(self.sd_hook)
        
        self.add_m = add
        # self.conv1 = QFWConv(ch, ch, 3)
        # self.conv2 = QFWConv(ch, ch, 3)
        self.res_m = torch.nn.Sequential(
            QFWConv(ch, ch, 3),
            QFWConv(ch, ch, 3)
        )
        
        if add:
            self.add = QAdd()
        
        self.is_transition = is_transition
        

        for param in self.parameters():
            param.requires_grad = False
    
    def sd_hook(self, state_dict, prefix, *_):
        self.res_m[0].convbnrelu.init(
            filter_state_dict_keys(state_dict, prefix + 'res_m.0.conv.weight'),
            filter_state_dict_keys(state_dict, prefix + 'res_m.0.norm.weight'),
            filter_state_dict_keys(state_dict, prefix + 'res_m.0.norm.bias'),
            filter_state_dict_keys(state_dict, prefix + 'res_m.0.norm.running_mean'),
            filter_state_dict_keys(state_dict, prefix + 'res_m.0.norm.running_var'),
            # filter_state_dict_keys(state_dict, prefix + 'res_m.0.norm.op_scale')
        )
        self.res_m[1].convbnrelu.init(
            filter_state_dict_keys(state_dict, prefix + 'res_m.1.conv.weight'),
            filter_state_dict_keys(state_dict, prefix + 'res_m.1.norm.weight'),
            filter_state_dict_keys(state_dict, prefix + 'res_m.1.norm.bias'),
            filter_state_dict_keys(state_dict, prefix + 'res_m.1.norm.running_mean'),
            filter_state_dict_keys(state_dict, prefix + 'res_m.1.norm.running_var'),
            # filter_state_dict_keys(state_dict, prefix + 'res_m.1.norm.op_scale')
        )
        
        if self.add_m:
            self.add.init()
        # if self.add_m:
        #     self.add.init(filter_state_dict_keys(state_dict, prefix + 'add.op_scale'))
    
    def forward(self, x):
        # out = self.conv2(self.conv1(x))
        out = self.res_m(x)
        if self.add_m:
            out = self.add(out, x)
        
        if self.is_transition:
            out = torch.dequantize(out)
        
        return out

class QFWCSP(torch.nn.Module):
    def __init__(self, in_ch, out_ch, n=1, add=True, is_transition=False):
        super().__init__()
        
        self._register_load_state_dict_pre_hook(self.sd_hook)
        
        self.conv1 = QFWConv(in_ch, out_ch // 2)
        self.conv2 = QFWConv(in_ch, out_ch // 2)
        self.conv3 = QFWConv((2 + n) * out_ch // 2, out_ch)
        self.res_m = torch.nn.ModuleList(QFWResidual(out_ch // 2, add) for _ in range(n))
        
        self.cat = QCat()
        self.is_transition = is_transition
        self.n = n
        
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

        for param in self.parameters():
            param.requires_grad = False
    
    def sd_hook(self, state_dict, prefix, *_):

        self.conv1.convbnrelu.init(
            filter_state_dict_keys(state_dict, prefix + 'conv1.conv.weight'),
            filter_state_dict_keys(state_dict, prefix + 'conv1.norm.weight'),
            filter_state_dict_keys(state_dict, prefix + 'conv1.norm.bias'),
            filter_state_dict_keys(state_dict, prefix + 'conv1.norm.running_mean'),
            filter_state_dict_keys(state_dict, prefix + 'conv1.norm.running_var'),
            # filter_state_dict_keys(state_dict, prefix + 'conv1.norm.op_scale')
        )
        self.conv2.convbnrelu.init(
            filter_state_dict_keys(state_dict, prefix + 'conv2.conv.weight'),
            filter_state_dict_keys(state_dict, prefix + 'conv2.norm.weight'),
            filter_state_dict_keys(state_dict, prefix + 'conv2.norm.bias'),
            filter_state_dict_keys(state_dict, prefix + 'conv2.norm.running_mean'),
            filter_state_dict_keys(state_dict, prefix + 'conv2.norm.running_var'),
            # filter_state_dict_keys(state_dict, prefix + 'conv2.norm.op_scale')
        )
        self.conv3.convbnrelu.init(
            filter_state_dict_keys(state_dict, prefix + 'conv3.conv.weight'),
            filter_state_dict_keys(state_dict, prefix + 'conv3.norm.weight'),
            filter_state_dict_keys(state_dict, prefix + 'conv3.norm.bias'),
            filter_state_dict_keys(state_dict, prefix + 'conv3.norm.running_mean'),
            filter_state_dict_keys(state_dict, prefix + 'conv3.norm.running_var'),
            # filter_state_dict_keys(state_dict, prefix + 'conv3.norm.op_scale')
        )
        
        # for i, module in enumerate(self.res_m):
        #     module.sd_hook(state_dict, prefix)
        
        self.cat.init()
        # self.cat.init(filter_state_dict_keys(state_dict, prefix + 'cat.op_scale'))

    def forward(self, x):
        if not x.is_quantized:
            x = torch.quantize_per_tensor(x, tensor_scale(x), 64, dtype=torch.quint8)
            # x = self.quant(x)
        y1 = self.conv1(x)
        y2 = self.conv2(x)
        
        outputs = [y1, y2]
        
        # Apply residual blocks to the second output
        last_output = y2
        for i in range(self.n):
            last_output = self.res_m[i](last_output)
            outputs.append(last_output)
        
        combined = self.cat(outputs, dim=1)
        out = self.conv3(combined)
        
        if self.is_transition:
            # out = torch.dequantize(out)
            out = self.dequant(out)
    
        return out

class QFWSPP(torch.nn.Module):
    def __init__(self, in_ch, out_ch, k=5, is_transition=False):
        super().__init__()
        
        self._register_load_state_dict_pre_hook(self.sd_hook)
        
        self.conv1 = QFWConv(in_ch, in_ch // 2)
        self.conv2 = QFWConv(in_ch * 2, out_ch)
        self.cat = QCat()
        self.is_transition = is_transition
        self.k = k

        for param in self.parameters():
            param.requires_grad = False
    
    def sd_hook(self, state_dict, prefix, *_):
        self.conv1.convbnrelu.init(
            filter_state_dict_keys(state_dict, prefix + 'conv1.conv.weight'),
            filter_state_dict_keys(state_dict, prefix + 'conv1.norm.weight'),
            filter_state_dict_keys(state_dict, prefix + 'conv1.norm.bias'),
            filter_state_dict_keys(state_dict, prefix + 'conv1.norm.running_mean'),
            filter_state_dict_keys(state_dict, prefix + 'conv1.norm.running_var'),
            # filter_state_dict_keys(state_dict, prefix + 'conv1.norm.op_scale')
        )
        self.conv2.convbnrelu.init(
            filter_state_dict_keys(state_dict, prefix + 'conv2.conv.weight'),
            filter_state_dict_keys(state_dict, prefix + 'conv2.norm.weight'),
            filter_state_dict_keys(state_dict, prefix + 'conv2.norm.bias'),
            filter_state_dict_keys(state_dict, prefix + 'conv2.norm.running_mean'),
            filter_state_dict_keys(state_dict, prefix + 'conv2.norm.running_var'),
            # filter_state_dict_keys(state_dict, prefix + 'conv2.norm.op_scale')
        )
        self.cat.init()
        # self.cat.init(filter_state_dict_keys(state_dict, prefix + 'cat.op_scale'))

    def forward(self, x):
        x = self.conv1(x)
        
        # Need to dequantize for maxpool operations
        x_dequant = torch.dequantize(x)
        y1_dequant = torch.nn.functional.max_pool2d(x_dequant, self.k, 1, self.k // 2)
        y2_dequant = torch.nn.functional.max_pool2d(y1_dequant, self.k, 1, self.k // 2)
        y3_dequant = torch.nn.functional.max_pool2d(y2_dequant, self.k, 1, self.k // 2)
        
        # Using QCat for concatenation
        outputs = [x]
        if torch.is_tensor(x) and x.is_quantized:
            # If x is quantized, quantize the maxpool outputs
            y1 = torch.quantize_per_tensor(y1_dequant, tensor_scale(y1_dequant), 64, dtype=torch.quint8)
            y2 = torch.quantize_per_tensor(y2_dequant, tensor_scale(y2_dequant), 64, dtype=torch.quint8)
            y3 = torch.quantize_per_tensor(y3_dequant, tensor_scale(y3_dequant), 64, dtype=torch.quint8)
            outputs.extend([y1, y2, y3])
            cat_quant = self.cat(outputs, dim=1)
        else:
            # Otherwise, concatenate the dequantized tensors
            cat_dequant = torch.cat([x_dequant, y1_dequant, y2_dequant, y3_dequant], 1)
            cat_quant = torch.quantize_per_tensor(cat_dequant, tensor_scale(cat_dequant), 64, dtype=torch.quint8)
        
        out = self.conv2(cat_quant)
        
        if self.is_transition:
            out = torch.dequantize(out)
        
        return out

class QFWDarkNet(torch.nn.Module):
    def __init__(self, width, depth, is_transition=False):
        super().__init__()

        self._register_load_state_dict_pre_hook(self.sd_hook)
        
        p1 = [QFWConv(width[0], width[1], 3, 2)]
        p2 = [QFWConv(width[1], width[2], 3, 2),
              QFWCSP(width[2], width[2], depth[0])]
        p3 = [QFWConv(width[2], width[3], 3, 2),
              QFWCSP(width[3], width[3], depth[1])]
        p4 = [QFWConv(width[3], width[4], 3, 2),
              QFWCSP(width[4], width[4], depth[2])]
        p5 = [QFWConv(width[4], width[5], 3, 2),
              QFWCSP(width[5], width[5], depth[0]),
              QFWSPP(width[5], width[5])]

        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

        self.p1 = torch.nn.Sequential(*p1)
        self.p2 = torch.nn.Sequential(*p2)
        self.p3 = torch.nn.Sequential(*p3)
        self.p4 = torch.nn.Sequential(*p4)
        self.p5 = torch.nn.Sequential(*p5)
        self.is_transition = is_transition
        

        for param in self.parameters():
            param.requires_grad = False
    
    def sd_hook(self, state_dict, prefix, *_):
        # Forward the state_dict to the submodules
        try:
            self.p1[0].convbnrelu.init(
                filter_state_dict_keys(state_dict, prefix + 'p1.0.conv.weight'),
                filter_state_dict_keys(state_dict, prefix + 'p1.0.norm.weight'),
                filter_state_dict_keys(state_dict, prefix + 'p1.0.norm.bias'),
                filter_state_dict_keys(state_dict, prefix + 'p1.0.norm.running_mean'),
                filter_state_dict_keys(state_dict, prefix + 'p1.0.norm.running_var'),
                # filter_state_dict_keys(state_dict, prefix + 'p1.0.norm.op_scale')
            )
            self.p2[0].convbnrelu.init(
                filter_state_dict_keys(state_dict, prefix + 'p2.0.conv.weight'),
                filter_state_dict_keys(state_dict, prefix + 'p2.0.norm.weight'),
                filter_state_dict_keys(state_dict, prefix + 'p2.0.norm.bias'),
                filter_state_dict_keys(state_dict, prefix + 'p2.0.norm.running_mean'),
                filter_state_dict_keys(state_dict, prefix + 'p2.0.norm.running_var'),
                # filter_state_dict_keys(state_dict, prefix + 'p2.0.norm.op_scale')
            )
            self.p3[0].convbnrelu.init(
                filter_state_dict_keys(state_dict, prefix + 'p3.0.conv.weight'),
                filter_state_dict_keys(state_dict, prefix + 'p3.0.norm.weight'),
                filter_state_dict_keys(state_dict, prefix + 'p3.0.norm.bias'),
                filter_state_dict_keys(state_dict, prefix + 'p3.0.norm.running_mean'),
                filter_state_dict_keys(state_dict, prefix + 'p3.0.norm.running_var'),
                # filter_state_dict_keys(state_dict, prefix + 'p3.0.norm.op_scale')
            )
            self.p3[1].sd_hook(state_dict, prefix + 'p3.1.')
            self.p4[0].convbnrelu.init(
                filter_state_dict_keys(state_dict, prefix + 'p4.0.conv.weight'),
                filter_state_dict_keys(state_dict, prefix + 'p4.0.norm.weight'),
                filter_state_dict_keys(state_dict, prefix + 'p4.0.norm.bias'),
                filter_state_dict_keys(state_dict, prefix + 'p4.0.norm.running_mean'),
                filter_state_dict_keys(state_dict, prefix + 'p4.0.norm.running_var'),
                # filter_state_dict_keys(state_dict, prefix + 'p4.0.norm.op_scale')
            )
            self.p4[1].sd_hook(state_dict, prefix + 'p4.1.')
            self.p5[0].convbnrelu.init(
                filter_state_dict_keys(state_dict, prefix + 'p5.0.conv.weight'),
                filter_state_dict_keys(state_dict, prefix + 'p5.0.norm.weight'),
                filter_state_dict_keys(state_dict, prefix + 'p5.0.norm.bias'),
                filter_state_dict_keys(state_dict, prefix + 'p5.0.norm.running_mean'),
                filter_state_dict_keys(state_dict, prefix + 'p5.0.norm.running_var'),
                # filter_state_dict_keys(state_dict, prefix + 'p5.0.norm.op_scale')
            )
            self.p5[1].sd_hook(state_dict, prefix + 'p5.1.')
            self.p5[2].sd_hook(state_dict, prefix + 'p5.2.')
        except Exception as e:

            import traceback

    def forward(self, x):
        if not self.is_transition and not torch.is_tensor(x) or not x.is_quantized:
            x = torch.quantize_per_tensor(x, tensor_scale(x), 64, dtype=torch.quint8)
            # x = self.quant(x)
        p1 = self.p1(x)
        p2 = self.p2(p1)
        p3 = self.p3(p2)
        p4 = self.p4(p3)
        p5 = self.p5(p4)
        
        # Dequantize outputs if this is the final layer before a non-quantized layer
        if self.is_transition and p3.is_quantized:
            p3 = torch.dequantize(p3)
            p4 = torch.dequantize(p4)
            p5 = torch.dequantize(p5)
            # p3 = self.dequant(p3)
            # p4 = self.dequant(p4)
            # p5 = self.dequant(p5)
        
        return p3, p4, p5

class QFWDarkFPN(torch.nn.Module):
    def __init__(self, width, depth, is_transition=False):
        super().__init__()
        
        self._register_load_state_dict_pre_hook(self.sd_hook)
        
        self.h1 = QFWCSP(width[4] + width[5], width[4], depth[0], False)
        self.h2 = QFWCSP(width[3] + width[4], width[3], depth[0], False)
        self.h3 = QFWConv(width[3], width[3], 3, 2)
        self.h4 = QFWCSP(width[3] + width[4], width[4], depth[0], False)
        self.h5 = QFWConv(width[4], width[4], 3, 2)
        self.h6 = QFWCSP(width[4] + width[5], width[5], depth[0], False)
        
        self.cat1 = QCat()
        self.cat2 = QCat()
        self.cat3 = QCat()
        self.cat4 = QCat()
        
        self.is_transition = is_transition
        

        for param in self.parameters():
            param.requires_grad = False
    
    def sd_hook(self, state_dict, prefix, *_):    
        self.h1.conv1.convbnrelu.init(
            filter_state_dict_keys(state_dict, prefix + 'h1.conv1.conv.weight'),
            filter_state_dict_keys(state_dict, prefix + 'h1.conv1.norm.weight'),
            filter_state_dict_keys(state_dict, prefix + 'h1.conv1.norm.bias'),
            filter_state_dict_keys(state_dict, prefix + 'h1.conv1.norm.running_mean'),
            filter_state_dict_keys(state_dict, prefix + 'h1.conv1.norm.running_var'),
            # filter_state_dict_keys(state_dict, prefix + 'h1.conv1.norm.op_scale')
        )
        self.h1.conv2.convbnrelu.init(
            filter_state_dict_keys(state_dict, prefix + 'h1.conv2.conv.weight'),
            filter_state_dict_keys(state_dict, prefix + 'h1.conv2.norm.weight'),
            filter_state_dict_keys(state_dict, prefix + 'h1.conv2.norm.bias'),
            filter_state_dict_keys(state_dict, prefix + 'h1.conv2.norm.running_mean'),
            filter_state_dict_keys(state_dict, prefix + 'h1.conv2.norm.running_var'),
            # filter_state_dict_keys(state_dict, prefix + 'h1.conv2.norm.op_scale')
        )
        self.h1.conv3.convbnrelu.init(
            filter_state_dict_keys(state_dict, prefix + 'h1.conv3.conv.weight'),
            filter_state_dict_keys(state_dict, prefix + 'h1.conv3.norm.weight'),
            filter_state_dict_keys(state_dict, prefix + 'h1.conv3.norm.bias'),
            filter_state_dict_keys(state_dict, prefix + 'h1.conv3.norm.running_mean'),
            filter_state_dict_keys(state_dict, prefix + 'h1.conv3.norm.running_var'),
            # filter_state_dict_keys(state_dict, prefix + 'h1.conv3.norm.op_scale')
        )
        for i, module in enumerate(self.h1.res_m):
            module.sd_hook(state_dict, prefix + f'h1.res_m.{i}.')
            
        self.h2.conv1.convbnrelu.init(
            filter_state_dict_keys(state_dict, prefix + 'h2.conv1.conv.weight'),
            filter_state_dict_keys(state_dict, prefix + 'h2.conv1.norm.weight'),
            filter_state_dict_keys(state_dict, prefix + 'h2.conv1.norm.bias'),
            filter_state_dict_keys(state_dict, prefix + 'h2.conv1.norm.running_mean'),
            filter_state_dict_keys(state_dict, prefix + 'h2.conv1.norm.running_var'),
            # filter_state_dict_keys(state_dict, prefix + 'h2.conv1.norm.op_scale')
        )
        self.h2.conv2.convbnrelu.init(
            filter_state_dict_keys(state_dict, prefix + 'h2.conv2.conv.weight'),
            filter_state_dict_keys(state_dict, prefix + 'h2.conv2.norm.weight'),
            filter_state_dict_keys(state_dict, prefix + 'h2.conv2.norm.bias'),
            filter_state_dict_keys(state_dict, prefix + 'h2.conv2.norm.running_mean'),
            filter_state_dict_keys(state_dict, prefix + 'h2.conv2.norm.running_var'),
            # filter_state_dict_keys(state_dict, prefix + 'h2.conv2.norm.op_scale')
        )
        self.h3.convbnrelu.init(
            filter_state_dict_keys(state_dict, prefix + 'h3.conv.weight'),
            filter_state_dict_keys(state_dict, prefix + 'h3.norm.weight'),
            filter_state_dict_keys(state_dict, prefix + 'h3.norm.bias'),
            filter_state_dict_keys(state_dict, prefix + 'h3.norm.running_mean'),
            filter_state_dict_keys(state_dict, prefix + 'h3.norm.running_var'),
            # filter_state_dict_keys(state_dict, prefix + 'h3.norm.op_scale')
        )
        self.h4.conv1.convbnrelu.init(
            filter_state_dict_keys(state_dict, prefix + 'h4.conv1.conv.weight'),
            filter_state_dict_keys(state_dict, prefix + 'h4.conv1.norm.weight'),
            filter_state_dict_keys(state_dict, prefix + 'h4.conv1.norm.bias'),
            filter_state_dict_keys(state_dict, prefix + 'h4.conv1.norm.running_mean'),
            filter_state_dict_keys(state_dict, prefix + 'h4.conv1.norm.running_var'),
            # filter_state_dict_keys(state_dict, prefix + 'h4.conv1.norm.op_scale')
        )
        self.h4.conv2.convbnrelu.init(
            filter_state_dict_keys(state_dict, prefix + 'h4.conv2.conv.weight'),
            filter_state_dict_keys(state_dict, prefix + 'h4.conv2.norm.weight'),
            filter_state_dict_keys(state_dict, prefix + 'h4.conv2.norm.bias'),
            filter_state_dict_keys(state_dict, prefix + 'h4.conv2.norm.running_mean'),
            filter_state_dict_keys(state_dict, prefix + 'h4.conv2.norm.running_var'),
            # filter_state_dict_keys(state_dict, prefix + 'h4.conv2.norm.op_scale')
        )
        self.h5.convbnrelu.init(
            filter_state_dict_keys(state_dict, prefix + 'h5.conv.weight'),
            filter_state_dict_keys(state_dict, prefix + 'h5.norm.weight'),
            filter_state_dict_keys(state_dict, prefix + 'h5.norm.bias'),
            filter_state_dict_keys(state_dict, prefix + 'h5.norm.running_mean'),
            filter_state_dict_keys(state_dict, prefix + 'h5.norm.running_var'),
            # filter_state_dict_keys(state_dict, prefix + 'h5.norm.op_scale')
        )
        self.h6.conv1.convbnrelu.init(
            filter_state_dict_keys(state_dict, prefix + 'h6.conv1.conv.weight'),    
            filter_state_dict_keys(state_dict, prefix + 'h6.conv1.norm.weight'),
            filter_state_dict_keys(state_dict, prefix + 'h6.conv1.norm.bias'),
            filter_state_dict_keys(state_dict, prefix + 'h6.conv1.norm.running_mean'),
            filter_state_dict_keys(state_dict, prefix + 'h6.conv1.norm.running_var'),
            # filter_state_dict_keys(state_dict, prefix + 'h6.conv1.norm.op_scale')
        )

        self.cat1.init()
        self.cat2.init()
        self.cat3.init()
        self.cat4.init()

    def forward(self, x):
        p3, p4, p5 = x
        is_quantized = getattr(p3, 'is_quantized', False) and getattr(p4, 'is_quantized', False) and getattr(p5, 'is_quantized', False)
        
        # Upsample p5 and concatenate with p4
        if is_quantized:
            p5_dequant = torch.dequantize(p5)
            p5_up = torch.nn.functional.interpolate(p5_dequant, scale_factor=2, mode='nearest')
            p4_dequant = torch.dequantize(p4)
            
            # Quantize the upsampled tensor
            p5_up_quant = torch.quantize_per_tensor(p5_up, tensor_scale(p5_up), 64, dtype=torch.quint8)
            p4_quant = p4  # p4 is already quantized
            
            # Use QCat for concatenation
            combined = self.cat1([p5_up_quant, p4_quant], dim=1)
        else:
            p5_up = torch.nn.functional.interpolate(p5, scale_factor=2, mode='nearest')
            combined = torch.cat([p5_up, p4], 1)
            if not self.is_transition:
                combined = torch.quantize_per_tensor(combined, tensor_scale(combined), 64, dtype=torch.quint8)
        
        h1 = self.h1(combined)
        
        if is_quantized:
            h1_dequant = torch.dequantize(h1)
            h1_up = torch.nn.functional.interpolate(h1_dequant, scale_factor=2, mode='nearest')
            
            # Quantize the upsampled tensor
            h1_up_quant = torch.quantize_per_tensor(h1_up, tensor_scale(h1_up), 64, dtype=torch.quint8)
            p3_quant = p3  # p3 is already quantized
            
            # Use QCat for concatenation
            combined = self.cat2([h1_up_quant, p3_quant], dim=1)
        else:
            h1_up = torch.nn.functional.interpolate(h1, scale_factor=2, mode='nearest')
            combined = torch.cat([h1_up, p3], 1)
            if not self.is_transition:
                combined = torch.quantize_per_tensor(combined, tensor_scale(combined), 64, dtype=torch.quint8)
        
        h2 = self.h2(combined)
        h3 = self.h3(h2)
        
        if is_quantized:
            h1_quant = h1  
            
            combined = self.cat3([h3, h1_quant], dim=1)
        else:
            combined = torch.cat([h3, h1], 1)
            if not self.is_transition:
                combined = torch.quantize_per_tensor(combined, tensor_scale(combined), 64, dtype=torch.quint8)
        
        h4 = self.h4(combined)
        h5 = self.h5(h4)
        
        if is_quantized:
            p5_quant = p5  
            
            # Use QCat for concatenation
            combined = self.cat4([h5, p5_quant], dim=1)
        else:
            combined = torch.cat([h5, p5], 1)
            if not self.is_transition:
                combined = torch.quantize_per_tensor(combined, tensor_scale(combined), 64, dtype=torch.quint8)
        
        h6 = self.h6(combined)
        
        if self.is_transition and is_quantized:
            h2 = torch.dequantize(h2)
            h4 = torch.dequantize(h4)
            h6 = torch.dequantize(h6)
        
        return h2, h4, h6

class QFWDFL(torch.nn.Module):
    def __init__(self, ch=16, is_transition=False):
        super().__init__()
        
        self._register_load_state_dict_pre_hook(self.sd_hook)
        
        self.ch = ch
        self.conv = torch.nn.Conv2d(ch, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(ch, dtype=torch.float).view(1, ch, 1, 1)
        self.conv.weight.data[:] = torch.nn.Parameter(x)
        
        self.is_transition = is_transition
        

        for param in self.parameters():
            param.requires_grad = False
    
    def sd_hook(self, state_dict, prefix, *_):
        key = prefix + 'conv.weight'
        if key in state_dict:
            self.conv.weight.copy_(state_dict[key])
    
    def forward(self, x):
        if getattr(x, 'is_quantized', False):
            x = torch.dequantize(x)
        
        b, c, a = x.shape
        x = x.view(b, 4, self.ch, a).transpose(2, 1)
        x_softmax = torch.nn.functional.softmax(x, dim=1)
        
        return self.conv(x_softmax).view(b, 4, a)

class QFWHead(torch.nn.Module):
    anchors = torch.empty(0)
    strides = torch.empty(0)
    
    def __init__(self, nc=80, filters=(), is_transition=False):
        super().__init__()
        
        self._register_load_state_dict_pre_hook(self.sd_hook)
        
        self.ch = 16  # DFL channels
        self.nc = nc  # number of classes
        self.nl = len(filters)  # number of detection layers
        self.no = nc + self.ch * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build

        c1 = max(filters[0], self.nc)
        c2 = max((filters[0] // 4, self.ch * 4))
        
        self.dfl = QFWDFL(self.ch, is_transition=is_transition)
        
        # Store component instances as direct attributes in regular Python lists
        # Box branch components
        self.box_conv1 = []
        self.box_conv2 = []
        self.box_final = []
        
        # Cls branch components
        self.cls_conv1 = []
        self.cls_conv2 = []
        self.cls_final = []
        
        for i, x in enumerate(filters):
            # Create individual components for box prediction
            self.box_conv1.append(QFWConv(x, c2, 3))
            self.box_conv2.append(QFWConv(c2, c2, 3))
            
            # Create final convolutional layers
            box_final = nn.Conv2d(c2, 4 * self.ch, 1, bias=True)
            box_final.requires_grad_(False)
            self.box_final.append(box_final)
            
            # Similarly for classification branch
            self.cls_conv1.append(QFWConv(x, c1, 3))
            self.cls_conv2.append(QFWConv(c1, c1, 3))
            
            cls_final = nn.Conv2d(c1, self.nc, 1, bias=True)
            cls_final.requires_grad_(False)
            self.cls_final.append(cls_final)
        
        self.stride = torch.zeros(self.nl)  # Will be computed later
        self.is_transition = is_transition
        

        for param in self.parameters():
            param.requires_grad = False
    
    def sd_hook(self, state_dict, prefix, *_):
        for i in range(self.nl):
            self.box_conv1[i].sd_hook(state_dict, prefix + f'box.{i}.0.')
            self.box_conv2[i].sd_hook(state_dict, prefix + f'box.{i}.1.')
            
            # Init the final box conv using weight and bias from state dict
            if f'{prefix}box.{i}.2.weight' in state_dict:
                self.box_final[i].weight.copy_(state_dict[f'{prefix}box.{i}.2.weight'])
            if f'{prefix}box.{i}.2.bias' in state_dict:
                self.box_final[i].bias.copy_(state_dict[f'{prefix}box.{i}.2.bias'])
            
            # Cls branch initialization
            self.cls_conv1[i].sd_hook(state_dict, prefix + f'cls.{i}.0.')
            self.cls_conv2[i].sd_hook(state_dict, prefix + f'cls.{i}.1.')
            
            # Init the final cls conv
            if f'{prefix}cls.{i}.2.weight' in state_dict:
                self.cls_final[i].weight.copy_(state_dict[f'{prefix}cls.{i}.2.weight'])
            if f'{prefix}cls.{i}.2.bias' in state_dict:
                self.cls_final[i].bias.copy_(state_dict[f'{prefix}cls.{i}.2.bias'])

    def forward(self, x):
        for i in range(self.nl):
            if getattr(x[i], 'is_quantized', False):
                x_dequant = torch.dequantize(x[i])
            else:
                x_dequant = x[i]
            
            box_out = self.box_conv1[i](x_dequant)
            box_out = self.box_conv2[i](box_out)
            
            if getattr(box_out, 'is_quantized', False):
                box_out = torch.dequantize(box_out)
            box_out = self.box_final[i](box_out)
            
            cls_out = self.cls_conv1[i](x_dequant)
            cls_out = self.cls_conv2[i](cls_out)
            
            # Dequantize for the final layer 
            if getattr(cls_out, 'is_quantized', False):
                cls_out = torch.dequantize(cls_out)
            cls_out = self.cls_final[i](cls_out)
            
            x[i] = torch.cat([box_out, cls_out], 1)
        
        if self.training:
            return x
            
        # Inference
        self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))

        x = torch.cat([i.view(x[0].shape[0], self.no, -1) for i in x], 2)
        box, cls = x.split((self.ch * 4, self.nc), 1)
        
        # Apply DFL
        dfl_out = self.dfl(box)
        a, b = torch.split(dfl_out, 2, 1)
        a = self.anchors.unsqueeze(0) - a
        b = self.anchors.unsqueeze(0) + b
        box = torch.cat(((a + b) / 2, b - a), 1)
        
        return torch.cat((box * self.strides, cls.sigmoid()), 1)
    
    def initialize_biases(self):
        for i, s in enumerate(self.stride):
            self.box_final[i].bias.data[:] = 1.0  # box
            self.cls_final[i].bias.data[:self.nc] = math.log(5 / self.nc / (640 / s) ** 2)  # cls
            
    def to(self, device):
        # Call parent to_device first and manually move list components
        super().to(device)
        for i in range(self.nl):
            self.box_conv1[i] = self.box_conv1[i].to(device)
            self.box_conv2[i] = self.box_conv2[i].to(device)
            self.box_final[i] = self.box_final[i].to(device)
            
            self.cls_conv1[i] = self.cls_conv1[i].to(device)
            self.cls_conv2[i] = self.cls_conv2[i].to(device)
            self.cls_final[i] = self.cls_final[i].to(device)
        
        return self

# Helper function
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