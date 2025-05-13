# import torch
# import random
# import json
# import math
# import logging
# import traceback

# from nets_quantized.nn_training import Conv, Residual, CSP, SPP, DarkNet, DarkFPN, DFL, Head
# from nets_quantized.nn_forward import QFWConv, QFWResidual, QFWCSP, QFWSPP, QFWDarkNet, QFWDarkFPN, QFWHead
# from nets_quantized.nn_backward import QBWConv, QBWResidual, QBWCSP, QBWSPP, QBWDarkNet, QBWDarkFPN, QBWHead
# from nets_quantized.utils.utils import filter_table

# # Configure logging
# logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [%(levelname)s] %(message)s')

# # Placeholder profiling table - would be populated in actual implementation
# # with open('nets/nets_quantized/YOLO/tables/table__CoCoFL_x64_QYOLO.json', 'r') as fd:
# #     _g_table_qyolo = json.load(fd)
# _g_table_qyolo = {}
# img_dummy = torch.zeros(1, 3, 640, 640)

# class QYOLO(torch.nn.Module):
#     def __init__(self, trained_block_list, fw_block_list, bw_block_list, width, depth, num_classes, freeze_idxs=[]):
#         super(QYOLO, self).__init__()
#         try:
#             logging.info("Initializing QYOLO model")
#             # Define layer indices
#             layer_idxs = [i for i in range(3)]  # Simplified - would be more in actual model
#             self.max_idx = 2  # 3 layers total: net, fpn, head

#             if not set(freeze_idxs) <= set(layer_idxs):
#                 raise ValueError("Invalid layer idxs provided for freezing.")

#             logging.info(f"Freeze indices: {freeze_idxs}")
#             self._trained_idxs = [idx for idx in layer_idxs if idx not in freeze_idxs]
#             logging.info(f"Trained indices: {self._trained_idxs}")
            
#             # Check for continuous block of trained layers
#             expected_trained = set([i for i in range(max(self._trained_idxs) + 1 - len(self._trained_idxs), max(self._trained_idxs) + 1)])
#             if set(self._trained_idxs) != expected_trained:
#                 raise ValueError("No continuous block of trained layers")
            
#             # Unpack block types from lists
#             self._trained_conv, self._trained_residual, self._trained_csp, self._trained_spp, \
#             self._trained_darknet, self._trained_darkfpn, self._trained_head = trained_block_list

#             self._fw_conv, self._fw_residual, self._fw_csp, self._fw_spp, \
#             self._fw_darknet, self._fw_darkfpn, self._fw_head = fw_block_list

#             self._bw_conv, self._bw_residual, self._bw_csp, self._bw_spp, \
#             self._bw_darknet, self._bw_darkfpn, self._bw_head = bw_block_list

#             self._block_idx = 0

#             self.net = None
#             self.fpn = None
#             self.head = None

#             # Create net component (DarkNet)
#             if self._block_idx < min(self._trained_idxs):
#                 transition = True if (self._block_idx + 1) == min(self._trained_idxs) else False
#                 logging.info(f"Initializing net with forward block. Block idx: {self._block_idx}, transition: {transition}")
#                 self.net = self._fw_darknet(width, depth, is_transition=transition)
#             elif self._block_idx > max(self._trained_idxs):
#                 logging.info(f"Initializing net with backward block. Block idx: {self._block_idx}")
#                 self.net = self._bw_darknet(width, depth)
#             elif self._block_idx in self._trained_idxs:
#                 is_first = True if self._block_idx == min(self._trained_idxs) else False
#                 logging.info(f"Initializing net with trained block. Block idx: {self._block_idx}, is_first: {is_first}")
#                 self.net = self._trained_darknet(width, depth, is_first=is_first)
#             else:
#                 raise RuntimeError("Unexpected block index for net component")
            
#             self._block_idx += 1

#             # Create FPN component (DarkFPN)
#             if self._block_idx < min(self._trained_idxs):
#                 transition = True if (self._block_idx + 1) == min(self._trained_idxs) else False
#                 logging.info(f"Initializing FPN with forward block. Block idx: {self._block_idx}, transition: {transition}")
#                 self.fpn = self._fw_darkfpn(width, depth, is_transition=transition)
#             elif self._block_idx > max(self._trained_idxs):
#                 logging.info(f"Initializing FPN with backward block. Block idx: {self._block_idx}")
#                 self.fpn = self._bw_darkfpn(width, depth)
#             elif self._block_idx in self._trained_idxs:
#                 is_first = True if self._block_idx == min(self._trained_idxs) else False
#                 logging.info(f"Initializing FPN with trained block. Block idx: {self._block_idx}, is_first: {is_first}")
#                 self.fpn = self._trained_darkfpn(width, depth, is_first=is_first)
#             else:
#                 raise RuntimeError("Unexpected block index for FPN component")
            
#             self._block_idx += 1

#             # Create Head component
#             # Note: A dummy image is used if forward information is needed during initialization.
#             logging.info(f"Initializing head component. Block idx: {self._block_idx}")
#             if self._block_idx < min(self._trained_idxs):
#                 transition = True if (self._block_idx + 1) == min(self._trained_idxs) else False
#                 logging.info(f"Initializing head with forward block. Transition: {transition}")
#                 self.head = self._fw_head(num_classes, (width[3], width[4], width[5]), is_transition=transition)
#             elif self._block_idx > max(self._trained_idxs):
#                 logging.info("Initializing head with backward block")
#                 self.head = self._bw_head(num_classes, (width[3], width[4], width[5]))
#             elif self._block_idx in self._trained_idxs:
#                 is_first = True if self._block_idx == min(self._trained_idxs) else False
#                 logging.info(f"Initializing head with trained block. is_first: {is_first}")
#                 self.head = self._trained_head(num_classes, (width[3], width[4], width[5]), is_first=is_first)
#             else:
#                 raise RuntimeError("Unexpected block index for head component")
            
#             # Get strides via a forward pass with a dummy image
#             logging.info("Performing forward pass with dummy image to compute strides")
#             outputs = self.forward(img_dummy)
#             strides_computed = [640 / x.shape[-2] for x in outputs]
#             self.head.stride = torch.tensor(strides_computed)
#             self.stride = self.head.stride
#             logging.info(f"Computed strides: {self.head.stride}")
            
#             # Initialize head biases
#             logging.info("Initializing head biases")
#             self.head.initialize_biases()

#             logging.info("QYOLO model initialization completed successfully.")
#         except Exception as e:
#             logging.error("An error occurred during QYOLO initialization:")
#             logging.error(traceback.format_exc())
#             raise

#     def forward(self, x):
#         try:
#             logging.info("Starting forward pass through net")
#             x = self.net(x)
#             logging.info("Completed net forward pass")
            
#             logging.info("Starting forward pass through FPN")
#             x = self.fpn(x)
#             logging.info("Completed FPN forward pass")
            
#             # Ensure x is in list form when passing to head
#             if not isinstance(x, list):
#                 x = [x]
            
#             logging.info("Starting forward pass through head")
#             head_out = self.head(list(x))
#             logging.info("Completed head forward pass")
#             return head_out
#         except Exception as e:
#             logging.error("An error occurred during the forward pass:")
#             logging.error(traceback.format_exc())
#             raise
        
# class QYOLOv8n(QYOLO):
#     def __init__(self, num_classes=80, freeze=[]):
#         depth = [1, 2, 2]
#         width = [3, 16, 32, 64, 128, 256]
#         trained_block_list = [Conv, Residual, CSP, SPP, DarkNet, DarkFPN, Head]
#         fw_block_list = [QFWConv, QFWResidual, QFWCSP, QFWSPP, QFWDarkNet, QFWDarkFPN, QFWHead]
#         bw_block_list = [QBWConv, QBWResidual, QBWCSP, QBWSPP, QBWDarkNet, QBWDarkFPN, QBWHead]
#         super(QYOLOv8n, self).__init__(trained_block_list, fw_block_list, bw_block_list, 
#                                       width, depth, num_classes, freeze_idxs=freeze)
    
#     @staticmethod
#     def n_freezable_layers():
#         return 3  # Simplified - would be more granular in actual implementation
    
#     @staticmethod
#     def get_freezing_config(resources):
#         configs = filter_table(resources, _g_table_qyolo, QYOLOv8n.n_freezable_layers())
#         return random.choice(configs)
    
#     @staticmethod   
#     def get_freezing_config2():
#         # return random.choice([],[0],[1],[2],[0,1],[0,2],[1,2],[0,1,2])
#         choices = [[], [0], [1], [2], [0, 1], [1, 2], [0, 1, 2]]

#         # Randomly select one of the choices
#         selected_choice = random.choice(choices)
#         # return selected_choice
#         return [0,2]

# class QYOLOv8s(QYOLO):
#     def __init__(self, num_classes=80, freeze=[]):
#         depth = [1, 2, 2]
#         width = [3, 32, 64, 128, 256, 512]
#         trained_block_list = [Conv, Residual, CSP, SPP, DarkNet, DarkFPN, Head]
#         fw_block_list = [QFWConv, QFWResidual, QFWCSP, QFWSPP, QFWDarkNet, QFWDarkFPN, QFWHead]
#         bw_block_list = [QBWConv, QBWResidual, QBWCSP, QBWSPP, QBWDarkNet, QBWDarkFPN, QBWHead]
        
#         #print("QYOLOv8s initialization started")
#         super(QYOLOv8s, self).__init__(trained_block_list, fw_block_list, bw_block_list, 
#                                       width, depth, num_classes, freeze_idxs=freeze)
#         #print("QYOLOv8s initialization finished")
#     @staticmethod
#     def n_freezable_layers():
#         return 3
    
#     @staticmethod
#     def get_freezing_config(resources):
#         configs = filter_table(resources, _g_table_qyolo, QYOLOv8s.n_freezable_layers())
#         return random.choice(configs)


# class QYOLOv8m(QYOLO):
#     def __init__(self, num_classes=80, freeze=[]):
#         depth = [2, 4, 4]
#         width = [3, 48, 96, 192, 384, 576]
#         trained_block_list = [Conv, Residual, CSP, SPP, DarkNet, DarkFPN, Head]
#         fw_block_list = [QFWConv, QFWResidual, QFWCSP, QFWSPP, QFWDarkNet, QFWDarkFPN, QFWHead]
#         bw_block_list = [QBWConv, QBWResidual, QBWCSP, QBWSPP, QBWDarkNet, QBWDarkFPN, QBWHead]
#         super(QYOLOv8m, self).__init__(trained_block_list, fw_block_list, bw_block_list, 
#                                       width, depth, num_classes, freeze_idxs=freeze)
    
#     @staticmethod
#     def n_freezable_layers():
#         return 3
    
#     @staticmethod
#     def get_freezing_config(resources):
#         configs = filter_table(resources, _g_table_qyolo, QYOLOv8m.n_freezable_layers())
#         return random.choice(configs)


# class QYOLOv8l(QYOLO):
#     def __init__(self, num_classes=80, freeze=[]):
#         depth = [3, 6, 6]
#         width = [3, 64, 128, 256, 512, 512]
#         trained_block_list = [Conv, Residual, CSP, SPP, DarkNet, DarkFPN, Head]
#         fw_block_list = [QFWConv, QFWResidual, QFWCSP, QFWSPP, QFWDarkNet, QFWDarkFPN, QFWHead]
#         bw_block_list = [QBWConv, QBWResidual, QBWCSP, QBWSPP, QBWDarkNet, QBWDarkFPN, QBWHead]
#         super(QYOLOv8l, self).__init__(trained_block_list, fw_block_list, bw_block_list, 
#                                       width, depth, num_classes, freeze_idxs=freeze)
    
#     @staticmethod
#     def n_freezable_layers():
#         return 3
    
#     @staticmethod
#     def get_freezing_config(resources):
#         configs = filter_table(resources, _g_table_qyolo, QYOLOv8l.n_freezable_layers())
#         return random.choice(configs)


# class QYOLOv8x(QYOLO):
#     def __init__(self, num_classes=80, freeze=[]):
#         depth = [3, 6, 6]
#         width = [3, 80, 160, 320, 640, 640]
#         trained_block_list = [Conv, Residual, CSP, SPP, DarkNet, DarkFPN, Head]
#         fw_block_list = [QFWConv, QFWResidual, QFWCSP, QFWSPP, QFWDarkNet, QFWDarkFPN, QFWHead]
#         bw_block_list = [QBWConv, QBWResidual, QBWCSP, QBWSPP, QBWDarkNet, QBWDarkFPN, QBWHead]
#         super(QYOLOv8x, self).__init__(trained_block_list, fw_block_list, bw_block_list, 
#                                       width, depth, num_classes, freeze_idxs=freeze)
    
#     @staticmethod
#     def n_freezable_layers():
#         return 3
    
#     @staticmethod
#     def get_freezing_config(resources):
#         configs = filter_table(resources, _g_table_qyolo, QYOLOv8x.n_freezable_layers())
#         return random.choice(configs)
    

import torch, time
import random
import json
import math
import logging
import traceback

from nets_quantized.nn_training import Conv, Residual, CSP, SPP, DarkNet, DarkFPN, DFL, Head
from nets_quantized.nn_forward import QFWConv, QFWResidual, QFWCSP, QFWSPP, QFWDarkNet, QFWDarkFPN, QFWHead
from nets_quantized.nn_backward import QBWConv, QBWResidual, QBWCSP, QBWSPP, QBWDarkNet, QBWDarkFPN, QBWHead
from nets_quantized.utils.utils import filter_table

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [%(levelname)s] %(message)s')

# Placeholder profiling table - would be populated in actual implementation
# with open('nets/nets_quantized/YOLO/tables/table__CoCoFL_x64_QYOLO.json', 'r') as fd:
#     _g_table_qyolo = json.load(fd)
_g_table_qyolo = {}
img_dummy = torch.zeros(1, 3, 640, 640)

class QYOLO(torch.nn.Module):
    def __init__(self, trained_block_list, fw_block_list, bw_block_list, width, depth, num_classes, freeze_idxs=[]):
        super(QYOLO, self).__init__()
        try:
            logging.info("Initializing QYOLO model")
            # Define layer indices
            layer_idxs = [i for i in range(3)]  # Simplified - would be more in actual model
            self.max_idx = 2  # 3 layers total: net, fpn, head

            if not set(freeze_idxs) <= set(layer_idxs):
                raise ValueError("Invalid layer idxs provided for freezing.")

            logging.info(f"Freeze indices: {freeze_idxs}")
            self._trained_idxs = [idx for idx in layer_idxs if idx not in freeze_idxs]
            logging.info(f"Trained indices: {self._trained_idxs}")
            
            # Check for continuous block of trained layers if there are any trained indices
            if self._trained_idxs:
                expected_trained = set([i for i in range(min(self._trained_idxs), max(self._trained_idxs) + 1)])
                if set(self._trained_idxs) != expected_trained:
                    raise ValueError("No continuous block of trained layers")
            
            # Unpack block types from lists
            self._trained_conv, self._trained_residual, self._trained_csp, self._trained_spp, \
            self._trained_darknet, self._trained_darkfpn, self._trained_head = trained_block_list

            self._fw_conv, self._fw_residual, self._fw_csp, self._fw_spp, \
            self._fw_darknet, self._fw_darkfpn, self._fw_head = fw_block_list

            self._bw_conv, self._bw_residual, self._bw_csp, self._bw_spp, \
            self._bw_darknet, self._bw_darkfpn, self._bw_head = bw_block_list

            self._block_idx = 0

            self.net = None
            self.fpn = None
            self.head = None

            # Create net component (DarkNet)
            if self._trained_idxs and self._block_idx < min(self._trained_idxs):
                transition = True if (self._block_idx + 1) == min(self._trained_idxs) else False
                logging.info(f"Initializing net with forward block. Block idx: {self._block_idx}, transition: {transition}")
                self.net = self._fw_darknet(width, depth, is_transition=transition)
            elif not self._trained_idxs or self._block_idx > max(self._trained_idxs):
                logging.info(f"Initializing net with backward block. Block idx: {self._block_idx}")
                self.net = self._bw_darknet(width, depth)
            elif self._block_idx in self._trained_idxs:
                is_first = True if self._block_idx == min(self._trained_idxs) else False
                logging.info(f"Initializing net with trained block. Block idx: {self._block_idx}, is_first: {is_first}")
                self.net = self._trained_darknet(width, depth, is_first=is_first)
            else:
                raise RuntimeError("Unexpected block index for net component")
            
            self._block_idx += 1

            # Create FPN component (DarkFPN)
            if self._trained_idxs and self._block_idx < min(self._trained_idxs):
                transition = True if (self._block_idx + 1) == min(self._trained_idxs) else False
                logging.info(f"Initializing FPN with forward block. Block idx: {self._block_idx}, transition: {transition}")
                self.fpn = self._fw_darkfpn(width, depth, is_transition=transition)
            elif not self._trained_idxs or self._block_idx > max(self._trained_idxs):
                logging.info(f"Initializing FPN with backward block. Block idx: {self._block_idx}")
                self.fpn = self._bw_darkfpn(width, depth)
            elif self._block_idx in self._trained_idxs:
                is_first = True if self._block_idx == min(self._trained_idxs) else False
                logging.info(f"Initializing FPN with trained block. Block idx: {self._block_idx}, is_first: {is_first}")
                self.fpn = self._trained_darkfpn(width, depth, is_first=is_first)
            else:
                raise RuntimeError("Unexpected block index for FPN component")
            
            self._block_idx += 1

            # Create Head component
            # Note: A dummy image is used if forward information is needed during initialization.
            logging.info(f"Initializing head component. Block idx: {self._block_idx}")
            if self._trained_idxs and self._block_idx < min(self._trained_idxs):
                transition = True if (self._block_idx + 1) == min(self._trained_idxs) else False
                logging.info(f"Initializing head with forward block. Transition: {transition}")
                self.head = self._fw_head(num_classes, (width[3], width[4], width[5]), is_transition=transition)
            elif not self._trained_idxs or self._block_idx > max(self._trained_idxs):
                logging.info("Initializing head with backward block")
                self.head = self._bw_head(num_classes, (width[3], width[4], width[5]))
            elif self._block_idx in self._trained_idxs:
                is_first = True if self._block_idx == min(self._trained_idxs) else False
                logging.info(f"Initializing head with trained block. is_first: {is_first}")
                self.head = self._trained_head(num_classes, (width[3], width[4], width[5]), is_first=is_first)
            else:
                raise RuntimeError("Unexpected block index for head component")
            
            # Get strides via a forward pass with a dummy image
            logging.info("Performing forward pass with dummy image to compute strides")
            # outputs = self.forward(img_dummy)
            # strides_computed = [640 / x.shape[-2] for x in outputs]
            # self.head.stride = torch.tensor(strides_computed)
            # self.stride = self.head.stride

            #TODO TODO TODO
            self.head.stride = [8, 16, 32]
            logging.info(f"Computed strides: {self.head.stride}")
            
            # Initialize head biases
            logging.info("Initializing head biases")
            self.head.initialize_biases()

            logging.info("QYOLO model initialization completed successfully.")
        except Exception as e:
            logging.error("An error occurred during QYOLO initialization:")
            logging.error(traceback.format_exc())
            raise

    def forward(self, x):
        try:
            # logging.info("Starting forward pass through net")
            x = self.net(x)
            # logging.info("Completed net forward pass")
            
            # logging.info("Starting forward pass through FPN")
            x = self.fpn(x)
            # logging.info("Completed FPN forward pass")
            # print("Output of FPN:", type(x), len(x), type(x[0]))
            # Ensure x is in list form when passing to head
            # if not isinstance(x, list):
            #     print("x is not a list but a", type(x))
            #     x = [x]
            
            # logging.info("Starting forward pass through head")
            # print(type(x), len(x), type(x[0]))
            head_out = self.head(list(x))
            # logging.info("Completed head forward pass")
            return head_out
        except Exception as e:
            logging.error("An error occurred during the forward pass:")
            logging.error(traceback.format_exc())
            raise
        
class QYOLOv8n(QYOLO):
    def __init__(self, num_classes=80, freeze=[]):
        depth = [1, 2, 2]
        width = [3, 16, 32, 64, 128, 256]
        trained_block_list = [Conv, Residual, CSP, SPP, DarkNet, DarkFPN, Head]
        fw_block_list = [QFWConv, QFWResidual, QFWCSP, QFWSPP, QFWDarkNet, QFWDarkFPN, QFWHead]
        bw_block_list = [QBWConv, QBWResidual, QBWCSP, QBWSPP, QBWDarkNet, QBWDarkFPN, QBWHead]
        super(QYOLOv8n, self).__init__(trained_block_list, fw_block_list, bw_block_list, 
                                      width, depth, num_classes, freeze_idxs=freeze)
    
    @staticmethod
    def n_freezable_layers():
        return 3  # Simplified - would be more granular in actual implementation
    
    @staticmethod
    def get_freezing_config(resources):
        configs = filter_table(resources, _g_table_qyolo, QYOLOv8n.n_freezable_layers())
        return random.choice(configs)
    
    @staticmethod   
    def get_freezing_config2():
        ##TODO write a script such that it will generate the combinations so the trainable layers are continuous and optimiser has atleast one layer of parameters
        choices = [[], [0], [2], [0, 1], [0, 2], [1, 2]]

        #Set seed to current time   
        random.seed(time.time())
        # Randomly select one of the choices
        selected_choice = random.choice(choices)
        # For testing specific combinations
        return selected_choice

class QYOLOv8s(QYOLO):
    def __init__(self, num_classes=80, freeze=[]):
        depth = [1, 2, 2]
        width = [3, 32, 64, 128, 256, 512]
        trained_block_list = [Conv, Residual, CSP, SPP, DarkNet, DarkFPN, Head]
        fw_block_list = [QFWConv, QFWResidual, QFWCSP, QFWSPP, QFWDarkNet, QFWDarkFPN, QFWHead]
        bw_block_list = [QBWConv, QBWResidual, QBWCSP, QBWSPP, QBWDarkNet, QBWDarkFPN, QBWHead]
        
        #print("QYOLOv8s initialization started")
        super(QYOLOv8s, self).__init__(trained_block_list, fw_block_list, bw_block_list, 
                                      width, depth, num_classes, freeze_idxs=freeze)
        #print("QYOLOv8s initialization finished")
    
    @staticmethod
    def n_freezable_layers():
        return 3
    
    @staticmethod
    def get_freezing_config(resources):
        configs = filter_table(resources, _g_table_qyolo, QYOLOv8s.n_freezable_layers())
        return random.choice(configs)


class QYOLOv8m(QYOLO):
    def __init__(self, num_classes=80, freeze=[]):
        depth = [2, 4, 4]
        width = [3, 48, 96, 192, 384, 576]
        trained_block_list = [Conv, Residual, CSP, SPP, DarkNet, DarkFPN, Head]
        fw_block_list = [QFWConv, QFWResidual, QFWCSP, QFWSPP, QFWDarkNet, QFWDarkFPN, QFWHead]
        bw_block_list = [QBWConv, QBWResidual, QBWCSP, QBWSPP, QBWDarkNet, QBWDarkFPN, QBWHead]
        super(QYOLOv8m, self).__init__(trained_block_list, fw_block_list, bw_block_list, 
                                      width, depth, num_classes, freeze_idxs=freeze)
    
    @staticmethod
    def n_freezable_layers():
        return 3
    
    @staticmethod
    def get_freezing_config(resources):
        configs = filter_table(resources, _g_table_qyolo, QYOLOv8m.n_freezable_layers())
        return random.choice(configs)


class QYOLOv8l(QYOLO):
    def __init__(self, num_classes=80, freeze=[]):
        depth = [3, 6, 6]
        width = [3, 64, 128, 256, 512, 512]
        trained_block_list = [Conv, Residual, CSP, SPP, DarkNet, DarkFPN, Head]
        fw_block_list = [QFWConv, QFWResidual, QFWCSP, QFWSPP, QFWDarkNet, QFWDarkFPN, QFWHead]
        bw_block_list = [QBWConv, QBWResidual, QBWCSP, QBWSPP, QBWDarkNet, QBWDarkFPN, QBWHead]
        super(QYOLOv8l, self).__init__(trained_block_list, fw_block_list, bw_block_list, 
                                      width, depth, num_classes, freeze_idxs=freeze)
    
    @staticmethod
    def n_freezable_layers():
        return 3
    
    @staticmethod
    def get_freezing_config(resources):
        configs = filter_table(resources, _g_table_qyolo, QYOLOv8l.n_freezable_layers())
        return random.choice(configs)


class QYOLOv8x(QYOLO):
    def __init__(self, num_classes=80, freeze=[]):
        depth = [3, 6, 6]
        width = [3, 80, 160, 320, 640, 640]
        trained_block_list = [Conv, Residual, CSP, SPP, DarkNet, DarkFPN, Head]
        fw_block_list = [QFWConv, QFWResidual, QFWCSP, QFWSPP, QFWDarkNet, QFWDarkFPN, QFWHead]
        bw_block_list = [QBWConv, QBWResidual, QBWCSP, QBWSPP, QBWDarkNet, QBWDarkFPN, QBWHead]
        super(QYOLOv8x, self).__init__(trained_block_list, fw_block_list, bw_block_list, 
                                      width, depth, num_classes, freeze_idxs=freeze)
    
    @staticmethod
    def n_freezable_layers():
        return 3
    
    @staticmethod
    def get_freezing_config(resources):
        configs = filter_table(resources, _g_table_qyolo, QYOLOv8x.n_freezable_layers())
        return random.choice(configs)