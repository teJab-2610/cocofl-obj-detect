import argparse
import copy
import os
import pickle
import socket
import struct
import sys
import time
import uuid
import yaml

import numpy as np
import torch
import tqdm
from torch.utils import data

from nets import nn
from utils import util
from utils.dataset import Dataset

# Configure logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("client.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("FL_Client")


class FederatedClient:
    def __init__(self, args, params):
        self.args = args
        self.params = params
        self.num_classes = len(params['names'])
        self.server_address = args.server_address
        self.server_port = args.server_port
        self.client_id = args.client_id or str(uuid.uuid4())[:8]
        self.local_epochs = args.local_epochs
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize model (will be replaced with server model)
        self.init_model()
        
        # Load datasets
        self.load_datasets()
        
        # Create directory for saving local results
        os.makedirs(args.save_dir, exist_ok=True)
        
        logger.info(f"Client {self.client_id} initialized")
        logger.info(f"Using device: {self.device}")
        logger.info(f"Number of classes: {self.num_classes}")
    
    def init_model(self):
        from nets_quantized import nn as nn_quantized
        """Initialize the model structure based on model size"""
        if self.args.model_size == 'n':
            self.model = nn.yolo_v8_n(self.num_classes).to(self.device)
        elif self.args.model_size == 's':
            self.model = nn.yolo_v8_s(self.num_classes).to(self.device)
        elif self.args.model_size == 'm':
            self.model = nn.yolo_v8_m(self.num_classes).to(self.device)
        elif self.args.model_size == 'l':
            self.model = nn.yolo_v8_l(self.num_classes).to(self.device)
        elif self.args.model_size == 'x':
            self.model = nn.yolo_v8_x(self.num_classes).to(self.device)
        elif self.args.model_size == 'coco-n':
            self.model = nn_quantized.QYOLOv8n(self.num_classes).to(self.device)
        elif self.args.model_size == 'coco-s':
            self.model = nn_quantized.QYOLOv8s(self.num_classes).to(self.device)
        elif self.args.model_size == 'coco-m':
            self.model = nn_quantized.QYOLOv8m(self.num_classes).to(self.device)
        elif self.args.model_size == 'coco-l':
            self.model = nn_quantized.QYOLOv8l(self.num_classes).to(self.device)
        elif self.args.model_size == 'coco-x':
            self.model = nn_quantized.QYOLOv8x(self.num_classes).to(self.device)    
        
        else:
            logger.error(f"Invalid model size: {self.args.model_size}")
            raise ValueError(f"Invalid model size: {self.args.model_size}")
        
    
    def load_datasets(self):
        """Load training and validation datasets"""
        # Load training filenames
        train_filenames = []
        with open(self.args.train_file) as reader:
            for filename in reader.readlines():
                filename = filename.rstrip()
                if os.path.exists(filename):
                    train_filenames.append(filename)
        
        if not train_filenames:
            logger.error(f"No training files found in {self.args.train_file}")
            raise FileNotFoundError(f"No training files found in {self.args.train_file}")
        
        logger.info(f"Loaded {len(train_filenames)} training images")
        
        # Load validation filenames
        val_filenames = []
        with open(self.args.val_file) as reader:
            for filename in reader.readlines():
                filename = filename.rstrip()
                if os.path.exists(filename):
                    val_filenames.append(filename)
        
        if not val_filenames:
            logger.error(f"No validation files found in {self.args.val_file}")
            raise FileNotFoundError(f"No validation files found in {self.args.val_file}")
        
        logger.info(f"Loaded {len(val_filenames)} validation images")
        
        # Create training dataset and loader
        self.train_dataset = Dataset(train_filenames, self.args.input_size, self.params, True)
        self.train_loader = data.DataLoader(
            self.train_dataset, 
            self.args.batch_size, 
            shuffle=True, 
            num_workers=4,
            pin_memory=True, 
            collate_fn=Dataset.collate_fn
        )
        # Create validation dataset and loader
        self.val_dataset = Dataset(val_filenames, self.args.input_size, self.params, False)
        self.val_loader = data.DataLoader(
            self.val_dataset, 
            self.args.batch_size, 
            False, 
            num_workers=4,
            pin_memory=True, 
            collate_fn=Dataset.collate_fn
        )
    
    def receive_data(self, conn):
        """Receive data with size header"""
        data_size_bytes = conn.recv(4)
        if not data_size_bytes:
            return None
        data_size = struct.unpack('!I', data_size_bytes)[0]
        
        # Receive data in chunks
        chunks = []
        bytes_received = 0
        while bytes_received < data_size:
            chunk = conn.recv(min(data_size - bytes_received, 4096))
            if not chunk:
                raise ConnectionError("Connection closed while receiving data")
            chunks.append(chunk)
            bytes_received += len(chunk)
        
        return b''.join(chunks)
    
    def send_data(self, conn, data):
        """Send data with size header"""
        data_size = len(data)
        conn.sendall(struct.pack('!I', data_size))
        conn.sendall(data)
    
    def connect_to_server(self):
        """Connect to the federated learning server
        
        Returns:
            tuple: (socket, current_round) - socket object connected to server, and current round number
                Returns (None, 0) on failure
        """
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((self.server_address, self.server_port))
            logger.info(f"Connected to server at {self.server_address}:{self.server_port}")
            
            # Send client ID
            self.send_data(sock, pickle.dumps(self.client_id))
            
            # Receive round number from server
            server_data = self.receive_data(sock)
            if not server_data:
                logger.error("Failed to receive data from server")
                sock.close()
                return None, 0
                
            current_round = pickle.loads(server_data)
            logger.info(f"Server says we're on round: {current_round}")
            
            # Handle special round numbers that indicate client should wait/retry
            if current_round in (-1, -2):
                # Don't close the socket here, let the participate method handle it
                # This way we can communicate the special status back to the main loop
                return sock, current_round
            
            # Receive global model
            server_data = self.receive_data(sock)
            if not server_data:
                logger.error("Failed to receive global model from server")
                sock.close()
                return None, 0
                
            global_model_state = pickle.loads(server_data)
            # Load global model parameters
            self.model.load_state_dict(global_model_state)
            logger.info("Global model received and loaded")
            
            return sock, current_round
        
        except Exception as e:
            logger.error(f"Error connecting to server: {e}")
            return None, 0

    def train_local_model(self, current_round):
        """Train the model on local data"""
        logger.info(f"Starting local training for {self.local_epochs} epochs")
        # Initialize optimizer and learning rate scheduler
        optimizer = self.setup_optimizer()
        scheduler = self.setup_scheduler(optimizer)
        
        # EMA model for evaluation
        ema = util.EMA(self.model)
        
        # Loss function
        criterion = util.ComputeLoss(self.model, self.params)
        
        # Setup AMP scaler for mixed precision training
        amp_scale = torch.amp.GradScaler()
        
        # Track metrics
        metrics = {
            'train_loss': [],
            'val_map50': [],
            'val_map': []
        }
        
        # Start training
        best_map = 0
        for epoch in range(self.local_epochs):
            self.model.train()
            epoch_loss = util.AverageMeter()
            
            # Progress bar for training
            p_bar = tqdm.tqdm(enumerate(self.train_loader), total=len(self.train_loader))
            p_bar.set_description(f"Round {current_round}, Epoch {epoch+1}/{self.local_epochs}")
            
            for i, (samples, targets, _) in p_bar:
                samples = samples.to(self.device).float() / 255
                targets = targets.to(self.device)
                
                # Forward pass with mixed precision
                with torch.cuda.amp.autocast():
                    outputs = self.model(samples)
                    loss = criterion(outputs, targets)
                
                # Update loss tracking
                epoch_loss.update(loss.item(), samples.size(0))
                
                # Backward pass with gradient scaling
                optimizer.zero_grad()
                amp_scale.scale(loss).backward()
                amp_scale.unscale_(optimizer)
                util.clip_gradients(self.model)
                amp_scale.step(optimizer)
                amp_scale.update()
                
                # Update EMA model
                ema.update(self.model)
                
                # Update progress bar
                p_bar.set_postfix({'loss': f'{epoch_loss.avg:.4f}'})
            
            # Step scheduler
            scheduler.step()
            
            # Save average epoch loss
            metrics['train_loss'].append(epoch_loss.avg)
            
            # Evaluate model
            if (epoch + 1) % self.args.eval_interval == 0 or epoch == self.local_epochs - 1:
                map50, mean_ap = self.evaluate_model(ema.ema)
                metrics['val_map50'].append(map50)
                metrics['val_map'].append(mean_ap)
                
                # Save best model
                if mean_ap > best_map:
                    best_map = mean_ap
                    self.save_model(ema.ema, current_round, epoch, is_best=True)
                
                logger.info(f"Epoch {epoch+1}/{self.local_epochs}, Loss: {epoch_loss.avg:.4f}, "
                        f"mAP50: {map50:.4f}, mAP: {mean_ap:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % self.args.save_interval == 0:
                self.save_model(ema.ema, current_round, epoch)
        
        # Get the dataset size
        dataset_size = len(self.train_dataset)
        
        # Return the EMA model, metrics, and dataset size
        return ema.ema.state_dict(), metrics, dataset_size

    
    def setup_optimizer(self):
        """Setup model optimizer"""
        # Split parameters into groups for different weight decays
        p = [], [], []
        for v in self.model.modules():
            if hasattr(v, 'bias') and isinstance(v.bias, torch.nn.Parameter):
                p[2].append(v.bias)
            if isinstance(v, torch.nn.BatchNorm2d):
                p[1].append(v.weight)
            elif hasattr(v, 'weight') and isinstance(v.weight, torch.nn.Parameter):
                p[0].append(v.weight)
        
        # Calculate weight decay adjustment based on batch size
        weight_decay = self.params['weight_decay'] * self.args.batch_size / 64
        
        # Create SGD optimizer with parameter groups
        optimizer = torch.optim.SGD(p[2], self.params['lr0'], self.params['momentum'], nesterov=True)
        optimizer.add_param_group({'params': p[0], 'weight_decay': weight_decay})
        optimizer.add_param_group({'params': p[1]})
        
        return optimizer
    
    def setup_scheduler(self, optimizer):
        """Setup learning rate scheduler"""
        # Define linear decay function
        def lr_fn(x):
            return (1 - x / self.local_epochs) * (1.0 - self.params['lrf']) + self.params['lrf']
        
        # Create LambdaLR scheduler
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_fn, last_epoch=-1)
        
        return scheduler
    
    @torch.no_grad()
    def evaluate_model(self, model):
        """Evaluate the model on validation data"""
        model.eval()
        model.half()  # Convert to half precision for evaluation
        
        # Configure
        iou_v = torch.linspace(0.5, 0.95, 10).to(self.device)  # iou vector for mAP@0.5:0.95
        n_iou = iou_v.numel()
        
        metrics = []
        for samples, targets, shapes in self.val_loader:
            samples = samples.to(self.device)
            targets = targets.to(self.device)
            samples = samples.half()  # uint8 to fp16/32
            samples = samples / 255  # 0 - 255 to 0.0 - 1.0
            _, _, height, width = samples.shape
            
            # Inference
            outputs = model(samples)
            
            # NMS
            targets[:, 2:] *= torch.tensor((width, height, width, height)).to(self.device)
            outputs = util.non_max_suppression(outputs, 0.001, 0.65)
            
            # Metrics
            for i, output in enumerate(outputs):
                labels = targets[targets[:, 0] == i, 1:]
                correct = torch.zeros(output.shape[0], n_iou, dtype=torch.bool).to(self.device)
                
                if output.shape[0] == 0:
                    if labels.shape[0]:
                        metrics.append((correct, *torch.zeros((3, 0)).to(self.device)))
                    continue
                
                detections = output.clone()
                util.scale(detections[:, :4], samples[i].shape[1:], shapes[i][0], shapes[i][1])
                
                # Evaluate
                if labels.shape[0]:
                    tbox = labels[:, 1:5].clone()
                    tbox[:, 0] = labels[:, 1] - labels[:, 3] / 2  # top left x
                    tbox[:, 1] = labels[:, 2] - labels[:, 4] / 2  # top left y
                    tbox[:, 2] = labels[:, 1] + labels[:, 3] / 2  # bottom right x
                    tbox[:, 3] = labels[:, 2] + labels[:, 4] / 2  # bottom right y
                    util.scale(tbox, samples[i].shape[1:], shapes[i][0], shapes[i][1])
                    
                    correct = np.zeros((detections.shape[0], iou_v.shape[0]))
                    correct = correct.astype(bool)
                    
                    t_tensor = torch.cat((labels[:, 0:1], tbox), 1)
                    iou = util.box_iou(t_tensor[:, 1:], detections[:, :4])
                    correct_class = t_tensor[:, 0:1] == detections[:, 5]
                    for j in range(len(iou_v)):
                        x = torch.where((iou >= iou_v[j]) & correct_class)
                        if x[0].shape[0]:
                            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1)
                            matches = matches.cpu().numpy()
                            if x[0].shape[0] > 1:
                                matches = matches[matches[:, 2].argsort()[::-1]]
                                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                            correct[matches[:, 1].astype(int), j] = True
                    correct = torch.tensor(correct, dtype=torch.bool, device=iou_v.device)
                metrics.append((correct, output[:, 4], output[:, 5], labels[:, 0]))
        
        # Compute metrics
        metrics = [torch.cat(x, 0).cpu().numpy() for x in zip(*metrics)]  # to numpy
        
        map50, mean_ap = 0.0, 0.0
        if len(metrics) and metrics[0].any():
            tp, fp, precision, recall, map50, mean_ap = util.compute_ap(*metrics)
        
        # Convert back to float for training
        model.float()
        
        return map50, mean_ap
    
    def save_model(self, model, round_num, epoch, is_best=False):
        """Save the model checkpoint"""
        if is_best:
            save_path = os.path.join(self.args.save_dir, f"client_{self.client_id}_round_{round_num}_best.pt")
        else:
            save_path = os.path.join(self.args.save_dir, 
                                    f"client_{self.client_id}_round_{round_num}_epoch_{epoch+1}.pt")
        
        torch.save({
            'model': model.state_dict(),
            'round': round_num,
            'epoch': epoch,
            'args': self.args,
        }, save_path)
        
        if is_best:
            logger.info(f"Best model saved to {save_path}")
    
    def participate(self):
        """Participate in the federated learning process
        
        Returns:
            tuple: (success, current_round) - success is a boolean indicating if participation was successful,
                                            current_round is the round number that was just completed
        """
        try:
            # Connect to the server
            sock, current_round = self.connect_to_server()
            if sock is None:
                logger.error("Failed to connect to server")
                return False, 0
            
            # Check if we got a special round number that indicates we should retry later
            if current_round in (-1, -2):
                # Server is busy or we already participated in this round
                reason = "Server is busy" if current_round == -1 else "Already participated in this round"
                logger.info(f"{reason}. Will retry later.")
                sock.close()
                return False, 0
            
            # Train the model on local data
            model_state, metrics, dataset_size = self.train_local_model(current_round)
            
            # Send the trained model back to the server along with dataset size
            self.send_data(sock, pickle.dumps((model_state, metrics, dataset_size)))
            logger.info(f"Trained model sent to server (dataset size: {dataset_size})")
            
            # Close connection
            sock.close()
            logger.info("Connection closed")
            
            return True, current_round
        
        except Exception as e:
            logger.error(f"Error during federated learning participation: {e}")
            return False, 0

def main():
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    #os.getlogin()
    parser = argparse.ArgumentParser(description="Federated Learning Client for YOLOv8")
    parser.add_argument('--server-address', type=str, default='10.23.105.40',
                        help='Server address to connect to')
    parser.add_argument('--server-port', type=int, default=5000,
                        help='Server port to connect to')
    import os

    parser.add_argument('--client-id', type=str, default=os.getlogin(), 
                        help='Client ID (default: system login name)')

    parser.add_argument('--input-size', type=int, default=640,
                        help='Input image size')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size for training and evaluation')
    parser.add_argument('--local-epochs', type=int, default=5,
                        help='Number of epochs to train locally per round')
    parser.add_argument('--model-size', type=str, default='n', 
                        choices=['n', 's', 'm', 'l', 'x', 'coco-n', 'coco-s', 'coco-m', 'coco-l', 'coco-x'],
                        help='YOLOv8 model size: n(ano), s(mall), m(edium), l(arge), x(large)')
    parser.add_argument('--train-file', type=str, default='/home/ssl40/cs21b048_37_dl/raw_datasets/soda10m/new_data2/train/client_1/train.txt',
                        help='Path to training image list')
    parser.add_argument('--val-file', type=str, default='/home/ssl40/cs21b048_37_dl/raw_datasets/soda10m/new_data2/train/client_1/val.txt',
                        help='Path to validation image list')
    parser.add_argument('--save-dir', type=str, default='',
                        help='Directory to save local models and metrics')
    parser.add_argument('--eval-interval', type=int, default=5,
                        help='Interval for model evaluation during training')
    parser.add_argument('--save-interval', type=int, default=5,
                        help='Interval for saving model checkpoints')
    parser.add_argument('--retry-interval', type=int, default=60,
                        help='Interval between connection retries (seconds)')
    parser.add_argument('--max-retries', type=int, default=5,
                        help='Maximum number of connection retries')
    parser.add_argument('--rounds', type=int, default=50,
                        help='Maximum number of rounds to participate in (default: 10)')
    args = parser.parse_args()
    
    # Load parameter configuration
    with open(os.path.join('utils', 'args.yaml'), errors='ignore') as f:
        params = yaml.safe_load(f)
    if args.save_dir == '':
        args.save_dir = os.path.join('fl_client_results_final', f'client_{args.client_id}_{timestamp}')
    else:
        args.save_dir = os.path.join(args.save_dir, f'client_{args.client_id}_{timestamp}')
    os.makedirs(args.save_dir, exist_ok=True)
    logger.info(f"Results will be saved in {args.save_dir}")
    logger.info(f"Client ID: {args.client_id}")
    with open(os.path.join(args.save_dir, 'args.yaml'), 'w') as f:
        yaml.dump(vars(args), f, default_flow_style=False)
    logger.info(f"Arguments saved to {os.path.join(args.save_dir, 'args.yaml')}")

    # Load parameter configuration
    with open(os.path.join('utils', 'args.yaml'), errors='ignore') as f:
        params = yaml.safe_load(f)
    
    # Initialize client
    client = FederatedClient(args, params)
    
    # Continue participating in rounds until maximum rounds reached
    current_round = 0
    max_rounds = args.rounds
    
    while current_round < max_rounds:
        logger.info(f"Attempting to participate in round {current_round+1}/{max_rounds}")
        
        # Retry logic for connection
        retries = 0
        success = False
        
        while retries < args.max_retries and not success:
            if retries > 0:
                logger.info(f"Retrying connection ({retries}/{args.max_retries}) in {args.retry_interval} seconds...")
                time.sleep(args.retry_interval)
                
            success, new_round = client.participate()
            
            if success:
                current_round = new_round
                logger.info(f"Successfully participated in round {current_round}")
                # Small delay before attempting next round
                time.sleep(5)
            else:
                retries += 1
        
        if not success:
            logger.error(f"Failed to participate after {args.max_retries} attempts. Waiting longer before next try...")
            time.sleep(args.retry_interval * 2)
    
    logger.info(f"Completed participation in {max_rounds} rounds of federated learning")


if __name__ == "__main__":
    main()
