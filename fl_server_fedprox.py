import argparse
import copy
import os
import pickle
import socket
import struct
import threading
import time
import yaml
from collections import defaultdict

from datetime import datetime

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
        logging.FileHandler("server.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("FL_Server")

class FederatedServer:
    def __init__(self, args, params):
        self.args = args
        self.params = params
        self.num_classes = len(params['names'])
        self.rounds = args.rounds
        self.port = args.port
        self.min_clients = args.min_clients
        self.current_round = 0
        self.unique_clients = set()
        self.client_models = {}
        
        self.client_fedprox_settings = {}
        
        self.metrics = defaultdict(list)
        self.round_in_progress = False  

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.init_global_model()
        self.load_validation_dataset()
        
        os.makedirs(args.save_dir, exist_ok=True)
        os.makedirs(os.path.join(args.save_dir, 'metrics'), exist_ok=True)
        
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(('0.0.0.0', self.port))
        self.sock.listen(10)
        
        logger.info(f"Server initialized. Listening on port {self.port}")
        logger.info(f"Using device: {self.device}")
        logger.info(f"Number of classes: {self.num_classes}")
    
    def init_global_model(self):
        """Initialize the global model based on model size"""
        if self.args.model_size == 'n':
            self.global_model = nn.yolo_v8_n(self.num_classes).to(self.device)
        elif self.args.model_size == 's':
            self.global_model = nn.yolo_v8_s(self.num_classes).to(self.device)
        elif self.args.model_size == 'm':
            self.global_model = nn.yolo_v8_m(self.num_classes).to(self.device)
        elif self.args.model_size == 'l':
            self.global_model = nn.yolo_v8_l(self.num_classes).to(self.device)
        elif self.args.model_size == 'x':
            self.global_model = nn.yolo_v8_x(self.num_classes).to(self.device)
        else:
            logger.error(f"Invalid model size: {self.args.model_size}")
            raise ValueError(f"Invalid model size: {self.args.model_size}")
        
        logger.info(f"Global model initialized: YOLOv8-{self.args.model_size}")
    
    def load_validation_dataset(self):
        """Load validation dataset for global model evaluation"""
        logger.info("Loading global validation dataset...")
        
        # Load validation filenames
        filenames = []
        print(f"Loading validation files from {self.args.val_file}")
        with open(self.args.val_file) as reader:
            
            for filename in reader.readlines():
                filename = filename.rstrip()
                if os.path.exists(filename):
                    filenames.append(filename)
        
        if not filenames:
            logger.error(f"No validation files found in {self.args.val_file}")
            raise FileNotFoundError(f"No validation files found in {self.args.val_file}")
        
        logger.info(f"Loaded {len(filenames)} validation images")
        
        self.val_dataset = Dataset(filenames, self.args.input_size, self.params, False)
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
    
    def handle_client(self, conn, addr):
        """Handle communication with a client"""
        client_id = None
        try:
            # Receive client ID
            client_data = self.receive_data(conn)
            if client_data:
                client_id = pickle.loads(client_data)
                
                if self.round_in_progress:
                    logger.warning(f"Client {client_id} from {addr} tried to connect during round {self.current_round} processing. Sending exit signal.")
                    # Send -1 as round number to indicate client should exit
                    self.send_data(conn, pickle.dumps(-1))
                    conn.close()
                    return
                
                # Check if this client has already participated in this round
                if client_id in self.unique_clients:
                    logger.warning(f"Client {client_id} from {addr} already participated in round {self.current_round}. Sending exit signal.")
                    # Send -2 as round number to indicate client should exit (already participated)
                    self.send_data(conn, pickle.dumps(-2))
                    conn.close()
                    return
                
                logger.info(f"Client {client_id} from {addr} connected")
                # Add to the set of unique clients for this round
                self.unique_clients.add(client_id)
            
                # Send current round to client
                self.send_data(conn, pickle.dumps(self.current_round))
                
                # Send global model to client
                model_state = {k: v.cpu() for k, v in self.global_model.state_dict().items()}
                self.send_data(conn, pickle.dumps(model_state))
                logger.info(f"Sent global model to client {client_id}")
                
                # Receive trained model from client
                client_data = self.receive_data(conn)
                if client_data:
                    client_model_state, client_metrics, dataset_size = pickle.loads(client_data)
                    
                    self.client_models[client_id] = (client_model_state, dataset_size)
                    if 'mu' in client_metrics:
                        self.client_fedprox_settings[client_id] = client_metrics['mu']
                        logger.info(f"Client {client_id} is using FedProx with mu={client_metrics['mu']}")
                    
                    for key, value in client_metrics.items():
                        self.metrics[f"client_{client_id}_{key}"].append(value)
                    
                    self.metrics[f"client_{client_id}_dataset_size"] = dataset_size
                    
                    logger.info(f"Received trained model from client {client_id} (dataset size: {dataset_size})")
            
            conn.close()
            
        except Exception as e:
            logger.error(f"Error handling client {client_id}: {e}")
            if client_id in self.unique_clients:
                self.unique_clients.remove(client_id)
            if conn:
                conn.close()
    
    def aggregate_models(self):
        """Federated averaging of client models weighted by dataset size"""
        if not self.client_models:
            logger.warning("No client models to aggregate")
            return False
        
        logger.info(f"Aggregating {len(self.client_models)} client models using weighted averaging")
        
        total_dataset_size = sum(dataset_size for _, dataset_size in self.client_models.values())
        logger.info(f"Total dataset size across all clients: {total_dataset_size}")
        
        global_dict = self.global_model.state_dict()
        for key in global_dict.keys():
            global_dict[key] = torch.zeros_like(global_dict[key])
        
        for client_id, (client_state, dataset_size) in self.client_models.items():
            weight = float(dataset_size) / float(total_dataset_size)
            
            if client_id in self.client_fedprox_settings:
                logger.info(f"Client {client_id}: dataset size = {dataset_size}, weight = {weight:.4f}, FedProx mu = {self.client_fedprox_settings[client_id]}")
            else:
                logger.info(f"Client {client_id}: dataset size = {dataset_size}, weight = {weight:.4f}, using FedAvg")
            
            for key in global_dict.keys():
                client_param = client_state[key].to(self.device)
                
                if client_param.dtype == torch.long or client_param.dtype == torch.int:
                    if dataset_size == max(ds for _, ds in self.client_models.values()):
                        global_dict[key] = client_param
                else:
                    global_dict[key] += client_param * weight
        
        self.global_model.load_state_dict(global_dict)
        return True

    
    @torch.no_grad()
    def evaluate_global_model(self):
        """Evaluate the global model on validation data"""
        logger.info("Evaluating global model...")
        
        self.global_model.eval()
        self.global_model.half()
        
        iou_v = torch.linspace(0.5, 0.95, 10).to(self.device) 
        n_iou = iou_v.numel()
        
        metrics = []
        for samples, targets, shapes in tqdm.tqdm(self.val_loader, desc="Evaluating"):
            samples = samples.to(self.device)
            targets = targets.to(self.device)
            samples = samples.half()  # uint8 to fp16/32
            samples = samples / 255  # 0 - 255 to 0.0 - 1.0
            _, _, height, width = samples.shape
            
            # Inference
            outputs = self.global_model(samples)
            
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
        metrics = [torch.cat(x, 0).cpu().numpy() for x in zip(*metrics)]
        
        if len(metrics) and metrics[0].any():
            tp, fp, precision, recall, map50, mean_ap = util.compute_ap(*metrics)
            results = {
                'mAP50': float(map50),
                'mAP': float(mean_ap),
                'precision': float(precision),
                'recall': float(recall)
            }
        else:
            results = {
                'mAP50': 0.0,
                'mAP': 0.0,
                'precision': 0.0,
                'recall': 0.0
            }
        
        logger.info(f"Global model evaluation - mAP: {results['mAP']:.4f}, mAP50: {results['mAP50']:.4f}")
        for key, value in results.items():
            self.metrics[f"global_{key}"].append(value)
            
        self.global_model.float()
        
        return results
    
    def save_global_model(self, round_num):
        """Save the global model"""
        save_path = os.path.join(self.args.save_dir, f"global_model_round_{round_num}.pt")
        torch.save({
            'model': self.global_model.state_dict(),
            'round': round_num,
            'args': self.args,
        }, save_path)
        logger.info(f"Global model saved to {save_path}")
    
    def save_metrics(self):
        """Save metrics to file"""
        metrics_path = os.path.join(self.args.save_dir, 'metrics', f"metrics_round_{self.current_round}.pkl")
        with open(metrics_path, 'wb') as f:
            pickle.dump(dict(self.metrics), f)
        
        # Also save FedProx settings for analysis
        fedprox_path = os.path.join(self.args.save_dir, 'metrics', f"fedprox_settings_round_{self.current_round}.pkl")
        with open(fedprox_path, 'wb') as f:
            pickle.dump(self.client_fedprox_settings, f)
            
        logger.info(f"Metrics saved to {metrics_path}")
        logger.info(f"FedProx settings saved to {fedprox_path}")
    
    def run(self):
        """Run the federated learning server"""
        logger.info(f"Starting federated learning with {self.rounds} rounds")
        
        # Evaluate initial model
        self.evaluate_global_model()
        self.save_global_model(0)
        
        try:
            for round_num in range(1, self.rounds + 1):
                self.current_round = round_num
                logger.info(f"Starting round {round_num}/{self.rounds}")
                
                # Reset for new round
                self.unique_clients = set()  
                self.client_models = {}
                self.client_fedprox_settings = {}
                self.round_in_progress = False  
                
                # Wait for minimum number of clients
                logger.info(f"Waiting for at least {self.min_clients} unique clients to connect...")
                
                client_threads = []
                # Accept clients until we have enough
                while len(self.unique_clients) < self.min_clients:
                    conn, addr = self.sock.accept()
                    client_thread = threading.Thread(target=self.handle_client, args=(conn, addr))
                    client_thread.start()
                    client_threads.append(client_thread)    
                    time.sleep(0.1)
                
                additional_wait_time = 30
                logger.info(f"Minimum clients reached. Waiting {additional_wait_time}s for additional clients...")
                time.sleep(additional_wait_time)
                
                self.round_in_progress = True
                logger.info(f"Round {round_num} is now in progress. No new clients will be accepted until next round.")
                
                # Wait for all client threads to complete
                for thread in client_threads:
                    thread.join()
                
                logger.info(f"Round {round_num} - {len(self.unique_clients)} unique clients participated")
                
                # Log FedProx usage statistics
                fedprox_clients = [client_id for client_id in self.unique_clients if client_id in self.client_fedprox_settings]
                logger.info(f"Round {round_num} - {len(fedprox_clients)} clients used FedProx, {len(self.unique_clients) - len(fedprox_clients)} used FedAvg")
                
                # Aggregate models
                if self.aggregate_models():
                    evaluation_results = self.evaluate_global_model()    
                    self.save_global_model(round_num)

                    self.save_metrics()                    
                    logger.info(f"Round {round_num} completed. mAP: {evaluation_results['mAP']:.4f}")
                else:
                    logger.warning(f"Round {round_num} - Model aggregation failed. Skipping evaluation.")
            
            logger.info("Federated learning completed")
        except KeyboardInterrupt:
            logger.info("Federated learning interrupted")
        except Exception as e:
            logger.error(f"Error in federated learning: {e}")
        finally:
            self.sock.close()


def main():

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(description="Federated Learning Server for YOLOv8")
    parser.add_argument('--port', type=int, default=5000, help='Port to listen on')
    parser.add_argument('--rounds', type=int, default=50, help='Number of federated learning rounds')
    parser.add_argument('--min-clients', type=int, default=3, help='Minimum number of clients per round')
    parser.add_argument('--input-size', type=int, default=640, help='Input image size')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size for evaluation')
    parser.add_argument('--model-size', type=str, default='n', choices=['n', 's', 'm', 'l', 'x'],
                        help='YOLOv8 model size: n(ano), s(mall), m(edium), l(arge), x(large)')
    parser.add_argument('--val-file', type=str, default='/home/ssl40/cs21b048_37_dl/raw_datasets/soda10m/new_data2/val/val.txt', 
                        help='Path to validation image list')
    parser.add_argument('--save-dir', type=str, default='',
                        help='Directory to save models and metrics')
    args = parser.parse_args()

    if args.save_dir == '':
        args.save_dir = os.path.join('fl_server_results_final', f'soda10m_fedprox_server_{timestamp}')
    else:
        args.save_dir = os.path.join(args.save_dir, f'soda10m_fedprox_server_{timestamp}')
    os.makedirs(args.save_dir, exist_ok=True)
    logger.info(f"Results will be saved in {args.save_dir}")
    # logger.info(f"Client ID: {args.client_id}")
    with open(os.path.join(args.save_dir, 'args.yaml'), 'w') as f:
        yaml.dump(vars(args), f, default_flow_style=False)
    logger.info(f"Arguments saved to {os.path.join(args.save_dir, 'args.yaml')}")
    
    with open(os.path.join('utils', 'args.yaml'), errors='ignore') as f:
        params = yaml.safe_load(f)
    
    server = FederatedServer(args, params)
    server.run()


if __name__ == "__main__":
    main()