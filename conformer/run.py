import os
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.optim as optim
import torch.nn as nn
import math
from torch.utils.data.distributed import DistributedSampler
from train import train
import argparse
import json
import shutil
import model
import evaluate
import data
import utils
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
import argparse
import torch

# pytorch 余弦周期退火参考 论文中分类任务使用余弦类型的学习率调整 同时使用weight_decay = 0.05的Adam
# batch_size = 1024 epochs = 300
# https://blog.csdn.net/qq_40723205/article/details/123198792

def run(rank, world_size, device, config):
  os.environ['MASTER_ADDR'] = 'localhost'
  os.environ['MASTER_PORT'] = '12357'
  # initialize the process group
  dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)

  net = getattr(model, config['model']['name'])(**config['model']['args'])
  optimizer = getattr(optim, config['optimizer']['name'])(net.parameters(), **config['optimizer']['args'])
  # lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=config['optimizer']['lr_decay']) 
  # lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 20) # T_max指T_max步部署走完cosin函数的1/4 后续调用同上，也是需要step函数
  
  # warmup + 余弦退火
  warm_up_iter = config['optimizer']['warm_up_times']
  T_max = config['optimizer']['T_max_times']
  lr_max = config['optimizer']['args']['lr']
  lr_min = config['optimizer']['lr_min']
  lr_lambda = lambda cur_iter: cur_iter / warm_up_iter if  cur_iter < warm_up_iter else \
          (lr_min + 0.5 * (lr_max - lr_min) * (1.0 + math.cos((cur_iter - warm_up_iter) / (T_max - warm_up_iter) * math.pi))) / lr_max
  lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

  net = nn.SyncBatchNorm.convert_sync_batchnorm(net).to(device)
  net = DDP(net, device_ids=[rank])

  train_dataset = getattr(data, config['dataset']['name'])(**config['dataset']['share_args'], **config['dataset']['train_args'])
  val_dataset = getattr(data, config['dataset']['name'])(**config['dataset']['share_args'], **config['dataset']['valid_args'])

  train_sampler = DistributedSampler(train_dataset, rank=rank, num_replicas=world_size)
  train_loader = DataLoader(train_dataset, batch_size=config['train_size'], drop_last=True, 
                            num_workers=config['num_workers'], sampler=train_sampler)
  val_sampler = DistributedSampler(val_dataset, rank=rank, num_replicas=world_size, shuffle=False)
  val_loader = DataLoader(val_dataset, batch_size=config['val_size'], drop_last=True, 
                          num_workers=config['val_num_workers'], sampler=val_sampler)

  train_loss_cls = getattr(evaluate, config['loss_cls']['name'])(**config['loss_cls']['args'])
  val_loss_cls = getattr(evaluate, config['loss_cls']['name'])(**config['loss_cls']['args'])
  train_metric_cls = getattr(evaluate, config['metric_cls']['name'])(**config['metric_cls']['args'])
  val_metric_cls = getattr(evaluate, config['metric_cls']['name'])(**config['metric_cls']['args'])

  logger = None
  writer = None
  if rank == 0:
    logfile = os.path.join(config['save_dir'], 'train.log')
    logger = utils.Logger(logfile).get_log()
    writer = SummaryWriter(config['save_dir'])

  train_config = {
                    'train_loader': train_loader,
                    'train_sampler': train_sampler,
                    'val_loader': val_loader,
                    'opt': optimizer,
                    'lr_scheduler': lr_scheduler,
                    'lr_decay_step': config['optimizer']['lr_decay_step'],
                    'train_loss_cls': train_loss_cls,
                    'val_loss_cls': val_loss_cls,
                    'train_metric_cls': train_metric_cls,
                    'val_metric_cls': val_metric_cls,
                    'rank': rank,
                    'world_size': world_size,
                    'epochs': config['epochs'],
                    'device': device,
                    'save_checkpoint_dir': os.path.join(config['save_dir'], 'ckpt')
                }
  
  log_config = {
                  'train_log_interval': config['interval']['train_log_interval'],
                  'val_interval': config['interval']['val_interval'],
                  'val_log_interval': config['interval']['val_log_interval'],
                  'save_log_interval': config['interval']['save_log_interval'],
                  'start_val_step': config['interval']['start_val_step']
              }

  train(net, train_config, log_config, logger, writer)

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--config', '-c', type=str, help='the config file path')
  parser.add_argument('--world_size', '-w', type=int, help='the gpu nums')
  return parser.parse_args()

def main(args):
  config_path = args.config
  world_size = args.world_size
  with open(config_path) as f:
    config = json.load(f)

  save_path = config['save_dir']
  
  if os.path.exists(save_path):
    shutil.rmtree(save_path)

  current_absolute_path = os.path.abspath(__file__)
  current_absolute_dir = os.path.dirname(current_absolute_path)
  parent_absolute_dir = os.path.dirname(current_absolute_dir)
  shutil.copytree(parent_absolute_dir, save_path)

  processes = []
  for rank in range(world_size):
    p = mp.Process(target=run, args=(rank, world_size, 'cuda:{}'.format(rank), config))
    processes.append(p)
  
  for p in processes:
    p.start()

  for p in processes:
    p.join()

if __name__ == '__main__':
  args = parse_args()
  main(args)