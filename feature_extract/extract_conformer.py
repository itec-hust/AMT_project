import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import torch.nn as nn
from conformer import Conformer
import torch.distributed as dist
import torch.multiprocessing as mp
import json
import argparse
import torch
import numpy as np
from utils import TransformImage
from data import TestDataset

def test(rank, world_size, gpu_no, device, config, save_dir):
  os.environ['MASTER_ADDR'] = 'localhost'
  os.environ['MASTER_PORT'] = '12358'
  # initialize the process group
  dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)

  net = Conformer(**config['model']['args'])
  net.load_state_dict(torch.load(config['test']['checkpoint_path'], map_location='cpu')['model_state_dict'])
  net = nn.SyncBatchNorm.convert_sync_batchnorm(net).to(device)
  net = DDP(net, device_ids=[gpu_no])

  test_dataset = TestDataset(config['test']['image_dir'])
  test_sampler = DistributedSampler(test_dataset, rank=rank, num_replicas=world_size, shuffle=False)
  test_loader = DataLoader(test_dataset, batch_size=1, drop_last=False, 
                            num_workers=config['num_workers'], sampler=test_sampler)

  image_transform = TransformImage(config['test']['loc_config_dir'], config['test']['image_size'])

  net.eval()
  with torch.no_grad():
    for fileno, filedir in enumerate(test_loader):
      if rank == 0:
        print('predict: %03d / %03d'%((fileno + 1) * world_size, len(test_loader) * world_size))
      filedir = filedir[0]
      basename = os.path.basename(filedir)
      conv_save_path = os.path.join(save_dir, 'conv', basename + '.npy')
      trans_save_path = os.path.join(save_dir, 'trans', basename + '.npy')
      if os.path.exists(conv_save_path) and os.path.exists(trans_save_path):
          continue
     
      image_num = len(os.listdir(filedir))
      data_num = image_num - config['test']['successive'] + 1
      conv_features = np.zeros((data_num, 768), dtype=np.float32)
      trans_features = np.zeros((data_num, 128), dtype=np.float32)

      for sample_num in range(data_num):
        if rank == 0 and sample_num % 100 == 0:
          print('predict %s: %05d / %05d'%(os.path.basename(filedir), sample_num, data_num))
        imgs = []
        for i in range(config['test']['successive']):
          filepath = os.path.join(filedir, '%05d.jpg'%(sample_num + i))
          img = image_transform.transform(filepath)
          imgs.append(img)

        imgs = torch.from_numpy(np.stack(imgs, axis=0)[None, :, None]).to(device)
        conv_feature, trans_feature = net(imgs)
        conv_features[sample_num] = conv_feature.cpu().numpy()
        trans_features[sample_num] = trans_feature.cpu().numpy()
    
      np.save(conv_save_path, conv_features)
      np.save(trans_save_path, trans_features)

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--config', '-c', type=str, help='the config file path')
  parser.add_argument('--save_dir', '-s', type=str, help='the feature save_dir')
  parser.add_argument('--gpu_nums', '-g', type=int, help='the gpu nums')
  parser.add_argument('--process_num', '-p', type=int, default=1, help='the process num per gpu')
  return parser.parse_args()

def main(args):
  config_path = args.config
  gpu_nums = args.gpu_nums
  process_num = args.process_num
  world_size = gpu_nums * process_num
  save_dir = args.save_dir
  with open(config_path) as f:
    config = json.load(f)

  os.makedirs(os.path.join(save_dir, 'conv'), exist_ok=True)
  os.makedirs(os.path.join(save_dir, 'trans'), exist_ok=True)
  processes = []
  # mp.set_start_method("spawn")
  for gpu in range(gpu_nums): # 显卡
    for pro in range(process_num): # 每个显卡进程数
      rank = gpu * process_num + pro
      p = mp.Process(target=test, args=(rank, world_size, gpu, 'cuda:{}'.format(gpu), config, save_dir))
      processes.append(p)

  for p in processes:
    p.start()
  
  for p in processes:
    p.join()


if __name__ == '__main__':
  args = parse_args()
  main(args)
