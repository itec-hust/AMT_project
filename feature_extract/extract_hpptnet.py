import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import torch.nn as nn
from hpptnet import HPPTNet
import torch.distributed as dist
import torch.multiprocessing as mp
import json
import argparse
import torch
import numpy as np
from data import TestDataset, AudioTestset


def test(rank, world_size, gpu_no, device, config, save_dir):
  os.environ['MASTER_ADDR'] = 'localhost'
  os.environ['MASTER_PORT'] = '12355'
  # initialize the process group
  dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)

  net = HPPTNet(**config['model']['args'])
  net.load_state_dict(torch.load(config['test']['checkpoint_path'], map_location='cpu')['model_state_dict'])
  net = nn.SyncBatchNorm.convert_sync_batchnorm(net).to(device)
  net = DDP(net, device_ids=[gpu_no])

  test_dataset = TestDataset(config['test']['audio_dir'])
  test_sampler = DistributedSampler(test_dataset, rank=rank, num_replicas=world_size, shuffle=False)
  test_loader = DataLoader(test_dataset, batch_size=1, drop_last=False,
                            num_workers=config['num_workers'], sampler=test_sampler)

  net.eval()
  with torch.no_grad():
    for fileno, filepath in enumerate(test_loader):
      filepath = filepath[0]
      basename = os.path.basename(filepath).replace('.wav', '.npy')
      onset_save_path = os.path.join(save_dir, 'onset', basename)
      offset_save_path = os.path.join(save_dir, 'offset', basename)
      frame_save_path = os.path.join(save_dir, 'frame', basename)
      if os.path.exists(onset_save_path) and os.path.exists(offset_save_path) and \
          os.path.exists(frame_save_path):
          continue

      if rank == 0:
        print('predict: %03d / %03d'%((fileno + 1) * world_size, len(test_loader) * world_size), filepath)

      test_set = AudioTestset(filepath, **config['test']['dataset_args'])
      loader = DataLoader(test_set, batch_size=config['test']['batch_size'], drop_last=False,
                          num_workers=config['test']['num_workers'])

      onset_features = np.zeros((len(test_set), 256), dtype=np.float32)
      offset_features = np.zeros((len(test_set), 256), dtype=np.float32)
      frame_features = np.zeros((len(test_set), 256), dtype=np.float32)
      start = 0
      for idx, audio in enumerate(loader):
        if rank == 0 and idx % 50 == 0:
          print('predicting: %03d / %03d'%(idx, len(loader)))
        audio = audio.to(device)
        onset_feature, offset_feature, frame_feature = net(audio)
        length = int(onset_feature.shape[0])
        onset_features[start: start + length] = onset_feature.cpu().numpy()
        offset_features[start: start + length] = offset_feature.cpu().numpy()
        frame_features[start: start + length] = frame_feature.cpu().numpy()
        start += length

      np.save(onset_save_path, onset_features)
      np.save(offset_save_path, offset_features)
      np.save(frame_save_path, frame_features)


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
  save_dir = args.save_dir
  world_size = gpu_nums * process_num
  with open(config_path) as f:
    config = json.load(f)

  os.makedirs(os.path.join(save_dir, 'onset'), exist_ok=True)
  os.makedirs(os.path.join(save_dir, 'offset'), exist_ok=True)
  os.makedirs(os.path.join(save_dir, 'frame'), exist_ok=True)

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

# CUDA_VISIBLE_DEVICES=1 python3 extract_hpptnet.py -c hppt.json  -g 1 -p 1 -s /home/data/wxk/tmp/features/hpptnet
if __name__ == '__main__':
  args = parse_args()
  main(args)