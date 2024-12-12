import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import torch.nn as nn
import model
import torch.distributed as dist
import torch.multiprocessing as mp
import json
import argparse
import torch
import numpy as np
from utils import write_res
from utils import note_search, note_search_with_peak
from data import TestDataset, AudioTestset


def test(rank, world_size, gpu_no, device, config):
  os.environ['MASTER_ADDR'] = 'localhost'
  os.environ['MASTER_PORT'] = '12355'
  # initialize the process group
  dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)

  ignore_offset_head = config['model']['ignore_offset_head']
  ignore_velocity_head = config['model']['ignore_velocity_head']
  net = getattr(model, config['model']['name'])(**config['model']['args'], 
          ignore_offset_head=ignore_offset_head, ignore_velocity_head=ignore_velocity_head)
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
      if rank == 0:
        print('predict: %03d / %03d'%((fileno + 1) * world_size, len(test_loader) * world_size), filepath)

      test_set = AudioTestset(filepath, **config['test']['dataset_args'])
      loader = DataLoader(test_set, batch_size=config['test']['batch_size'], drop_last=False, 
                          num_workers=config['test']['num_workers'])
      
      onset_probs = np.zeros((len(test_set), 88), dtype=np.float32)
      start = 0
      for idx, audio in enumerate(loader):
        if rank == 0 and idx % 50 == 0:
          print('predicting: %03d / %03d'%(idx, len(loader)))
        audio = audio.to(device)
        if ignore_offset_head:
          onset_logits, frame_logits = net(audio)
        elif ignore_velocity_head:
          onset_logits, offset_logits, frame_logits = net(audio)
        else:
          onset_logits, offset_logits, frame_logits, velocity_logits = net(audio)
        onset_prob = onset_logits.sigmoid()
        length = int(onset_prob.shape[0])
        onset_probs[start: start + length] = onset_prob.cpu().numpy()
        start += length
    
      prob_threshould = config['test']['prob_threshould']
      fps = 1. / config['test']['stride']

      notes = note_search_with_peak(onset_probs, fps, 0., prob_threshould)
      basename = os.path.basename(filepath).replace('.wav', '').replace('16k_', '')
      save_path = os.path.join(config['test']['save_dir'], basename)
      write_res(notes, save_path)


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--config', '-c', type=str, help='the config file path')
  parser.add_argument('--gpu_nums', '-g', type=int, help='the gpu nums')
  parser.add_argument('--process_num', '-p', type=int, default=1, help='the process num per gpu')
  return parser.parse_args()

def main(args):
  config_path = args.config
  gpu_nums = args.gpu_nums
  process_num = args.process_num
  world_size = gpu_nums * process_num
  with open(config_path) as f:
    config = json.load(f)

  config['test']['save_dir'] = os.path.join(config['save_dir'], 'res')
  os.makedirs(config['test']['save_dir'], exist_ok=True)

  processes = []
  # mp.set_start_method("spawn")
  for gpu in range(gpu_nums): # 显卡
    for pro in range(process_num): # 每个显卡进程数
      rank = gpu * process_num + pro
      p = mp.Process(target=test, args=(rank, world_size, gpu, 'cuda:{}'.format(gpu), config))
      processes.append(p)

  for p in processes:
    p.start()
  
  for p in processes:
    p.join()

if __name__ == '__main__':
  args = parse_args()
  main(args)