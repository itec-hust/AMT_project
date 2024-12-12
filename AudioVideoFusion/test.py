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
from utils import note_search_with_peak
from data import VideoAudioTestset
import math

def test(rank, world_size, gpu_no, device, config):
  os.environ['MASTER_ADDR'] = 'localhost'
  os.environ['MASTER_PORT'] = '12357'
  # initialize the process group
  dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)

  net = getattr(model, config['model']['name'])(**config['model']['args'])
  net.load_state_dict(torch.load(config['test']['checkpoint_path'], map_location='cpu')['model_state_dict'])
  net = nn.SyncBatchNorm.convert_sync_batchnorm(net).to(device)
  net = DDP(net, device_ids=[gpu_no])

  test_dataset = VideoAudioTestset(config['test']['video_feature_dir'], config['test']['audio_feature_dir'],
                                  config['test']['video_label_dir'], config['test']['audio_label_dir'])
  test_sampler = DistributedSampler(test_dataset, rank=rank, num_replicas=world_size, shuffle=False)
  test_loader = DataLoader(test_dataset, batch_size=1, drop_last=False, 
                            num_workers=config['num_workers'], sampler=test_sampler)

  with open(config['test']['fps_config_path']) as f:
    fps_config = json.load(f)

  net.eval()
  with torch.no_grad():
    for fileno, data in enumerate(test_loader):
      if rank == 0:
        print('predict: %03d / %03d'%((fileno + 1) * world_size, len(test_loader) * world_size))
      video_feature_path = data[0][0]
      audio_feature_path = data[1][0]
      video_features = np.load(video_feature_path)
      video_features = np.pad(video_features, ((config['test']['successive_frame']//2, 0), (0, 0)))
      audio_features = np.load(audio_feature_path)
      time_shift = float(data[2])

      data_num = len(audio_features)
      probs = np.zeros((data_num, 88), dtype=np.float32)
      basename = os.path.basename(video_feature_path).replace('.npy', '')
      fps = fps_config[basename]

      for sample_num in range(data_num):
        if rank == 0 and sample_num % 500 == 0:
          print('predict %s: %05d / %05d'%(os.path.basename(video_feature_path), sample_num, data_num))

        audio_feature = audio_features[sample_num]
        audio_feature = torch.from_numpy(audio_feature[None, ]).to(device)

        time = sample_num * config['dataset']['share_args']['stride'] + time_shift
        video_frame = int(time * fps + 0.5)
        if video_frame >= len(video_features):
          break
        video_feature = video_features[video_frame]
        video_feature = torch.from_numpy(video_feature[None, ]).to(device)
        logit = net(video_feature, audio_feature)
        prob = logit.sigmoid()
        probs[sample_num] = prob.cpu().numpy()
    
      prob_threshould = config['test']['prob_threshould']
      audio_fps = 1. / config['test']['stride']
      notes = note_search_with_peak(probs, audio_fps, 0., prob_threshould)
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