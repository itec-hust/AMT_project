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
from utils import dump_data
from utils import write_res
from utils import note_search, note_search_with_peak
from utils import TransformImage
from data import TestDataset

def test(rank, world_size, gpu_no, device, config):
  os.environ['MASTER_ADDR'] = 'localhost'
  os.environ['MASTER_PORT'] = '12355'
  # initialize the process group
  dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)

  net = getattr(model, config['model']['name'])(**config['model']['args'])
  net.load_state_dict(torch.load(config['test']['checkpoint_path'], map_location='cpu')['model_state_dict'])
  net = nn.SyncBatchNorm.convert_sync_batchnorm(net).to(device)
  net = DDP(net, device_ids=[gpu_no])

  test_dataset = TestDataset(config['test']['image_dir'])
  test_sampler = DistributedSampler(test_dataset, rank=rank, num_replicas=world_size, shuffle=False)
  test_loader = DataLoader(test_dataset, batch_size=1, drop_last=False, 
                            num_workers=config['num_workers'], sampler=test_sampler)

  with open(config['test']['fps_config_path']) as f:
    fps_config = json.load(f)

  image_transform = TransformImage(config['test']['loc_config_dir'], config['test']['image_size'])

  avg_path = os.path.join(config['test']['save_dir'], 'avg')
  sum_path = os.path.join(config['test']['save_dir'], 'sum')
  conv_path = os.path.join(config['test']['save_dir'], 'conv')
  trans_path = os.path.join(config['test']['save_dir'], 'trans')

  os.makedirs(avg_path, exist_ok=True)
  os.makedirs(sum_path, exist_ok=True)
  os.makedirs(conv_path, exist_ok=True)
  os.makedirs(trans_path, exist_ok=True)

  net.eval()
  with torch.no_grad():
    for fileno, filedir in enumerate(test_loader):
      if rank == 0:
        print('predict: %03d / %03d'%((fileno + 1) * world_size, len(test_loader) * world_size))
      filedir = filedir[0]
      image_num = len(os.listdir(filedir))
      
      data_num = image_num - config['test']['successive'] + 1
      avg_probs = np.zeros((data_num, 88), dtype=np.float32)
      sum_probs = np.zeros((data_num, 88), dtype=np.float32)
      conv_probs = np.zeros((data_num, 88), dtype=np.float32)
      trans_probs = np.zeros((data_num, 88), dtype=np.float32)

      for sample_num in range(data_num):
        if rank == 0 and sample_num % 100 == 0:
          print('predict %s: %05d / %05d'%(os.path.basename(filedir), sample_num, data_num))
        imgs = []
        for i in range(config['test']['successive']):
          filepath = os.path.join(filedir, '%05d.jpg'%(sample_num + i))
          img = image_transform.transform(filepath)
          imgs.append(img)

        if config['test']['dump_path']:
          dump_data(imgs, None, config['test']['dump_path'], os.path.basename(filedir), (config['test']['successive'] // 2 + sample_num) / fps_config[os.path.basename(filedir)])

        imgs = torch.from_numpy(np.stack(imgs, axis=0)[None, :, None]).to(device)
        conv_logit, trans_logit = net(imgs)
        avg_prob = (conv_logit.sigmoid() + trans_logit.sigmoid()) / 2
        sum_prob = (conv_logit + trans_logit).sigmoid()
        conv_prob = conv_logit.sigmoid()
        trans_prob = trans_logit.sigmoid()
        avg_probs[sample_num] = avg_prob.cpu().numpy()
        sum_probs[sample_num] = sum_prob.cpu().numpy()
        conv_probs[sample_num] = conv_prob.cpu().numpy()
        trans_probs[sample_num] = trans_prob.cpu().numpy()
    
      basename = os.path.basename(filedir)
      fps = fps_config[basename]
      offset = config['test']['successive'] // 2
      prob_threshould = config['test']['prob_threshould']

      avg_notes = note_search_with_peak(avg_probs, fps, offset, prob_threshould)
      save_path = os.path.join(avg_path, basename)
      write_res(avg_notes, save_path)

      sum_notes = note_search_with_peak(sum_probs, fps, offset, prob_threshould)
      save_path = os.path.join(sum_path, basename)
      write_res(sum_notes, save_path)

      conv_notes = note_search_with_peak(conv_probs, fps, offset, prob_threshould)
      save_path = os.path.join(conv_path, basename)
      write_res(conv_notes, save_path)

      trans_notes = note_search_with_peak(trans_probs, fps, offset, prob_threshould)
      save_path = os.path.join(trans_path, basename)
      write_res(trans_notes, save_path)


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