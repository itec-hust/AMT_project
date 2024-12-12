import logging
import torch.distributed as dist
import os
import cv2
import numpy as np
import json
import soundfile as sf
import librosa

def dump_data(images, labels, dump_path, name, mid_time, info=''):
  save_path = os.path.join(dump_path, name + '_%.3f'%mid_time + '_' + info)
  os.makedirs(save_path, exist_ok=True)
  for idx, image in enumerate(images):
    save_img_path = os.path.join(save_path, '%d.jpg'%idx)
    image = (image.squeeze() * 255).astype('uint8')
    cv2.imwrite(save_img_path, image)
  
  if labels is not None:
    notes = []
    for i in range(88):
      if labels[i] > 0.:
        notes.append(str(i + 1))
    label_path = os.path.join(save_path, 'label.txt')
    with open(label_path, 'wt') as f:
      f.write('%s'%('\t'.join(notes)))


def dump_image_data(images, image_labels, dump_path, name, mid_time):
  save_path = os.path.join(dump_path, name + '_%.3f'%mid_time)
  os.makedirs(save_path, exist_ok=True)
  for idx, image in enumerate(images):
    save_img_path = os.path.join(save_path, '%d.jpg'%idx)
    if image.dtype != np.uint8:
      image = (image.squeeze() * 255).astype('uint8')
    cv2.imwrite(save_img_path, image)
  
  notes = []
  for i in range(10):
    if image_labels[i] > 0.:
      notes.append(str(i + 1))
  label_path = os.path.join(save_path, 'label.txt')
  with open(label_path, 'wt') as f:
    f.write('%s'%('\t'.join(notes)))

def dump_raw_data(dump_path, data_dict):
  os.makedirs(os.path.dirname(dump_path), exist_ok=True)
  np.save(dump_path, data_dict)

def dump_audio_data(audio, sr, onset_label, offset_label, frame_label, 
                    dump_path, name, mid_time, time_around):
  save_path = os.path.join(dump_path, name + '_%.3f'%mid_time)
  os.makedirs(save_path, exist_ok=True)

  audio_path = os.path.join(save_path, 'audio.wav')
  sf.write(audio_path, audio, sr)

  duration = len(audio) / sr
  start = mid_time - duration / 2
  stop = mid_time + duration / 2
  label_start = mid_time - time_around
  label_stop = mid_time + time_around
  onsets, offsets, frames = [] , [], []
  for i in range(88):
    if onset_label[i] > 0:
      onsets.append(i + 21)
    if offset_label[i] > 0:
      offsets.append(i + 21)
    if frame_label[i] > 0:
      frames.append(i + 21)
  label_path = os.path.join(save_path, 'label.txt')
  with open(label_path, 'wt') as f:
    f.write('audio start: %.3f stop: %.3f\n'%(start, stop))
    f.write('label start: %.3f stop: %.3f\n'%(label_start, label_stop))
    f.write('onsets: ' + ' '.join(str(midi) for midi in onsets) + '\n')
    f.write('offsets: ' + ' '.join(str(midi) for midi in offsets) + '\n')
    f.write('frames: ' + ' '.join(str(midi) for midi in frames) + '\n')

    f.write('onsets: ' + ' '.join('{:.2f}Hz'.format(librosa.midi_to_hz(midi)) for midi in onsets) + '\n')
    f.write('offsets: ' + ' '.join('{:.2f}Hz'.format(librosa.midi_to_hz(midi)) for midi in offsets) + '\n')
    f.write('frames: ' + ' '.join('{:.2f}Hz'.format(librosa.midi_to_hz(midi)) for midi in frames) + '\n')

def all_reduce(data_config, world_size):
  for key in data_config.keys():
    dist.all_reduce(data_config[key], op=dist.ReduceOp.SUM)
    data_config[key] = float(data_config[key].item()) / world_size

class Logger:
  def __init__(self, logfile, logger=None, level=logging.INFO):
    self.logger = logging.getLogger(logger)
    self.logger.propagate = False # 防止终端重复打印
    self.logger.setLevel(level)
    fh = logging.FileHandler(logfile, 'a', encoding='utf-8')
    fh.setLevel(level)

    sh = logging.StreamHandler()
    sh.setLevel(level)
    format = logging.Formatter('%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
    fh.setFormatter(format)
    sh.setFormatter(format)
    self.logger.handlers.clear()
    self.logger.addHandler(fh)
    self.logger.addHandler(sh)
    fh.close()
    sh.close()

  def get_log(self):
    return self.logger
  

class TransformImage:
  def __init__(self, loc_config_dir, image_size):
    super(TransformImage, self).__init__()
    
    self.loc_config_dir = loc_config_dir
    self.image_size = image_size

  def transform(self, img_path, to_gray=True, shift_bottom=100):
    if to_gray:
      img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    else:
      img = cv2.imread(img_path)
    
    videoname = os.path.basename(os.path.dirname(img_path))
    loc_config_path = os.path.join(self.loc_config_dir, videoname, 'label_new.json')
    with open(loc_config_path) as f:
      loc_config = json.load(f)
    rect = loc_config['rect']
    x1, y1, x2, y2 = rect

    img = img[y1: y2 + shift_bottom, x1: x2]
    img = cv2.resize(img, self.image_size[::-1])

    img = (img.astype('float32') / 255.0).astype('float32')
    return img

def write_res(notes, save_path, probs=None):
  if len(notes) == 0:
    return
  save_notes_path = save_path
  if not save_notes_path.endswith('.txt'):
    save_notes_path = save_path + '.txt'
  with open(save_notes_path, 'wt') as f:
    if len(notes[0]) == 2:
      for onset, pitch in notes:
        offset = onset + 0.02
        velocity = 0
        f.write('%.6f\t%.6f\t%d\t%d\n'%(onset, offset, pitch, velocity))
    elif len(notes[0]) == 3:
      for onset, offset, pitch in notes:
        velocity = 0
        f.write('%.6f\t%.6f\t%d\t%d\n'%(onset, offset, pitch, velocity))
    elif len(notes[0]) == 4:
      for onset, offset, pitch, velocity in notes:
        f.write('%.6f\t%.6f\t%d\t%d\n'%(onset, offset, pitch, velocity))
    else:
      raise ValueError('not supported yet!')
  
  if probs is not None:
    save_probs_path = save_path + '.res'
    np.savetxt(save_probs_path, probs, fmt='%.3f')

def note_search(probs, fps, offset, prob_threshould, midi_no=None):
  assert  probs.shape[1] == 88 or midi_no is not None
  notes = []
  if probs.shape[1] == 88:
    for i in range(probs.shape[0]):
      for j in range(probs.shape[1]):
        if probs[i][j] > prob_threshould:
          notes.append([(i + offset) / fps, j + 21])
  else:
    for j in range(len(probs)):
      if probs[j] > prob_threshould:
        notes.append([(i + offset) / fps, midi_no])
  
  return notes

def note_search_with_peak(probs, fps, offset, prob_threshould):
  assert  probs.shape[1] == 88 or probs.shape[1] == 10
  is_note = True if probs.shape[1] == 88 else False
  notes = []
  for i in range(probs.shape[1]): # for each 88 notes or 10 fingers
    sub_mask = (probs[:, i] > prob_threshould).tolist()
    sub_probs = probs[:, i] * sub_mask
    sub_mask.append(False)
    start = 0
    while start < len(sub_mask):
      end = start
      while sub_mask[end]:
        end += 1
      if end > start:
        idx = np.argmax(sub_probs[start: end])
        if is_note:
          notes.append([(idx + start + offset) / fps, i + 21])
        else:
          notes.append([(idx + start + offset) / fps, i])
      while end < len(sub_mask) and not sub_mask[end]:
        end += 1
      start = end
  
  notes.sort(key=lambda x: x[0])
      
  return notes
