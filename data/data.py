from torch.utils.data import Dataset
import json
import numpy as np
import os
import cv2
from utils import dump_data, dump_audio_data, dump_raw_data
from utils import dump_finger_data, dump_image_data
import librosa
import soundfile as sf
import math
import pandas as pd
from note_seq import midi_io
from note_seq import sequences_lib
import copy
import random

class TestDataset(Dataset):
  def __init__(self, file_dir):
    super(TestDataset, self).__init__()
    self.file_dir = file_dir
    filenames = os.listdir(file_dir)
    new_filenames = [filename for filename in filenames if filename.startswith('16k_')]
    if len(new_filenames) > 0:
      self.filenames = new_filenames
    else:
      self.filenames = filenames
    self.filenames.sort(key=lambda x: x.lower())
    random.shuffle(self.filenames)

  def __len__(self):
    return len(self.filenames)
  
  def __getitem__(self, index):
    return os.path.join(self.file_dir, self.filenames[index])

class VideoAudioTestset(Dataset):
  def __init__(self, video_feature_dir, audio_feature_dir, video_label_dir, audio_label_dir):
    super(VideoAudioTestset, self).__init__()

    assert len(os.listdir(video_feature_dir)) == len(os.listdir(audio_feature_dir)), 'the two data should be the same length'
    self.filenames = os.listdir(video_feature_dir)
    random.shuffle(self.filenames)

    self.video_feature_dir = video_feature_dir
    self.audio_feature_dir = audio_feature_dir
    self.video_label_dir = video_label_dir
    self.audio_label_dir = audio_label_dir

  def __len__(self):
    return len(self.filenames)
  
  def __getitem__(self, index):
    video_feature_path = os.path.join(self.video_feature_dir, self.filenames[index])
    audio_feature_dir_path = os.path.join(self.audio_feature_dir, self.filenames[index])
    video_midipath = os.path.join(self.video_label_dir, self.filenames[index].replace('.npy', '.txt'))
    audio_midipath = os.path.join(self.audio_label_dir, self.filenames[index].replace('.npy', '.txt'))
    video_midi = np.loadtxt(video_midipath).reshape(-1, 4)
    audio_midi = np.loadtxt(audio_midipath).reshape(-1, 4)
    return video_feature_path, audio_feature_dir_path, video_midi[0, 0] - audio_midi[0, 0]

class VideoAudioAttnTestset(Dataset):
  def __init__(self, feature_dir):
    super(VideoAudioAttnTestset, self).__init__()

    self.filenames = os.listdir(feature_dir)
    random.shuffle(self.filenames)

  def __len__(self):
    return len(self.filenames)
  
  def __getitem__(self, index):
    return self.filenames[index]



# 视频抽取的特征统一没有进行padding，因此前两帧的视频特征没有
# 音频模型特征统一进行了padding
class VideoAudioDataset(Dataset):
  def __init__(self, video_feature_dir, audio_feature_dir, video_label_dir, audio_label_dir,
                fps_config_path, successive_frame, stride, fmin, sr, bins_per_octave, frame):
    super(VideoAudioDataset, self).__init__()

    n_fft = sr / fmin / (2** (1 / bins_per_octave) - 1)
    n_fft = 2 ** int(math.ceil(math.log2(n_fft))) # 65536 4.096s
    window_t = (frame - 1) * stride + n_fft / sr # 4.216s
    
    # 检查两个模态数据量是否一致
    assert len(os.listdir(video_feature_dir)) == len(os.listdir(audio_feature_dir)), \
            'video feature and audio feature file num should be the same'  

    filenames = os.listdir(video_feature_dir)
    label_filenames = os.listdir(video_label_dir)
    assert len(filenames) <= len(label_filenames), 'image num: %d > label num: %d'%(len(filenames), len(label_filenames))
    
    with open(fps_config_path) as f:
      self.fps_config = json.load(f)

    self.filename_num = []
    self.filename_data = dict()
    for filename in filenames:
      video_feature_path = os.path.join(video_feature_dir, filename)
      audio_feature_path = os.path.join(audio_feature_dir, filename)
      video_feature = np.load(video_feature_path)
      audio_feature = np.load(audio_feature_path)
      filename = filename.replace('.npy', '')
      fps = self.fps_config[filename]
      video_mid_time = (len(video_feature) - 1 + successive_frame // 2) / fps
      audio_mid_time = (len(audio_feature) - 1) * stride

      videolabelpath = os.path.join(video_label_dir, filename+'.txt')
      audiolabelpath = os.path.join(audio_label_dir, filename+'.txt')
      videomidi = np.loadtxt(videolabelpath, dtype='float32').reshape([-1, 4])
      audiomidi = np.loadtxt(audiolabelpath, dtype='float32').reshape([-1, 4])
      time_shift = videomidi[0, 0] - audiomidi[0, 0]
      sample_num = int(min(video_mid_time - time_shift, audio_mid_time) / stride + 1)
      self.filename_num.append([filename, sample_num])
      video_feature = np.pad(video_feature, ((successive_frame // 2, 0), (0, 0)))
      self.filename_data[filename] = [video_feature, audio_feature, audiomidi, time_shift]

    self.filename_num.sort(key=lambda x: -x[1])

    for i in range(1, len(self.filename_num)):
      self.filename_num[i][1] += self.filename_num[i-1][1]

    self.window_t = window_t
    self.video_label_dir = video_label_dir
    self.audio_label_dir = audio_label_dir
    self.time_around = stride
    self.video_feature_dir = video_feature_dir
    self.audio_feature_dir = audio_feature_dir
    self.successive_frame = successive_frame
    self.stride = stride

  def __len__(self):
    return self.filename_num[-1][1]
  
  def __getitem__(self, index):

    # sample是指音频特征的下标
    filename = None
    sample = -1
    for i in range(len(self.filename_num)):
      if index < self.filename_num[i][1]:
        filename = self.filename_num[i][0]
        sample = index if i == 0 else index - self.filename_num[i-1][1]
        break

    if filename is None:
      raise ValueError('cannot find the required frame')

    # video_feature_path = os.path.join(self.video_feature_dir, filename + '.npy')
    # audio_feature_path = os.path.join(self.audio_feature_dir, filename + '.npy')
    # video_features = np.load(video_feature_path)
    video_features = self.filename_data[filename][0]
    # audio_feature = np.load(audio_feature_path)[sample]
    audio_feature = self.filename_data[filename][1][sample]


    # videolabelpath = os.path.join(self.video_label_dir, filename+'.txt')
    # audiolabelpath = os.path.join(self.audio_label_dir, filename+'.txt')
    # videomidi = np.loadtxt(videolabelpath).reshape([-1, 4])
    # audiomidi = np.loadtxt(audiolabelpath).reshape([-1, 4])

    fps = self.fps_config[filename]
    mid_time = sample* self.stride
    # time = mid_time + videomidi[0, 0] - audiomidi[0, 0]
    time = mid_time + self.filename_data[filename][-1]
    video_frame = int(time * fps + 0.5)
    video_feature = video_features[video_frame]

    onset_label = np.zeros((88, ), dtype=np.float32)
    # for onset, offset, pitch, velocity in audiomidi:
    for onset, offset, pitch, velocity in self.filename_data[filename][2]:
      if onset >= mid_time - self.time_around and onset <= mid_time + self.time_around:      
        onset_label[int(pitch)-21] = 1.

    return video_feature, audio_feature, onset_label

# 首先实现音频和视频长度一致的数据集
class VideoAudioSeqDataset(Dataset):
  def __init__(self, seq_len, video_feature_dir, audio_feature_dir, video_label_dir, audio_label_dir,
                fps_config_path, successive_frame, stride):
    super(VideoAudioSeqDataset, self).__init__()
    
    # 检查两个模态数据量是否一致
    assert len(os.listdir(video_feature_dir)) == len(os.listdir(audio_feature_dir)), \
            'video feature and audio feature file num should be the same'  

    filenames = os.listdir(video_feature_dir)
    label_filenames = os.listdir(video_label_dir)
    assert len(filenames) <= len(label_filenames), 'image num: %d > label num: %d'%(len(filenames), len(label_filenames))
    
    with open(fps_config_path) as f:
      self.fps_config = json.load(f)

    filename_num = []
    for filename in filenames:
      video_feature_path = os.path.join(video_feature_dir, filename)
      audio_feature_path = os.path.join(audio_feature_dir, filename)
      video_feature = np.load(video_feature_path)
      audio_feature = np.load(audio_feature_path)
      filename = filename.replace('.npy', '')
      fps = self.fps_config[filename]
      video_mid_time = (len(video_feature) - 1 + successive_frame // 2) / fps
      audio_mid_time = (len(audio_feature) - 1) * stride

      videolabelpath = os.path.join(video_label_dir, filename+'.txt')
      audiolabelpath = os.path.join(audio_label_dir, filename+'.txt')
      videomidi = np.loadtxt(videolabelpath).reshape([-1, 4])
      audiomidi = np.loadtxt(audiolabelpath).reshape([-1, 4])
      sample_num = int(min(video_mid_time - (videomidi[0, 0] - audiomidi[0, 0]), audio_mid_time) / stride + 1)
      filename_num.append([filename, sample_num])

    filename_num.sort(key=lambda x: -x[1])
    self.seq_filename_num = []
    self.seq_len = seq_len
    self.seq_hop = seq_len // 2
    seq_num = (filename_num[0][1] - self.seq_len) // self.seq_hop + 1

    self.seq_filename_num.append([filename_num[0][0], seq_num])
    for i in range(1, len(filename_num)):
      seq_num += (filename_num[i][1] - self.seq_len) // self.seq_hop + 1
      self.seq_filename_num.append([filename_num[i][0], seq_num])
    
    self.video_label_dir = video_label_dir
    self.audio_label_dir = audio_label_dir
    self.time_around = stride
    self.video_feature_dir = video_feature_dir
    self.audio_feature_dir = audio_feature_dir
    self.successive_frame = successive_frame
    self.stride = stride

  def __len__(self):
    return self.seq_filename_num[-1][1]
  
  def __getitem__(self, index):

    # sample是指音频特征的下标
    filename = None
    sample = -1
    for i in range(len(self.seq_filename_num)):
      if index < self.seq_filename_num[i][1]:
        filename = self.seq_filename_num[i][0]
        sample = index if i == 0 else index - self.seq_filename_num[i-1][1]
        break

    if filename is None:
      raise ValueError('cannot find the required frame')

    video_feature_path = os.path.join(self.video_feature_dir, filename + '.npy')
    audio_feature_path = os.path.join(self.audio_feature_dir, filename + '.npy')
    video_features = np.load(video_feature_path)
    video_features = np.pad(video_features, ((self.successive_frame // 2, 0), (0, 0)))
    audio_features = np.load(audio_feature_path)

    videolabelpath = os.path.join(self.video_label_dir, filename+'.txt')
    audiolabelpath = os.path.join(self.audio_label_dir, filename+'.txt')
    videomidi = np.loadtxt(videolabelpath).reshape([-1, 4])
    audiomidi = np.loadtxt(audiolabelpath).reshape([-1, 4])

    start_frame = sample * self.seq_hop
    stop_frame = sample * self.seq_hop + self.seq_len
    start_t = start_frame * self.stride - self.time_around
    stop_t = stop_frame * self.stride + self.time_around

    audio_feature = audio_features[start_frame: stop_frame]

    onset_label = np.zeros((self.seq_len, 88), dtype='float32')
    for onset, offset, pitch, velocity in audiomidi:
      if onset >= start_t and onset <= stop_t:
        left_frame = int((onset - self.time_around) / self.stride + 0.5) - start_frame
        right_frame = int((onset + self.time_around) / self.stride + 0.5) - start_frame
        for frame in range(left_frame, right_frame + 1):
          if frame >= 0 and frame < self.seq_len:
            onset_label[frame][int(pitch) - 21] = 1.

    fps = self.fps_config[filename]
    T_v, video_dim = video_features.shape
    video_feature = np.zeros((self.seq_len, video_dim), dtype='float32')

    for idx, frame in enumerate(range(start_frame, stop_frame)):
      mid_time = frame * self.stride
      time = mid_time + videomidi[0, 0] - audiomidi[0, 0]
      video_frame = int(time * fps + 0.5)
      video_feature[idx] = video_features[video_frame]

    return video_feature, audio_feature, onset_label


class VideoAudioSeqTestDataset(Dataset):
  def __init__(self, seq_len, filename, video_feature_dir, audio_feature_dir, video_label_dir, audio_label_dir,
                fps_config_path, successive_frame, stride):
    super(VideoAudioSeqTestDataset, self).__init__()
    
    with open(fps_config_path) as f:
      fps_config = json.load(f)

    video_feature_path = os.path.join(video_feature_dir, filename)
    audio_feature_path = os.path.join(audio_feature_dir, filename)
    video_feature = np.load(video_feature_path)
    self.video_feature = np.pad(video_feature, ((successive_frame // 2, 0), (0, 0)))
    self.audio_feature = np.load(audio_feature_path)
    filename = filename.replace('.npy', '')
    self.fps = fps_config[filename]
    video_mid_time = (len(self.video_feature) - 1 + successive_frame // 2) / self.fps
    audio_mid_time = (len(self.audio_feature) - 1) * stride

    videolabelpath = os.path.join(video_label_dir, filename+'.txt')
    audiolabelpath = os.path.join(audio_label_dir, filename+'.txt')
    videomidi = np.loadtxt(videolabelpath).reshape([-1, 4])
    audiomidi = np.loadtxt(audiolabelpath).reshape([-1, 4])
    self.time_shift = videomidi[0, 0] - audiomidi[0, 0]
    sample_num = int(min(video_mid_time - self.time_shift, audio_mid_time) / stride + 1)

    self.seq_len = seq_len
    self.seq_num = int(math.ceil(sample_num / self.seq_len))
    self.video_label_dir = video_label_dir
    self.audio_label_dir = audio_label_dir
    self.time_around = stride
    self.video_feature_dir = video_feature_dir
    self.audio_feature_dir = audio_feature_dir
    self.successive_frame = successive_frame
    self.stride = stride

  def __len__(self):
    return self.seq_num
  
  def __getitem__(self, index):

    sample = index
    start_frame = sample * self.seq_len
    stop_frame = (sample + 1) * self.seq_len
    start_t = start_frame * self.stride - self.time_around
    stop_t = stop_frame * self.stride + self.time_around

    audio_feature = self.audio_feature[start_frame: stop_frame]
    if len(audio_feature) < self.seq_len:
      pad_len = self.seq_len - len(audio_feature)
      audio_feature = np.pad(audio_feature, ((0, pad_len), (0, 0)))

    T_v, video_dim = self.video_feature.shape
    video_feature = np.zeros((self.seq_len, video_dim), dtype='float32')

    for idx, frame in enumerate(range(start_frame, stop_frame)):
      mid_time = frame * self.stride
      time = mid_time + self.time_shift
      video_frame = int(time * self.fps + 0.5)
      if video_frame < len(self.video_feature):
        video_feature[idx] = self.video_feature[video_frame]

    return video_feature, audio_feature


class VideoDataset(Dataset):
  def __init__(self, image_dir, label_dir, fps_config_path, image_size, is_train,  
               valid_files=None, dump_path=None, shift_bottom=100, successive_frame=5, time_around=0.032):
    super(VideoDataset, self).__init__()

    filenames = os.listdir(image_dir) if valid_files is None else valid_files
    label_filenames = os.listdir(label_dir)
    
    assert len(filenames) <= len(label_filenames), 'image num: %d > label num: %d'%(len(filenames), len(label_filenames))

    with open(fps_config_path) as f:
      self.fps_config = json.load(f)

    self.filename_num = []
    for filename in filenames:
      if filename not in self.fps_config:
        raise ValueError(filename, 'fps has not been got!')
      if 'midiTestSet' in image_dir:
        sub_image_dir = os.path.join(image_dir, filename, 'images')
      else:
        sub_image_dir = os.path.join(image_dir, filename)
      self.filename_num.append([filename, len(os.listdir(sub_image_dir)) - successive_frame + 1])
    self.filename_num.sort(key=lambda x: -x[1])

    for i in range(1, len(self.filename_num)):
      self.filename_num[i][1] += self.filename_num[i-1][1]
    
    self.image_dir = image_dir
    self.label_dir = label_dir
    self.image_size = image_size
    self.is_train = is_train
    self.dump_path = dump_path
    self.shift_bottom = shift_bottom
    self.successive_frame = successive_frame
    self.time_around = time_around

    # 从1开始编号
    self.black_keys = [2, 5, 7, 10, 12, 14, 17, 19, 22, 24, 26, 29, 31, 34, 36, 38, 
                       41, 43, 46, 48, 50, 53, 55, 58, 60, 62, 65, 67, 70, 72, 74,
                       77, 79, 82, 84, 86]



  def get_image(self, filename, image_idx=None, img=None, to_gray=True):
    if img is None:
      if 'midiTestSet' in self.image_dir:
        imgpath = os.path.join(self.image_dir, filename, 'images', '%05d.jpg'%image_idx)
      else:
        imgpath = os.path.join(self.image_dir, filename, '%05d.jpg'%image_idx)
      if to_gray:
        img = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE)
      else:
        img = cv2.imread(imgpath)


    img = cv2.resize(img, self.image_size[::-1])
    return img

  # 添加transforms参数
  def __getitem__(self, index):

    filename = None
    sample = -1
    for i in range(len(self.filename_num)):
      if index < self.filename_num[i][1]:
        filename = self.filename_num[i][0]
        sample = index if i == 0 else index - self.filename_num[i-1][1]
        break

    if filename is None:
      raise ValueError('cannot find the required frame')


    imgs = []
    
    for i in range(self.successive_frame):
      img = self.get_image(filename, i + sample)
      imgs.append(img)
    
    imgs = np.stack(imgs, axis=0)[:, None]
    imgs = (imgs.astype('float32') / 255.0).astype('float32')

    fps = self.fps_config[filename]
    mid_time = (sample + self.successive_frame//2) / fps

    labelpath = os.path.join(self.label_dir, filename+'.txt')
    midi = np.loadtxt(labelpath).reshape([-1, 4])
    onset_labels = np.zeros(88, dtype=np.float32)

    for onset, offset, pitch, velocity in midi:
      if onset >= mid_time - self.time_around and onset <= mid_time + self.time_around:
        if move_aug:
          note = int(pitch) - 21 + move_key
          if note >=0 and note < 88: # 少数情况下手部不准导致数组越界
            onset_labels[note] = 1.
        else:          
          onset_labels[int(pitch)-21] = 1.

    if self.dump_path is not None:
      dump_data(imgs, onset_labels, self.dump_path, filename, mid_time, '{}'.format(move_key))

    return imgs, onset_labels

  def __len__(self):
    # return 2400
    return self.filename_num[-1][1]


class VideoTestDataset(Dataset):
  def __init__(self, image_dir, image_size, shift_bottom=100, successive_frame=5):
    super(VideoTestDataset, self).__init__()    
    if not os.path.exists(os.path.join(image_dir, '00000.jpg')):
      image_dir = os.path.join(image_dir, 'images')
    self.image_dir = image_dir
    self.image_size = image_size
    self.shift_bottom = shift_bottom
    self.successive_frame = successive_frame
    self.file_num = len(os.listdir(image_dir)) - successive_frame + 1

  def get_image(self, img_path, to_gray=True):
    if to_gray:
      img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    else:
      img = cv2.imread(img_path)
    
    img = cv2.resize(img, self.image_size[::-1])
    return img

  # 添加transforms参数
  def __getitem__(self, index):
    
    imgs = []
    for i in range(self.successive_frame):
      img_path = os.path.join(self.image_dir, '{:05d}.jpg'.format(i + index))
      img = self.get_image(img_path)
      imgs.append(img)
    imgs = np.stack(imgs, axis=0)[:, None]
    imgs = (imgs.astype('float32') / 255.0).astype('float32')

    return imgs

  def __len__(self):
    return self.file_num

def clean_key_points(key_points_config, y1, hight):

  # 会出现大于2只手的情况，需要去除不在键盘上的手
  assert len(key_points_config) % 21 == 0, 'key points should be 21 times!'
  hands = len(key_points_config) // 21

  new_key_points_config = dict()
  point = 0
  for hand in range(hands):
    points_in = points_out = 0
    for i in range(hand * 21, (hand + 1) * 21):
      y = hight * key_points_config[str(i)]['y']
      if y < y1:
        points_out += 1
      else:
        points_in += 1
    if points_in > points_out:
      for i in range(hand * 21, (hand + 1) * 21):
        new_key_points_config[str(point)] = key_points_config[str(i)]
        point += 1

  # 手的数量还是>3(重影造成) 选择手腕最左边和最右边的的两只手
  if len(new_key_points_config) > 42:
    left_hand_idx = -1
    right_hand_idx = -1
    left = float('inf')
    right = 0
    for i in range(0, len(new_key_points_config), 21):
      if new_key_points_config[str(i)]['x'] < left:
        left_hand_idx = i
        left = new_key_points_config[str(i)]['x']
      if new_key_points_config[str(i)]['x'] > right:
        right_hand_idx = i
        right = new_key_points_config[str(i)]['x']
    assert left_hand_idx != -1 and right_hand_idx != -1

    key_points_config = dict()
    no = 0
    for i in range(left_hand_idx, left_hand_idx+21):
      key_points_config[str(no)] = new_key_points_config[str(i)]
      no += 1
    for i in range(right_hand_idx, right_hand_idx+21):
      key_points_config[str(no)] = new_key_points_config[str(i)]
      no += 1
  else:
    key_points_config = new_key_points_config

  if key_points_config:
    # 标注中的label Left Right有bug，需要自己纠正
    if len(key_points_config) == 21:
      # 小拇指在大拇指右边，右手
      if key_points_config['20']['x'] > key_points_config['4']['x']:
        for key in key_points_config:
          key_points_config[key]['label'] = 'Right'
      else:
        for key in key_points_config:
          key_points_config[key]['label'] = 'Left'
    elif len(key_points_config) == 42:
      # 0-20右手 21-41左手
      if key_points_config['0']['x'] > key_points_config['21']['x']:
        for key in range(0, 21):
          key_points_config[str(key)]['label'] = 'Right'
        for key in range(21, 42):
          key_points_config[str(key)]['label'] = 'Left'
      # 0-20左手 21-41右手
      else:
        for key in range(0, 21):
          key_points_config[str(key)]['label'] = 'Left'
        for key in range(21, 42):
          key_points_config[str(key)]['label'] = 'Right'
    else:
      raise ValueError('key points should be 21 or 42, but get %d!'%len(key_points_config))
    
    return key_points_config


# onset标签进行扩散 frame不动 offset标签也进行扩散3帧
# 滑窗每次选择帧数 使用一部分历史 一部分将来信息
class AudioDataset(Dataset):
  def __init__(self, audio_dir, label_dir, is_train, fmin, bins_per_octave,
                frame, stride, sr, dump_path=None, ignore_offset_head=False,
                ignore_velocity_head=False):
    super(AudioDataset, self).__init__()

    filenames = os.listdir(audio_dir)
    label_filenames = os.listdir(label_dir)
    
    assert len(filenames) <= len(label_filenames), 'image num: %d > label num: %d'%(len(filenames), len(label_filenames))

    n_fft = sr / fmin / (2** (1 / bins_per_octave) - 1)
    n_fft = 2 ** int(math.ceil(math.log2(n_fft))) # 65536 4.096s
    window_t = (frame - 1) * stride + n_fft / sr # 4.216s

    self.filename_num = []
    for filename in filenames:
      audio_path = os.path.join(audio_dir, filename)
      audio_duration = librosa.get_duration(filename=audio_path)
      samples = int((audio_duration - window_t) // stride + 1)
      self.filename_num.append([filename, samples])
    self.filename_num.sort(key=lambda x: -x[1])

    for i in range(1, len(self.filename_num)):
      self.filename_num[i][1] += self.filename_num[i-1][1]

    self.audio_dir = audio_dir
    self.label_dir = label_dir
    self.is_train = is_train
    self.stride = stride
    self.sr = sr
    self.time_around = stride
    self.dump_path = dump_path
    self.window_t = window_t
    self.ignore_offset_head = ignore_offset_head
    self.ignore_velocity_head = ignore_velocity_head

  def __len__(self):
    return self.filename_num[-1][1]
    # return 28

  def __getitem__(self, index):
    filename = None
    sample = -1
    for i in range(len(self.filename_num)):
      if index < self.filename_num[i][1]:
        filename = self.filename_num[i][0]
        sample = index if i == 0 else index - self.filename_num[i-1][1]
        break
    
    if filename is None:
      raise ValueError('cannot find the required frame')

    start_t = sample* self.stride
    start = int(start_t * self.sr)
    stop = start + int(self.window_t * self.sr)
    audio, sr = sf.read(os.path.join(self.audio_dir, filename), start=start, 
                          stop=stop, dtype='float32')
    assert sr == self.sr, 'audio sr: %d, should resample to %d first!'%(sr, self.sr)
    mid_time = start_t + self.window_t / 2 # 2.108

    label_path = os.path.join(self.label_dir, filename.replace('.wav', '.txt'))
    label = np.loadtxt(label_path, dtype='float32').reshape(-1, 4)
    onset_label = np.zeros(88, dtype='float32')
    if not self.ignore_offset_head:
      offset_label = np.zeros(88, dtype='float32')
    frame_label = np.zeros(88, dtype='float32')
    if not self.ignore_velocity_head:
      velocity_label = np.zeros(88, dtype='float32')

    for onset, offset, pitch, velocity in label:
      if onset >= mid_time - self.time_around and onset <= mid_time + self.time_around:
        onset_label[int(pitch) - 21] = 1.
        if not self.ignore_velocity_head:
          velocity_label[int(pitch) - 21] = velocity / 128.
      if not self.ignore_offset_head:
        if offset >= mid_time - self.time_around and offset <= mid_time + self.time_around:
          offset_label[int(pitch) - 21] = 1.
      if onset <= mid_time + self.time_around and offset >= mid_time - self.time_around:
        frame_label[int(pitch) - 21] = 1.

    if self.ignore_offset_head:
      return audio, onset_label, frame_label

    if self.dump_path is not None:
      dump_audio_data(audio, self.sr, onset_label, offset_label, frame_label, self.dump_path, 
                      filename, mid_time, self.time_around)
    
    if self.ignore_velocity_head:
      return audio, onset_label, offset_label, frame_label
    
    return audio, onset_label, offset_label, frame_label, velocity_label
    
class AudioTestset(Dataset):
  def __init__(self, audio_path, fmin, bins_per_octave,
                frame, stride, sr):
    super(AudioTestset, self).__init__()

    self.audio, _sr = sf.read(audio_path, dtype='float32')
    assert sr == _sr, 'audio sr: %d, should resample to %d first'%(_sr, sr)

    n_fft = sr / fmin / (2** (1 / bins_per_octave) - 1) # 40000
    n_fft = 2 ** int(math.ceil(math.log2(n_fft))) # 65536 4.096s
    window_t = (frame - 1) * stride + n_fft / sr # 4.216s

    self.audio = np.concatenate([np.zeros(int(window_t*sr/2), dtype='float32'),
                            self.audio], axis=0)

    audio_duration = len(self.audio) / sr
    self.length = int((audio_duration - window_t)//stride + 1)
    self.window_t = window_t
    self.stride = stride
    self.sr = sr
  
  def __len__(self):
    return self.length
  
  def __getitem__(self, index):
    start_t = index*self.stride
    start = int(start_t * self.sr)
    stop = start + int(self.window_t * self.sr)
    return self.audio[start: stop]

class MAESTRODataset(Dataset):
  def __init__(self, dataset_dir, csv_name, is_train, fmin, bins_per_octave,
                frame, stride, sr, dump_path=None, ignore_offset_head=False,
                ignore_velocity_head=False, load_static=None):
    super(MAESTRODataset, self).__init__()

    if load_static:
      if dump_path is not None:
        dump_father_dir = os.path.dirname(dump_path)
        filepaths = []
        for filename in os.listdir(dump_father_dir):
          dump_path = os.path.join(dump_father_dir, filename)
          subfilepaths = [os.path.join(dump_path, filename) for filename in os.listdir(dump_path)]
          filepaths.extend(subfilepaths)
        self.static_data = filepaths
      else:
        self.static_data = [0 for i in range(10)]
    else:
      csv_path = os.path.join(dataset_dir, csv_name)
      csv_data = pd.read_csv(csv_path)
      if is_train:
        data_info = csv_data.loc[csv_data['split'] == 'train']
      else:
        data_info = csv_data.loc[csv_data['split'] == 'validation']

      filenames = list(data_info['audio_filename'])
      label_filenames = list(data_info['midi_filename'])
      
      assert len(filenames) <= len(label_filenames), 'image num: %d > label num: %d'%(len(filenames), len(label_filenames))

      n_fft = sr / fmin / (2** (1 / bins_per_octave) - 1)
      n_fft = 2 ** int(math.ceil(math.log2(n_fft))) # 65536 4.096s
      window_t = (frame - 1) * stride + n_fft / sr # 4.216s

      self.filename_num = []
      audio_durations = csv_data['duration']
      for filename in filenames:
        audio_duration = audio_durations.loc[csv_data['audio_filename'] == filename]
        audio_duration = float(audio_duration)
        samples = int((audio_duration - window_t) // stride + 1)
        self.filename_num.append([filename, samples])
      self.filename_num.sort(key=lambda x: -x[1])

      for i in range(1, len(self.filename_num)):
        self.filename_num[i][1] += self.filename_num[i-1][1]
      
      self.window_t = window_t

    self.dataset_dir = dataset_dir
    self.is_train = is_train
    self.stride = stride
    self.sr = sr
    self.time_around = stride
    self.dump_path = dump_path
    self.ignore_offset_head = ignore_offset_head
    self.ignore_velocity_head = ignore_velocity_head
    self.load_static = load_static

  def __len__(self):
    if self.load_static:
      return len(self.static_data)
    return self.filename_num[-1][1]

  def __getitem__(self, index):

    if self.load_static:
      filepath = self.static_data[index]
      data = np.load(filepath, allow_pickle=True).item()
      return data['audio'], data['onset_label'], data['offset_label'], data['frame_label']

    dump_path = os.path.join(self.dump_path, '%08d.npy'%index)
    if os.path.exists(dump_path):
      return    

    filename = None
    sample = -1
    for i in range(len(self.filename_num)):
      if index < self.filename_num[i][1]:
        filename = self.filename_num[i][0]
        sample = index if i == 0 else index - self.filename_num[i-1][1]
        break
    
    if filename is None:
      raise ValueError('cannot find the required frame')

    start_t = sample* self.stride
    start = int(start_t * self.sr)
    stop = start + int(self.window_t * self.sr)
    audio_path = os.path.join(self.dataset_dir, filename)
    dirname = os.path.dirname(audio_path)
    basename = os.path.basename(audio_path)
    audio_path = os.path.join(dirname, '16k_' + basename)
    audio, sr = sf.read(audio_path, start=start, stop=stop, dtype='float32')
    assert sr == self.sr, 'audio sr: %d, should resample to %d first!'%(sr, self.sr)
    mid_time = start_t + self.window_t / 2 # 2.108

    label_path = os.path.join(self.dataset_dir, filename.replace('.wav', '.midi'))
    ns = midi_io.midi_file_to_note_sequence(label_path)
    sequence = sequences_lib.apply_sustain_control_changes(ns)

    onset_label = np.zeros(88, dtype='float32')
    if not self.ignore_offset_head:
      offset_label = np.zeros(88, dtype='float32')
    frame_label = np.zeros(88, dtype='float32')
    if not self.ignore_velocity_head:
      velocity_label = np.zeros(88, dtype='float32')

    for note in sequence.notes:
      onset, offset, pitch, velocity = note.start_time, note.end_time, note.pitch, note.velocity
      if onset >= mid_time - self.time_around and onset <= mid_time + self.time_around:
        onset_label[int(pitch) - 21] = 1.
        if not self.ignore_velocity_head:
          velocity_label[int(pitch) - 21] = velocity / 128.
      if not self.ignore_offset_head:
        if offset >= mid_time - self.time_around and offset <= mid_time + self.time_around:
          offset_label[int(pitch) - 21] = 1.
      if onset <= mid_time + self.time_around and offset >= mid_time - self.time_around:
        frame_label[int(pitch) - 21] = 1.

    if self.ignore_offset_head:
      return audio, onset_label, frame_label

    if self.dump_path is not None and self.load_static is None:
      dump_audio_data(audio, self.sr, onset_label, offset_label, frame_label, self.dump_path, 
                      filename, mid_time, self.time_around)
    
    if self.ignore_velocity_head:
      data_dict = {
        'audio': audio,
        'onset_label': onset_label,
        'offset_label': offset_label,
        'frame_label': frame_label
      }
      dump_raw_data(dump_path, data_dict)
      return audio, onset_label, offset_label, frame_label
    return audio, onset_label, offset_label, frame_label, velocity_label


class MAESTRODatasetV2(Dataset):
  def __init__(self, dump_path=None):
    super(MAESTRODatasetV2, self).__init__()

    if dump_path is not None:
      self.static_data = [os.path.join(dump_path, filename) for filename in os.listdir(dump_path)]
    else:
      self.static_data = [0 for _ in range(10)]

  def __len__(self):
    return len(self.static_data)
    # return len(self.static_data) // 2

  def __getitem__(self, index):
      
    audios, onset_labels, offset_labels, frame_labels = [], [], [], []
    for k in [index]:
    # for k in [2 * index, 2 * index+1]:
      filepath = self.static_data[k]
      data = np.load(filepath, allow_pickle=True).item()
      audio = data['audio']
      onset_label = data['onset_label']
      offset_label = data['offset_label']
      frame_label = data['frame_label']
      audios.append(audio)
      onset_labels.append(onset_label)
      offset_labels.append(offset_label)
      frame_labels.append(frame_label)
    audios = np.concatenate(audios, axis=0)
    onset_labels = np.concatenate(onset_labels, axis=0)
    offset_labels = np.concatenate(offset_labels, axis=0)
    frame_labels = np.concatenate(frame_labels, axis=0)

    return audios, onset_labels, offset_labels, frame_labels

