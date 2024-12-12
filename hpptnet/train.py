import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import all_reduce
import torch
import os

def train(model, train_config, log_config, logger, writer):

  opt = train_config['opt']
  lr_scheduler = train_config['lr_scheduler']
  learning_rate_decay_step = train_config['lr_decay_step']
  train_loss_cls = train_config['train_loss_cls']
  val_loss_cls = train_config['val_loss_cls']
  train_metric_cls = train_config['train_metric_cls']
  val_metric_cls = train_config['val_metric_cls']
  train_loader = train_config['train_loader']
  val_loader = train_config['val_loader']
  train_sampler = train_config['train_sampler']
  epochs = train_config['epochs']
  rank = train_config['rank']
  world_size = train_config['world_size']
  device = train_config['device']
  save_checkpoint_dir = train_config['save_checkpoint_dir']

  if rank == 0:
    os.makedirs(save_checkpoint_dir, exist_ok=True)

  train_log_interval = log_config['train_log_interval']
  val_interval = log_config['val_interval']
  val_log_interval = log_config['val_log_interval']
  save_log_interval = log_config['save_log_interval']
  start_val_step = log_config['start_val_step']

  train_num = len(train_loader)
  valid_num = len(val_loader)

  total_step = 0
  for epoch in range(epochs):
    if rank == 0:
      logger.info("======== New Epoch ========")
    train_sampler.set_epoch(epoch)
    step = 0
    train_loss_cls.zero()
    train_metric_cls.zero()

    for audio, onset_labels, offset_labels, frame_labels in train_loader:
      audio = audio[0]
      onset_labels = onset_labels[0]
      offset_labels = offset_labels[0]
      frame_labels = frame_labels[0]
      model.train()
      audio = audio.to(device)
      onset_labels = onset_labels.to(device)
      offset_labels = offset_labels.to(device)
      frame_labels = frame_labels.to(device)

      onset_logits, offset_logits, frame_logits = model(audio)
      loss, onset_loss, offset_loss, frame_loss = train_loss_cls(onset_logits, offset_logits, 
                      frame_logits, onset_labels, offset_labels, frame_labels)
      smoth_loss, smoth_onset_loss, smoth_offset_loss, smoth_frame_loss = train_loss_cls.step()

      onset_p, onset_r, onset_f1, offset_p, offset_r, offset_f1, \
        frame_p, frame_r, frame_f1 = train_metric_cls(onset_logits, offset_logits, 
          frame_logits, onset_labels, offset_labels, frame_labels)
      avg_onset_p, avg_onset_r, avg_onset_f1, avg_offset_p, avg_offset_r, avg_offset_f1, \
        avg_frame_p, avg_frame_r, avg_frame_f1 = train_metric_cls.step()

      opt.zero_grad()
      loss.backward()
      opt.step()

      loss_config = {
        'loss': loss.detach(),
        'onset_loss': onset_loss.detach(),
        'offset_loss': offset_loss.detach(),
        'frame_loss': frame_loss.detach(),
        'smoth_loss': smoth_loss,
        'smoth_onset_loss': smoth_onset_loss,
        'smoth_offset_loss': smoth_offset_loss, # 有问题
        'smoth_frame_loss': smoth_frame_loss # 有问题
      }
      metric_config = {
        'onset_p': onset_p,
        'onset_r': onset_r,
        'onset_f1': onset_f1,
        'offset_p': offset_p,
        'offset_r': offset_r,
        'offset_f1': offset_f1,
        'frame_p': frame_p,
        'frame_r': frame_r,
        'frame_f1': frame_f1,
        'smoth_onset_p': avg_onset_p,
        'smoth_onset_r': avg_onset_r,
        'smoth_onset_f1': avg_onset_f1,
        'smoth_offset_p': avg_offset_p,
        'smoth_offset_r': avg_offset_r,
        'smoth_offset_f1': avg_offset_f1,
        'smoth_frame_p': avg_frame_p,
        'smoth_frame_r': avg_frame_r,
        'smoth_frame_f1': avg_frame_f1
      }
      data_config = {
        **loss_config,
        **metric_config
      }
      all_reduce(data_config, world_size)

      if total_step % train_log_interval == 0 and rank == 0:
        # 日志
        log1 = 'Epoch: %02d / %02d Step: %05d / %05d Total_Step: %05d Loss: %.3f Smoth Loss: %.3f'% \
                (epoch, epochs, step, train_num, total_step, data_config['loss'], data_config['smoth_loss'])
        log2 = 'Loss onset: %.3f offset: %.3f frame: %.3f Smoth onset: %.3f offset: %.3f frame: %.3f'% \
                (data_config['onset_loss'], data_config['offset_loss'], data_config['frame_loss'],
                data_config['smoth_onset_loss'], data_config['smoth_offset_loss'], data_config['smoth_frame_loss'])
        log3 = 'Onset P: %.2f%% R: %.2f%% F1: %.2f%% Smoth Onset: P: %.2f%% R: %.2f%% F1: %.2f%%'% \
                (data_config['onset_p'], data_config['onset_r'], data_config['onset_f1'],
                data_config['smoth_onset_p'], data_config['smoth_onset_r'], data_config['smoth_onset_f1'])
        log4 = 'Offset P: %.2f%% R: %.2f%% F1: %.2f%% Smoth Offset: P: %.2f%% R: %.2f%% F1: %.2f%%'% \
                (data_config['offset_p'], data_config['offset_r'], data_config['offset_f1'],
                data_config['smoth_offset_p'], data_config['smoth_offset_r'], data_config['smoth_offset_f1'])
        log5 = 'Frame P: %.2f%% R: %.2f%% F1: %.2f%% Smoth Frame: P: %.2f%% R: %.2f%% F1: %.2f%%'% \
                (data_config['frame_p'], data_config['frame_r'], data_config['frame_f1'],
                data_config['smoth_frame_p'], data_config['smoth_frame_r'], data_config['smoth_frame_f1'])
    
        logger.info(log1)
        logger.info(log2)
        logger.info(log3)
        logger.info(log4)
        logger.info(log5)

        for name, value in data_config.items():
          if 'onset' in name:
            writer.add_scalar('train/onset/' + name, value, total_step)
          elif 'offset' in name:
            writer.add_scalar('train/offset/' + name, value, total_step)
          elif 'frame' in name:
            writer.add_scalar('train/frame/' + name, value, total_step)
          else:
            writer.add_scalar('train/global/' + name, value, total_step)

      if total_step % val_interval == 0 and total_step >= start_val_step:
        model.eval()
        data_config = None
        val_step = 0
        val_loss_cls.zero()
        val_metric_cls.zero()
        with torch.no_grad():
          for audio, onset_labels, offset_labels, frame_labels in val_loader:
            audio = audio.to(device)
            onset_labels = onset_labels.to(device)
            offset_labels = offset_labels.to(device)
            frame_labels = frame_labels.to(device)

            onset_logits, offset_logits, frame_logits =  model(audio)
            loss, onset_loss, offset_loss, frame_loss = val_loss_cls(onset_logits, offset_logits, 
                frame_logits, onset_labels, offset_labels, frame_labels)

            smoth_loss, smoth_onset_loss, smoth_offset_loss, smoth_frame_loss = val_loss_cls.step()
            onset_p, onset_r, onset_f1, offset_p, offset_r, offset_f1, \
              frame_p, frame_r, frame_f1 =  val_metric_cls(onset_logits, offset_logits, 
                  frame_logits, onset_labels, offset_labels, frame_labels)

            avg_onset_p, avg_onset_r, avg_onset_f1, avg_offset_p, avg_offset_r, avg_offset_f1, \
              avg_frame_p, avg_frame_r, avg_frame_f1 = val_metric_cls.step()

            loss_config = {
              'loss': loss.detach(),
              'onset_loss': onset_loss.detach(),
              'offset_loss': offset_loss.detach(),
              'frame_loss': frame_loss.detach(),
              'smoth_loss': smoth_loss,
              'smoth_onset_loss': smoth_onset_loss,
              'smoth_offset_loss': smoth_offset_loss,
              'smoth_frame_loss': smoth_frame_loss
            }
            metric_config = {
              'onset_p': onset_p,
              'onset_r': onset_r,
              'onset_f1': onset_f1,
              'offset_p': offset_p,
              'offset_r': offset_r,
              'offset_f1': offset_f1,
              'frame_p': frame_p,
              'frame_r': frame_r,
              'frame_f1': frame_f1,
              'smoth_onset_p': avg_onset_p,
              'smoth_onset_r': avg_onset_r,
              'smoth_onset_f1': avg_onset_f1,
              'smoth_offset_p': avg_offset_p,
              'smoth_offset_r': avg_offset_r,
              'smoth_offset_f1': avg_offset_f1,
              'smoth_frame_p': avg_frame_p,
              'smoth_frame_r': avg_frame_r,
              'smoth_frame_f1': avg_frame_f1
            }
            data_config = {
              **loss_config,
              **metric_config
            }
            all_reduce(data_config, world_size)

            if val_step % val_log_interval == 0 and rank == 0:

              log1 = 'Evaluate: %02d / %02d Loss: %.3f Smoth loss: %.3f'% \
                      (val_step, valid_num, data_config['loss'], data_config['smoth_loss'])
              log2 = 'Evaluate Loss onset: %.3f offset: %.3f frame: %.3f Smoth onset: %.3f offset: %.3f frame: %.3f'% \
                      (data_config['onset_loss'], data_config['offset_loss'], data_config['frame_loss'],
                      data_config['smoth_onset_loss'], data_config['smoth_offset_loss'], data_config['smoth_frame_loss'])
              log3 = 'Evaluate Onset P: %.2f%% R: %.2f%% F1: %.2f%% Smoth Onset: P: %.2f%% R: %.2f%% F1: %.2f%%'% \
                      (data_config['onset_p'], data_config['onset_r'], data_config['onset_f1'],
                      data_config['smoth_onset_p'], data_config['smoth_onset_r'], data_config['smoth_onset_f1'])
              log4 = 'Evaluate Offset P: %.2f%% R: %.2f%% F1: %.2f%% Smoth Offset: P: %.2f%% R: %.2f%% F1: %.2f%%'% \
                      (data_config['offset_p'], data_config['offset_r'], data_config['offset_f1'],
                      data_config['smoth_offset_p'], data_config['smoth_offset_r'], data_config['smoth_offset_f1'])
              log5 = 'Evaluate Frame P: %.2f%% R: %.2f%% F1: %.2f%% Smoth Frame: P: %.2f%% R: %.2f%% F1: %.2f%%'% \
                      (data_config['frame_p'], data_config['frame_r'], data_config['frame_f1'],
                      data_config['smoth_frame_p'], data_config['smoth_frame_r'], data_config['smoth_frame_f1'])
              
              logger.info("======== Evaluate ========")
              logger.info(log1)
              logger.info(log2)
              logger.info(log3)
              logger.info(log4)
              logger.info(log5)

            val_step += 1

        if rank == 0:
          log1 = 'Evalate Epoch: %02d / %02d Step: %05d / %05d Total_Step: %05d'% \
                  (epoch, epochs, step, train_num, total_step)

          log2 = 'Evalate Smoth_loss: total %.3f onset %.3f offset %.3f frame %.3f'% (data_config['smoth_loss'], 
                  data_config['smoth_onset_loss'], data_config['smoth_offset_loss'], data_config['smoth_frame_loss'])
          
          log3 = ('Evaluate Avg_metric: Onset P: %.2f%% R: %.2f%% F: %.2f%% Offset P: %.2f%% R: %.2f%% F: %.2f%% '
                  'Frame P: %.2f%% R: %.2f%% F: %.2f%%'%(data_config['smoth_onset_p'], data_config['smoth_onset_r'],
                  data_config['smoth_onset_f1'], data_config['smoth_offset_p'], data_config['smoth_offset_r'],
                  data_config['smoth_offset_f1'], data_config['smoth_frame_p'], data_config['smoth_frame_r'],
                  data_config['smoth_frame_f1']))

          logger.info("======== Evaluate ========")
          logger.info(log1)
          logger.info(log2)
          logger.info(log3)

          for name, value in data_config.items():
            if 'smoth' in name:
              if 'onset' in name:
                writer.add_scalar('valid/onset/' + name, value, total_step)
              elif 'offset' in name:
                writer.add_scalar('valid/offset/' + name, value, total_step)
              elif 'frame' in name:
                writer.add_scalar('valid/frame/' + name, value, total_step)
              else:
                writer.add_scalar('valid/global/' + name, value, total_step)

      if total_step % save_log_interval == 0 and rank == 0:
        log1 = 'Epoch: %02d / %02d Step: %05d / %05d Total_Step: %05d'% \
                  (epoch, epochs, step, train_num, total_step)
        log2 = 'Saving checkpoint'
        logger.info(log1)
        logger.info(log2)
        checkpoint_path = os.path.join(save_checkpoint_dir, 'epoch_%02d_total_step_%02d.pt' % (epoch, total_step))
        learning_rate = opt.param_groups[0]['lr']
        torch.save(
          {
            'epoch': epoch,
            'train_num': train_num,
            'total_step': total_step,
            'model_state_dict': model.module.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'learning_rate': learning_rate
          },
          checkpoint_path
          )

      if total_step % learning_rate_decay_step == 0:
        lr_scheduler.step()
        if rank == 0:
          learning_rate = lr_scheduler.get_last_lr()[0]
          log1 = 'Epoch: %02d / %02d Step: %05d / %05d Total_Step: %05d'% \
                    (epoch, epochs, step, train_num, total_step)
          log2 = 'Learning rate decay to: %.7f' % learning_rate
          logger.info(log1)
          logger.info(log2)
          writer.add_scalar('train/lr', learning_rate, total_step)

      step += 1
      total_step += 1