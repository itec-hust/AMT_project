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

    for datas in train_loader:
      model.train()
      datas = [data.to(device) for data in datas]
      onset_labels = datas[-1]
      features = datas[: -1]
      conv_logits, trans_logits =  model(*features)
      loss, conv_loss, trans_loss = train_loss_cls(conv_logits, trans_logits, onset_labels)
      smoth_loss, smoth_conv_loss, smoth_trans_loss = train_loss_cls.step()

      p, r, f1, conv_p, conv_r, conv_f1, trans_p, trans_r, trans_f1 = \
            train_metric_cls(conv_logits, trans_logits, onset_labels)
      avg_p, avg_r, avg_f1, avg_conv_p, avg_conv_r, avg_conv_f1, \
          avg_trans_p, avg_trans_r, avg_trans_f1 = train_metric_cls.step()

      opt.zero_grad()
      loss.backward()
      opt.step()

      # loss
      loss_config = {
        'loss': loss.detach(), 
        'conv_loss': conv_loss.detach(), 
        'tran_loss': trans_loss.detach(), 
        'smoth_loss': smoth_loss, 
        'smoth_conv_loss': smoth_conv_loss, 
        'smoth_trans_loss': smoth_trans_loss,
      }
      # p r f1
      metric_config = {
        'p': p,
        'r': r,
        'f1': f1,
        'conv_p': conv_p,
        'conv_r': conv_r,
        'conv_f1': conv_f1,
        'trans_p': trans_p,
        'trans_r': trans_r,
        'trans_f1': trans_f1
      }
      # avg p r f1
      avg_metric_config = {
        'avg_p': avg_p,
        'avg_r': avg_r,
        'avg_f1': avg_f1,
        'avg_conv_p': avg_conv_p,
        'avg_conv_r': avg_conv_r,
        'avg_conv_f1': avg_conv_f1,
        'avg_trans_p': avg_trans_p,
        'avg_trans_r': avg_trans_r,
        'avg_trans_f1': avg_trans_f1
      }

      data_config = {**loss_config, **metric_config, **avg_metric_config}
      all_reduce(data_config, world_size)

      if total_step % train_log_interval == 0 and rank == 0:
        # 日志
        log1 = 'Epoch: %02d / %02d Step: %05d / %05d Total_Step: %05d'% \
                (epoch, epochs, step, train_num, total_step)
        log2 = 'Loss: %.6f Smoth_loss: %.6f P: %.2f%% R: %.2f%% F1: %.2f%% Avg_P: %.2f%% ' \
               'Avg_R: %.2f%% Avg_F1: %.2f%%'% (data_config['loss'], data_config['smoth_loss'], \
                data_config['p'], data_config['r'], data_config['f1'], data_config['avg_p'], \
                data_config['avg_r'], data_config['avg_f1'])
        logger.info(log1)
        logger.info(log2)
        for name, value in data_config.items():
          if 'conv' in name:
            writer.add_scalar('train/conv/' + name, value, total_step)
          elif 'trans' in name:
            writer.add_scalar('train/trans/' + name, value, total_step)
          else:
            writer.add_scalar('train/out/' + name, value, total_step)

      if total_step % val_interval == 0 and total_step >= start_val_step:
        model.eval()
        data_config = None
        val_step = 0
        val_loss_cls.zero()
        val_metric_cls.zero()
        with torch.no_grad():
          for datas in val_loader:
            datas = [data.to(device) for data in datas]
            onset_labels = datas[-1]
            features = datas[: -1]
            conv_logits, trans_logits =  model(*features)
            loss, conv_loss, trans_loss = val_loss_cls(conv_logits, trans_logits, onset_labels)
            smoth_loss, smoth_conv_loss, smoth_trans_loss = val_loss_cls.step()
            p, r, f1, conv_p, conv_r, conv_f1, trans_p, trans_r, trans_f1 = \
                val_metric_cls(conv_logits, trans_logits, onset_labels)
            avg_p, avg_r, avg_f1, avg_conv_p, avg_conv_r, avg_conv_f1, \
                avg_trans_p, avg_trans_r, avg_trans_f1= val_metric_cls.step()

            # loss
            loss_config = {
              'loss': loss.detach(), 
              'conv_loss': conv_loss.detach(), 
              'tran_loss': trans_loss.detach(), 
              'smoth_loss': smoth_loss, 
              'smoth_conv_loss': smoth_conv_loss, 
              'smoth_trans_loss': smoth_trans_loss
            }
            # p r f1
            metric_config = {
              'p': p,
              'r': r,
              'f1': f1,
              'conv_p': conv_p,
              'conv_r': conv_r,
              'conv_f1': conv_f1,
              'trans_p': trans_p,
              'trans_r': trans_r,
              'trans_f1': trans_f1
            }
            # avg p r f1
            avg_metric_config = {
              'avg_p': avg_p,
              'avg_r': avg_r,
              'avg_f1': avg_f1,
              'avg_conv_p': avg_conv_p,
              'avg_conv_r': avg_conv_r,
              'avg_conv_f1': avg_conv_f1,
              'avg_trans_p': avg_trans_p,
              'avg_trans_r': avg_trans_r,
              'avg_trans_f1': avg_trans_f1
            }

            data_config = {**loss_config, **metric_config, **avg_metric_config}
            all_reduce(data_config, world_size)

            if val_step % val_log_interval == 0 and rank == 0:

              log1 = 'Evaluate: %02d / %02d'% (val_step, valid_num)
              log2 = 'Evaluate Loss: %.6f Smoth_loss: %.6f P: %.2f%% R: %.2f%% F1: %.2f%% Avg_P: %.2f%% ' \
               'Avg_R: %.2f%% Avg_F1: %.2f%%'% (data_config['loss'], data_config['smoth_loss'], \
                data_config['p'], data_config['r'], data_config['f1'], data_config['avg_p'], \
                data_config['avg_r'], data_config['avg_f1'])
              logger.info(log1)
              logger.info(log2)

            val_step += 1

        if rank == 0:
          log1 = 'Evalate Epoch: %02d / %02d Step: %05d / %05d Total_Step: %05d'% \
                  (epoch, epochs, step, train_num, total_step)
          log2 = 'Evalate Smoth_loss: %.6f Avg_P: %.2f%% Avg_R: %.2f%% Avg_F1: %.2f%%'% (data_config['smoth_loss'], 
                  data_config['avg_p'], data_config['avg_r'], data_config['avg_f1'])
          logger.info(log1)
          logger.info(log2)
          # loss
          for name in ['smoth_loss', 'smoth_conv_loss', 'smoth_trans_loss']:
            value = data_config[name]
            if 'conv' in name:
              writer.add_scalar('valid/conv/' + name, value, total_step)
            elif 'trans' in name:
              writer.add_scalar('valid/trans/' + name, value, total_step)
            else:
              writer.add_scalar('valid/out/' + name, value, total_step)

          for name in ['avg_p', 'avg_r', 'avg_f1', 'avg_conv_p', 'avg_conv_r', 'avg_conv_f1',
                        'avg_trans_p', 'avg_trans_r', 'avg_trans_f1']:
            value = data_config[name]
            if 'conv' in name:
              writer.add_scalar('valid/conv/' + name, value, total_step)
            elif 'trans' in name:
              writer.add_scalar('valid/trans/' + name, value, total_step)
            else:
              writer.add_scalar('valid/out/' + name, value, total_step)

      if total_step % save_log_interval == 0 and rank == 0: #and total_step >= start_val_step:
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