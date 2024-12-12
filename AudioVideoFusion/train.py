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
      logits =  model(*features)
      onset_labels = onset_labels.flatten()
      logits = logits.flatten()

      loss = train_loss_cls(logits, onset_labels)
      smoth_loss = train_loss_cls.step()

      p, r, f1 = train_metric_cls(logits, onset_labels)
      avg_p, avg_r, avg_f1 = train_metric_cls.step()

      opt.zero_grad()
      loss.backward()
      opt.step()

      # 变量聚合
      data_config = {
        'loss': loss.detach(),
        'smoth_loss': smoth_loss,
        'p': p,
        'r': r,
        'f1': f1,
        'avg_p': avg_p,
        'avg_r': avg_r,
        'avg_f1': avg_f1
      }
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
        writer.add_scalar('train/loss', data_config['loss'], total_step)
        writer.add_scalar('train/smoth_loss', data_config['smoth_loss'], total_step)
        writer.add_scalar('train/p', data_config['p'], total_step)
        writer.add_scalar('train/r', data_config['r'], total_step)
        writer.add_scalar('train/f1', data_config['f1'], total_step)
        writer.add_scalar('train/avg_p', data_config['avg_p'], total_step)
        writer.add_scalar('train/avg_r', data_config['avg_r'], total_step)
        writer.add_scalar('train/avg_f1', data_config['avg_f1'], total_step)

      if total_step % val_interval == 0 and total_step >= start_val_step:
        model.eval()
        smoth_loss = None
        avg_p = avg_r = avg_f1 = None
        data_config = None
        val_step = 0
        val_loss_cls.zero()
        val_metric_cls.zero()
        with torch.no_grad():
          for datas in val_loader:
            datas = [data.to(device) for data in datas]
            onset_labels = datas[-1]
            features = datas[: -1]
            logits =  model(*features)
            loss = val_loss_cls(logits, onset_labels)
            smoth_loss = val_loss_cls.step()
            p, r, f1 = val_metric_cls(logits, onset_labels)
            avg_p, avg_r, avg_f1 = val_metric_cls.step()

            data_config = {
                'loss': loss.detach(),
                'smoth_loss': smoth_loss,
                'p': p,
                'r': r,
                'f1': f1,
                'avg_p': avg_p,
                'avg_r': avg_r,
                'avg_f1': avg_f1
              }
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
          writer.add_scalar('valid/smoth_loss', data_config['smoth_loss'], total_step)
          writer.add_scalar('valid/avg_p', data_config['avg_p'], total_step)
          writer.add_scalar('valid/avg_r', data_config['avg_r'], total_step)
          writer.add_scalar('valid/avg_f1', data_config['avg_f1'], total_step)

      if total_step % save_log_interval == 0 and rank == 0 and total_step >= start_val_step:
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