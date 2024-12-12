import torch.nn as nn
import torch

class FrameMetrics(nn.Module):
  def __init__(self, prob_threshould):
    super(FrameMetrics, self).__init__()
    self.pro_threshould = prob_threshould
    self.TP = 0.
    self.FP = 0.
    self.FN = 0.
    self.eps = 1e-7

  def forward(self, logits, labels):
    pred = logits.detach().sigmoid() > self.pro_threshould
    labels = labels.detach()
    tp = (pred * labels).sum()
    fp = (pred * (1 - labels)).sum()
    fn = (~pred * labels).sum()

    self.TP += tp
    self.FP += fp
    self.FN += fn

    p = tp / (tp + fp + self.eps) * 100
    r = tp / (tp + fn + self.eps) * 100
    f1 = 2 * p * r / (p + r + self.eps)

    return p, r, f1
  
  def step(self):

    p = self.TP / (self.TP + self.FP + self.eps) * 100
    r = self.TP / (self.TP + self.FN + self.eps) * 100
    f1 = 2 * p * r / (p + r + self.eps)

    return p, r, f1

  def zero(self):
    
    self.TP = 0.
    self.FP = 0.
    self.FN = 0.


class LossContainer(nn.Module):
  def __init__(self, loss_func, pos_weight=1.0, reduction='mean', beta=0.3):
    super(LossContainer, self).__init__()
    if loss_func == 'cross_entropy':
      # logits targets
      self.loss_func = nn.BCEWithLogitsLoss(reduction='none')
    else:
      raise ValueError(loss_func, 'not supported')
    self.reduction = reduction
    self.loss = 0.
    self.beta = beta
    self.pos_weight = pos_weight

  def _apply_reduction(self, value):
    if self.reduction == 'mean':
      return value.mean()
    if self.reduction == 'sum':
      return value.sum()
    raise NotImplementedError(self.reduction, 'is not implemented!')

  def forward(self, logits, labels):
    loss = self.loss_func(logits, labels)
    device = loss.device
    mask = torch.where(labels > 0.0, torch.tensor(self.pos_weight, device=device), 
                        torch.tensor(1.0, device=device)).to(device)
    loss = loss * mask
    loss = self._apply_reduction(loss)
    self.loss = self.beta * self.loss + (1 - self.beta) * loss.detach()
    return loss
  
  def step(self):
    return self.loss
  
  def zero(self):
    self.loss = 0.


class FrameConformerMetrics(nn.Module):
  def __init__(self, prob_threshould):
    super(FrameConformerMetrics, self).__init__()
    self.pro_threshould = prob_threshould
    self.conv_TP = 0.
    self.conv_FP = 0.
    self.conv_FN = 0.
    self.trans_TP = 0.
    self.trans_FP = 0.
    self.trans_FN = 0.
    self.TP = 0.
    self.FP = 0.
    self.FN = 0.
    self.eps = 1e-7

  def forward(self, conv_logits, trans_logits, labels):
    labels = labels.detach()
    conv_pred = conv_logits.detach().sigmoid() > self.pro_threshould
    conv_tp = (conv_pred * labels).sum()
    conv_fp = (conv_pred * (1 - labels)).sum()
    conv_fn = (~conv_pred * labels).sum()

    self.conv_TP += conv_tp
    self.conv_FP += conv_fp
    self.conv_FN += conv_fn

    conv_p = conv_tp / (conv_tp + conv_fp + self.eps) * 100
    conv_r = conv_tp / (conv_tp + conv_fn + self.eps) * 100
    conv_f1 = 2 * conv_p * conv_r / (conv_p + conv_r + self.eps)

    trans_pred = trans_logits.detach().sigmoid() > self.pro_threshould
    trans_tp = (trans_pred * labels).sum()
    trans_fp = (trans_pred * (1 - labels)).sum()
    trans_fn = (~trans_pred * labels).sum()

    self.trans_TP += trans_tp
    self.trans_FP += trans_fp
    self.trans_FN += trans_fn

    trans_p = trans_tp / (trans_tp + trans_fp + self.eps) * 100
    trans_r = trans_tp / (trans_tp + trans_fn + self.eps) * 100
    trans_f1 = 2 * trans_p * trans_r / (trans_p + trans_r + self.eps)

    pred = ( conv_logits.detach().sigmoid() + trans_logits.detach().sigmoid()) / 2. > self.pro_threshould
    tp = (pred * labels).sum()
    fp = (pred * (1 - labels)).sum()
    fn = (~pred * labels).sum()

    self.TP += tp
    self.FP += fp
    self.FN += fn

    p = tp / (tp + fp + self.eps) * 100
    r = tp / (tp + fn + self.eps) * 100
    f1 = 2 * p * r / (p + r + self.eps)

    return p, r, f1, conv_p, conv_r, conv_f1, trans_p, trans_r, trans_f1
  
  def step(self):
    conv_p = self.conv_TP / (self.conv_TP + self.conv_FP + self.eps) * 100
    conv_r = self.conv_TP / (self.conv_TP + self.conv_FN + self.eps) * 100
    conv_f1 = 2 * conv_p * conv_r / (conv_p + conv_r + self.eps)

    trans_p = self.trans_TP / (self.trans_TP + self.trans_FP + self.eps) * 100
    trans_r = self.trans_TP / (self.trans_TP + self.trans_FN + self.eps) * 100
    trans_f1 = 2 * trans_p * trans_r / (trans_p + trans_r + self.eps)

    p = self.TP / (self.TP + self.FP + self.eps) * 100
    r = self.TP / (self.TP + self.FN + self.eps) * 100
    f1 = 2 * p * r / (p + r + self.eps)

    return p, r, f1, conv_p, conv_r, conv_f1, trans_p, trans_r, trans_f1

  def zero(self):
    
    self.TP = 0.
    self.FP = 0.
    self.FN = 0.
    self.conv_TP = 0.
    self.conv_FP = 0.
    self.conv_FN = 0.
    self.trans_TP = 0.
    self.trans_FP = 0.
    self.trans_FN = 0.

class LossConformer(nn.Module):
  def __init__(self, loss_func, pos_weight=1.0, reduction='mean', beta=0.3):
    super(LossConformer, self).__init__()
    if loss_func == 'cross_entropy':
      # logits targets
      self.conv_loss_func = nn.BCEWithLogitsLoss(reduction='none')
      self.trans_loss_func = nn.BCEWithLogitsLoss(reduction='none')
    else:
      raise ValueError(loss_func, 'not supported')
    self.reduction = reduction
    self.conv_loss = 0.
    self.trans_loss = 0.
    self.loss = 0.
    self.beta = beta
    self.pos_weight = pos_weight

  def _apply_reduction(self, value):
    if self.reduction == 'mean':
      return value.mean()
    if self.reduction == 'sum':
      return value.sum()
    raise NotImplementedError(self.reduction, 'is not implemented!')

  def forward(self, conv_logits, trans_logits, labels):
    device = labels.device
    mask = torch.where(labels > 0.0, torch.tensor(self.pos_weight, device=device), 
                        torch.tensor(1.0, device=device)).to(device)
    
    conv_loss = self.conv_loss_func(conv_logits, labels)
    conv_loss = conv_loss * mask
    conv_loss = self._apply_reduction(conv_loss)
    self.conv_loss = self.beta * self.conv_loss + (1 - self.beta) * conv_loss.detach()

    trans_loss = self.trans_loss_func(trans_logits, labels)
    trans_loss = trans_loss * mask
    trans_loss = self._apply_reduction(trans_loss)
    self.trans_loss = self.beta * self.trans_loss + (1 - self.beta) * trans_loss.detach()

    loss = conv_loss + trans_loss
    self.loss = self.beta * self.loss + (1 - self.beta) * loss.detach()

    return loss, conv_loss, trans_loss
  
  def step(self):
    return self.loss, self.conv_loss, self.trans_loss
  
  def zero(self):
    self.loss = 0.
    self.conv_loss = 0.
    self.trans_loss = 0.

class ThreeFrameMetrics(nn.Module):
  def __init__(self, prob_threshould):
    super(ThreeFrameMetrics, self).__init__()
    self.pro_threshould = prob_threshould
    self.onset_TP, self.onset_FP, self.onset_FN = 0., 0., 0.
    self.offset_TP, self.offset_FP, self.offset_FN = 0., 0., 0.
    self.frame_TP, self.frame_FP, self.frame_FN = 0., 0., 0.
    self.eps = 1e-7

  def _compute_prf(self, tp, fp, fn):
    p = tp / (tp + fp + self.eps) * 100
    r = tp / (tp + fn + self.eps) * 100
    f1 = 2 * p * r / (p + r + self.eps)
    
    return p, r, f1

  def compute_prf(self, logits, labels, cls_type):
    pred = logits.detach().sigmoid() > self.pro_threshould
    labels = labels.detach()
    tp = (pred * labels).sum()
    fp = (pred * (1 - labels)).sum()
    fn = (~pred * labels).sum()

    if cls_type == 'onset':
      self.onset_TP += tp
      self.onset_FP += fp
      self.onset_FN += fn
    elif cls_type == 'offset':
      self.offset_TP += tp
      self.offset_FP += fp
      self.offset_FN += fn
    elif cls_type == 'frame':
      self.frame_TP += tp
      self.frame_FP += fp
      self.frame_FN += fn
    else:
      raise TypeError('not valid type:', cls_type)

    p, r, f1 = self._compute_prf(tp, fp, fn)

    return p, r, f1


  def forward(self, onset_logits, offset_logits, frame_logits,
                onset_labels, offset_labels, frame_labels):

    onset_p, onset_r, onset_f1 = self.compute_prf(onset_logits, onset_labels, 'onset')
    offset_p, offset_r, offset_f1 = self.compute_prf(offset_logits, offset_labels, 'offset')
    frame_p, frame_r, frame_f1 = self.compute_prf(frame_logits, frame_labels, 'frame')
    return (onset_p, onset_r, onset_f1, offset_p, offset_r, offset_f1, 
            frame_p, frame_r, frame_f1)
  
  def step(self):
    onset_p, onset_r, onset_f1 = self._compute_prf(self.onset_TP, 
                                    self.onset_FP, self.onset_FN)
    offset_p, offset_r, offset_f1 = self._compute_prf(self.offset_TP, 
                                    self.offset_FP, self.offset_FN)
    frame_p, frame_r, frame_f1 = self._compute_prf(self.frame_TP, 
                                    self.frame_FP, self.frame_FN)

    return (onset_p, onset_r, onset_f1, offset_p, offset_r, offset_f1,
            frame_p, frame_r, frame_f1)

  def zero(self):
    
    self.onset_TP, self.onset_FP, self.onset_FN = 0., 0., 0.
    self.offset_TP, self.offset_FP, self.offset_FN = 0., 0., 0.
    self.frame_TP, self.frame_FP, self.frame_FN = 0., 0., 0.


class TwoFrameMetrics(nn.Module):
  def __init__(self, prob_threshould):
    super(TwoFrameMetrics, self).__init__()
    self.pro_threshould = prob_threshould
    self.onset_TP, self.onset_FP, self.onset_FN = 0., 0., 0.
    self.frame_TP, self.frame_FP, self.frame_FN = 0., 0., 0.
    self.eps = 1e-7

  def _compute_prf(self, tp, fp, fn):
    p = tp / (tp + fp + self.eps) * 100
    r = tp / (tp + fn + self.eps) * 100
    f1 = 2 * p * r / (p + r + self.eps)
    
    return p, r, f1

  def compute_prf(self, logits, labels, cls_type):
    pred = logits.detach().sigmoid() > self.pro_threshould
    labels = labels.detach()
    tp = (pred * labels).sum()
    fp = (pred * (1 - labels)).sum()
    fn = (~pred * labels).sum()

    if cls_type == 'onset':
      self.onset_TP += tp
      self.onset_FP += fp
      self.onset_FN += fn
    elif cls_type == 'frame':
      self.frame_TP += tp
      self.frame_FP += fp
      self.frame_FN += fn
    else:
      raise TypeError('not valid type:', cls_type)

    p, r, f1 = self._compute_prf(tp, fp, fn)

    return p, r, f1

  def forward(self, onset_logits, frame_logits, onset_labels, frame_labels):

    onset_p, onset_r, onset_f1 = self.compute_prf(onset_logits, onset_labels, 'onset')
    frame_p, frame_r, frame_f1 = self.compute_prf(frame_logits, frame_labels, 'frame')
    return (onset_p, onset_r, onset_f1, frame_p, frame_r, frame_f1)
  
  def step(self):
    onset_p, onset_r, onset_f1 = self._compute_prf(self.onset_TP, 
                                    self.onset_FP, self.onset_FN)
    frame_p, frame_r, frame_f1 = self._compute_prf(self.frame_TP, 
                                    self.frame_FP, self.frame_FN)

    return (onset_p, onset_r, onset_f1, frame_p, frame_r, frame_f1)

  def zero(self):
    
    self.onset_TP, self.onset_FP, self.onset_FN = 0., 0., 0.
    self.frame_TP, self.frame_FP, self.frame_FN = 0., 0., 0.


class ThreeLossContainer(nn.Module):
  def __init__(self, loss_func, pos_weights=[1.0, 1.0, 1.0], reduction='mean', beta=0.3):
    super(ThreeLossContainer, self).__init__()
    if loss_func == 'cross_entropy':
      # logits targets
      self.onset_loss_func = nn.BCEWithLogitsLoss(reduction='none')
      self.offset_loss_func = nn.BCEWithLogitsLoss(reduction='none')
      self.frame_loss_func = nn.BCEWithLogitsLoss(reduction='none')
    else:
      raise ValueError(loss_func, 'not supported')
    self.reduction = reduction
    self.onset_loss = 0.
    self.offset_loss = 0.
    self.frame_loss = 0.
    self.loss = 0.
    self.beta = beta
    self.pos_weights = pos_weights

  def _apply_reduction(self, value):
    if self.reduction == 'mean':
      return value.mean()
    if self.reduction == 'sum':
      return value.sum()
    raise NotImplementedError(self.reduction, 'is not implemented!')

  def compute_loss(self, logits, labels, cls_type):
    if cls_type == 'onset':
      loss_func = self.onset_loss_func
      pos_weight = self.pos_weights[0]
    elif cls_type == 'offset':
      loss_func = self.offset_loss_func
      pos_weight = self.pos_weights[1]
    elif cls_type == 'frame':
      loss_func = self.frame_loss_func
      pos_weight = self.pos_weights[2]
    else:
      raise TypeError('not valid type:', cls_type)
    
    loss = loss_func(logits, labels)
    device = loss.device
    mask = torch.where(labels > 0.0, torch.tensor(pos_weight, device=device), 
                        torch.tensor(1.0, device=device)).to(device)
    loss = loss * mask
    loss = self._apply_reduction(loss)

    if cls_type == 'onset':
      self.onset_loss = self.beta * self.onset_loss + (1 - self.beta) * loss.detach()
    elif cls_type == 'offset':
      self.offset_loss = self.beta * self.offset_loss + (1 - self.beta) * loss.detach()
    else:
      self.frame_loss = self.beta * self.frame_loss + (1 - self.beta) * loss.detach()

    return loss    

  def forward(self, onset_logits, offset_logits, frame_logits, 
                onset_labels, offset_labels, frame_labels):
    onset_loss = self.compute_loss(onset_logits, onset_labels, 'onset')
    offset_loss = self.compute_loss(offset_logits, offset_labels, 'offset')
    frame_loss = self.compute_loss(frame_logits, frame_labels, 'frame')

    loss = onset_loss + offset_loss + frame_loss
    self.loss = self.beta * self.loss + (1 - self.beta) * loss.detach()

    return loss, onset_loss, offset_loss, frame_loss
  
  def step(self):
    return self.loss, self.onset_loss, self.offset_loss, self.frame_loss
  
  def zero(self):
    self.onset_loss = 0.
    self.offset_loss = 0.
    self.frame_loss = 0.
    self.loss = 0.


class FourLossContainer(nn.Module):
  def __init__(self, loss_func, pos_weights=[1.0, 1.0, 1.0], reduction='mean', beta=0.3):
    super(FourLossContainer, self).__init__()
    if loss_func == 'cross_entropy':
      # logits targets
      self.onset_loss_func = nn.BCEWithLogitsLoss(reduction='none')
      self.offset_loss_func = nn.BCEWithLogitsLoss(reduction='none')
      self.frame_loss_func = nn.BCEWithLogitsLoss(reduction='none')
    else:
      raise ValueError(loss_func, 'not supported')
    self.reduction = reduction
    self.onset_loss = 0.
    self.offset_loss = 0.
    self.frame_loss = 0.
    self.velocity_loss = 0.
    self.loss = 0.
    self.beta = beta
    self.pos_weights = pos_weights

  def _apply_reduction(self, value):
    if self.reduction == 'mean':
      return value.mean()
    if self.reduction == 'sum':
      return value.sum()
    raise NotImplementedError(self.reduction, 'is not implemented!')

  def compute_loss(self, logits, labels, cls_type, onset_labels=None):
    if cls_type == 'onset':
      loss_func = self.onset_loss_func
      pos_weight = self.pos_weights[0]
    elif cls_type == 'offset':
      loss_func = self.offset_loss_func
      pos_weight = self.pos_weights[1]
    elif cls_type == 'frame':
      loss_func = self.frame_loss_func
      pos_weight = self.pos_weights[2]
    elif cls_type == 'velocity':
      loss_func = lambda logits, labels: (logits.sigmoid() - labels) ** 2
      pos_weight = self.pos_weights[3]
    else:
      raise TypeError('not valid type:', cls_type)
    
    loss = loss_func(logits, labels)
    device = loss.device

    if cls_type == 'velocity':
      mask = torch.where(onset_labels > 0.0, torch.tensor(pos_weight, device=device), 
                          torch.tensor(0.0, device=device)).to(device)
      loss = (loss * mask).sum() / (onset_labels.sum() + 1e-7)
    else:
      mask = torch.where(labels > 0.0, torch.tensor(pos_weight, device=device), 
                          torch.tensor(1.0, device=device)).to(device)
      loss = loss * mask
      loss = self._apply_reduction(loss)      

    if cls_type == 'onset':
      self.onset_loss = self.beta * self.onset_loss + (1 - self.beta) * loss.detach()
    elif cls_type == 'offset':
      self.offset_loss = self.beta * self.offset_loss + (1 - self.beta) * loss.detach()
    elif cls_type == 'frame':
      self.frame_loss = self.beta * self.frame_loss + (1 - self.beta) * loss.detach()
    else:
      self.velocity_loss = self.beta * self.velocity_loss + (1 - self.beta) * loss.detach()

    return loss    

  def forward(self, onset_logits, offset_logits, frame_logits, velocity_logits,
                onset_labels, offset_labels, frame_labels, velocity_labels):
    onset_loss = self.compute_loss(onset_logits, onset_labels, 'onset')
    offset_loss = self.compute_loss(offset_logits, offset_labels, 'offset')
    frame_loss = self.compute_loss(frame_logits, frame_labels, 'frame')
    velocity_loss = self.compute_loss(velocity_logits, velocity_labels, 'velocity', onset_labels)

    loss = onset_loss + offset_loss + frame_loss + velocity_loss
    self.loss = self.beta * self.loss + (1 - self.beta) * loss.detach()

    return loss, onset_loss, offset_loss, frame_loss, velocity_loss
  
  def step(self):
    return self.loss, self.onset_loss, self.offset_loss, self.frame_loss, self.velocity_loss
  
  def zero(self):
    self.onset_loss = 0.
    self.offset_loss = 0.
    self.frame_loss = 0.
    self.velocity_loss = 0.
    self.loss = 0.


class TwoLossContainer(nn.Module):
  def __init__(self, loss_func, pos_weights=[1.0, 1.0, 1.0], reduction='mean', beta=0.3):
    super(TwoLossContainer, self).__init__()
    if loss_func == 'cross_entropy':
      # logits targets
      self.onset_loss_func = nn.BCEWithLogitsLoss(reduction='none')
      self.frame_loss_func = nn.BCEWithLogitsLoss(reduction='none')
    else:
      raise ValueError(loss_func, 'not supported')
    self.reduction = reduction
    self.onset_loss = 0.
    self.offset_loss = 0.
    self.frame_loss = 0.
    self.loss = 0.
    self.beta = beta
    self.pos_weights = pos_weights

  def _apply_reduction(self, value):
    if self.reduction == 'mean':
      return value.mean()
    if self.reduction == 'sum':
      return value.sum()
    raise NotImplementedError(self.reduction, 'is not implemented!')

  def compute_loss(self, logits, labels, cls_type):
    if cls_type == 'onset':
      loss_func = self.onset_loss_func
      pos_weight = self.pos_weights[0]
    elif cls_type == 'frame':
      loss_func = self.frame_loss_func
      pos_weight = self.pos_weights[2]
    else:
      raise TypeError('not valid type:', cls_type)
    
    loss = loss_func(logits, labels)
    device = loss.device
    mask = torch.where(labels > 0.0, torch.tensor(pos_weight, device=device), 
                        torch.tensor(1.0, device=device)).to(device)
    loss = loss * mask
    loss = self._apply_reduction(loss)

    if cls_type == 'onset':
      self.onset_loss = self.beta * self.onset_loss + (1 - self.beta) * loss.detach()
    else:
      self.frame_loss = self.beta * self.frame_loss + (1 - self.beta) * loss.detach()

    return loss    

  def forward(self, onset_logits, frame_logits, 
                onset_labels, frame_labels):
    onset_loss = self.compute_loss(onset_logits, onset_labels, 'onset')
    frame_loss = self.compute_loss(frame_logits, frame_labels, 'frame')

    loss = onset_loss + frame_loss
    self.loss = self.beta * self.loss + (1 - self.beta) * loss.detach()

    return loss, onset_loss, frame_loss
  
  def step(self):
    return self.loss, self.onset_loss, self.frame_loss
  
  def zero(self):
    self.onset_loss = 0.
    self.frame_loss = 0.
    self.loss = 0.