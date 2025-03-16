"""Define Softmax Classifier for semantic segmentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import spml.models.predictions.refine_fpn as refine_fpn
import spml.models.predictions.lscloss as LSCLoss


class SaliencyClassifier(nn.Module):

  def __init__(self, config):
    super(SaliencyClassifier, self).__init__()

    # network
    self.semantic_classifier = refine_fpn.RefineFPN()

    # pce loss
    self.pCE_loss = torch.nn.CrossEntropyLoss(weight=None, ignore_index=255, reduction='mean').cuda()

    # l1 crf loss
    self.l1_crf_loss_weight = config.train.l1_crf_loss_weight
    self.feat_crf_loss_weight = config.train.feat_crf_loss_weight
    if (self.l1_crf_loss_weight > 0.0) or (self.feat_crf_loss_weight > 0.0):
      self.loss_lsc = LSCLoss.LocalSaliencyCoherence().cuda()
    else:
      self.loss_lsc = None

    # ssc loss
    self.ssc_loss_weight = config.train.ssc_loss_weight

    # other init
    self.ignore_index = config.dataset.semantic_ignore_index
    self.num_classes = config.dataset.num_classes
    self.img_size = config.train.crop_size[0]

  def SSIM(self, x, y):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_x = nn.AvgPool2d(3, 1, 1)(x)
    mu_y = nn.AvgPool2d(3, 1, 1)(y)
    mu_x_mu_y = mu_x * mu_y
    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)

    sigma_x = nn.AvgPool2d(3, 1, 1)(x * x) - mu_x_sq
    sigma_y = nn.AvgPool2d(3, 1, 1)(y * y) - mu_y_sq
    sigma_xy = nn.AvgPool2d(3, 1, 1)(x * y) - mu_x_mu_y

    SSIM_n = (2 * mu_x_mu_y + C1) * (2 * sigma_xy + C2)
    SSIM_d = (mu_x_sq + mu_y_sq + C1) * (sigma_x + sigma_y + C2)
    SSIM = SSIM_n / SSIM_d

    return torch.clamp((1 - SSIM) / 2, 0, 1)

  def SaliencyStructureConsistency(self, x, y, alpha):
    ssim = torch.mean(self.SSIM(x, y))
    l1_loss = torch.mean(torch.abs(x - y))
    loss_ssc = alpha * ssim + (1 - alpha) * l1_loss
    return loss_ssc

  def forward(self, datas, targets=None):
    """Predict semantic segmenation and loss.

    Args:
      datas: A dict with an entry `embedding`, which is a 4-D float
        tensor of shape `[batch_size, num_channels, height, width]`.
      targets: A dict with an entry `semantic_label`, which is a 3-D
        long tensor of shape `[batch_size, height, width]`.

    Return:
      A dict of tensors and scalars.
    """
    targets = targets if targets is not None else {}

    # Predict semantic labels.
    semantic_pred = self.semantic_classifier(datas['div1'])

    # Compute semantic loss.
    partedCE_loss, l1_crf_loss, ssc_loss, semantic_acc = None, None, None, None
    semantic_labels = targets.get('semantic_label', None)
    if semantic_labels is not None:
      semantic_labels = semantic_labels.masked_fill(
          semantic_labels >= self.num_classes, self.ignore_index)
      semantic_labels = semantic_labels.squeeze_(1).long()

      semantic_pos_neg = torch.cat((1-semantic_pred, semantic_pred), dim=1);
      semantic_labels_mask = semantic_labels.squeeze(1).long()
      bg_label = semantic_labels_mask.clone()
      fg_label = semantic_labels_mask.clone()
      bg_label[semantic_labels_mask != 0] = 255
      fg_label[semantic_labels_mask == 0] = 255
      partedCE_loss = self.pCE_loss(semantic_pos_neg, bg_label) + self.pCE_loss(semantic_pos_neg, fg_label)

      # Ssc loss
      if self.ssc_loss_weight > 0.0:
        ssc_loss = self.SaliencyStructureConsistency(datas['div1_r'], datas['div1'], 0.85)
        ssc_loss *= self.ssc_loss_weight

      # l1 crf loss
      if self.loss_lsc is not None:
        # perpare embedding rgb
        loss_lsc_radius = 5
        rgb_weight = self.l1_crf_loss_weight
        feat_weight = self.feat_crf_loss_weight
        lsc_h = lsc_w = self.img_size // 4
        sample = {}
        loss_lsc_kernels_desc_defaults = {"weight": 1, "xy": 6}
        if rgb_weight > 0.0:
          sample['rgb'] = F.interpolate(targets['image'], scale_factor=0.25, mode='bilinear', align_corners=True)
          loss_lsc_kernels_desc_defaults['rgb'] = rgb_weight
        if feat_weight > 0.0:
          sample['vis_rgb'] = self.embedding_to_rgb(datas)
          loss_lsc_kernels_desc_defaults['vis_rgb'] = feat_weight

        # layer div1
        out2_ = F.interpolate(semantic_pred, scale_factor=0.25, mode='bilinear', align_corners=True)
        l1_crf_loss = self.loss_lsc(out2_, [loss_lsc_kernels_desc_defaults], loss_lsc_radius, sample, lsc_h, lsc_w)['loss']
        l1_crf_loss *= 0.3

      semantic_acc = torch.eq((semantic_pred > 0.5), semantic_labels)
      valid_pixels = torch.ne(semantic_labels,
                              self.ignore_index)
      semantic_acc = torch.masked_select(semantic_acc, valid_pixels).float().mean()

    outputs = {'semantic_pred': semantic_pred,
               'partedCE_loss': partedCE_loss,
               'l1_crf_loss': l1_crf_loss,
               'ssc_loss': ssc_loss,
               'accuracy': semantic_acc,}

    return outputs

  def get_params_lr(self):
    """Helper function to adjust learning rate for each sub modules.
    """
    # Specify learning rate for each sub modules.
    ret = []
    ret.append({
      'params': [n for n in model_utils.get_params(
          self,
          ['semantic_classifier'],
          ['weight'])],
      'lr': 10})
    ret.append({
      'params': [n for n in model_utils.get_params(
          self,
          ['semantic_classifier'],
          ['bias'])],
      'lr': 20,
      'weight_decay': 0})

    return ret


def saliency_classifier(config):
  """Pixel semantic segmentation model.
  """
  return SaliencyClassifier(config)
