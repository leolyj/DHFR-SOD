"""Define Softmax Classifier for semantic segmentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import spml.models.utils as model_utils


class SoftmaxClassifier(nn.Module):

  def __init__(self, config):
    super(SoftmaxClassifier, self).__init__()
    self.pCE_loss = torch.nn.CrossEntropyLoss(weight=None, ignore_index=255, reduction='mean').cuda()
    self.ignore_index = config.dataset.semantic_ignore_index
    self.num_classes = config.dataset.num_classes


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
    semantic_pred = datas['div1']
    contour_pred = datas['contour_div1']

    # Compute semantic loss.
    partedCE_loss, semantic_acc = None, None
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

      semantic_acc = torch.eq((semantic_pred > 0.5), semantic_labels)
      valid_pixels = torch.ne(semantic_labels,
                              self.ignore_index)
      semantic_acc = torch.masked_select(semantic_acc, valid_pixels).float().mean()

    outputs = {'semantic_pred': semantic_pred,
               'contour_pred': contour_pred,
               'partedCE_loss': partedCE_loss,
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


def softmax_classifier(config):
  """Pixel semantic segmentation model.
  """
  return SoftmaxClassifier(config)
