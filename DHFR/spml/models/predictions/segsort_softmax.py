"""Define SegSort with Softmax Classifier for semantic segmentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import spml.models.utils as model_utils
import torchvision.transforms as transforms
import spml.utils.general.vis as vis_utils
import spml.utils.general.common as common_utils
import spml.utils.segsort.loss as segsort_loss
import spml.utils.segsort.eval as segsort_eval
import spml.utils.segsort.common as segsort_common
import spml.models.predictions.lscloss as LSCLoss
import spml.models.predictions.smooth_loss as Smooth


class SegsortSoftmax(nn.Module):

  def __init__(self, config):
    
    super(SegsortSoftmax, self).__init__()

    # Define regularization by semantic annotation.
    self.sem_ann_loss = self._construct_loss(
        config.train.sem_ann_loss_types,
        concentration=config.train.sem_ann_concentration)
    self.sem_ann_loss_weight = config.train.sem_ann_loss_weight

    # Define regularization by semantic cooccurrence.
    loss_type = (
      'set_segsort' if config.train.sem_occ_loss_types == 'segsort' else 'none')
    self.sem_occ_loss = self._construct_loss(
        loss_type,
        concentration=config.train.sem_occ_concentration)
    self.sem_occ_loss_weight = config.train.sem_occ_loss_weight

    # Define regularization by low-level image similarity.
    self.img_sim_loss = self._construct_loss(
        config.train.img_sim_loss_types,
        concentration=config.train.img_sim_concentration)
    self.img_sim_loss_weight = config.train.img_sim_loss_weight

    # Define regularization by feature affinity.
    loss_type = (
      'set_segsort' if config.train.feat_aff_loss_types == 'segsort' else 'none')
    self.feat_aff_loss = self._construct_loss(
        config.train.feat_aff_loss_types,
        concentration=config.train.feat_aff_concentration)
    self.feat_aff_loss_weight = config.train.feat_aff_loss_weight

    # Softmax classifier head.
    self.parted_ce_loss_weight = config.train.parted_ce_loss_weight
    if self.parted_ce_loss_weight > 0.0:
      self.pCE_loss = torch.nn.CrossEntropyLoss(weight=None, ignore_index=255, reduction='mean').cuda()
      self.contourCE_loss = torch.nn.BCELoss().cuda()
    else:
      self.pCE_loss = None
      self.contourCE_loss = None

    # l1 crf loss
    self.l1_crf_loss_weight = config.train.l1_crf_loss_weight
    self.feat_crf_loss_weight = config.train.feat_crf_loss_weight
    if (self.l1_crf_loss_weight > 0.0) or (self.feat_crf_loss_weight > 0.0):
      self.loss_lsc = LSCLoss.LocalSaliencyCoherence().cuda()
    else:
      self.loss_lsc = None

    # lsmo
    self.loss_smo = Smooth.smoothness_loss().cuda()

    # ssc loss
    self.ssc_loss_weight = config.train.ssc_loss_weight

    # need segsort loss
    self.use_segsort_flag = False
    if ((config.train.sem_ann_loss_types != 'none') or (config.train.sem_occ_loss_types != 'none') or (
        config.train.img_sim_loss_types != 'none') or (config.train.feat_aff_loss_types != 'none')):
      self.use_segsort_flag = True

    # other init
    self.img_size = config.train.crop_size[0]
    self.batch_size = config.train.batch_size
    self.img_mean = config.network.pixel_means
    self.img_std = config.network.pixel_stds
    self.transform_norm = transforms.Normalize(self.img_mean, self.img_std)
    self.semantic_ignore_index = config.dataset.semantic_ignore_index
    self.num_classes = config.dataset.num_classes
    self.label_divisor = config.network.label_divisor

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
    ssim = torch.mean(self.SSIM(x,y))
    l1_loss = torch.mean(torch.abs(x-y))
    loss_ssc = alpha*ssim + (1-alpha)*l1_loss
    return loss_ssc
    
  def _construct_loss(self, loss_types, **kwargs):

    if loss_types == 'segsort':
      return segsort_loss.SegSortLoss(kwargs['concentration'],
                                      group_mode='segsort+',
                                      reduction='mean')
    elif loss_types == 'set_segsort':
      return segsort_loss.SetSegSortLoss(kwargs['concentration'],
                                         group_mode='segsort+',
                                         reduction='mean')
    elif loss_types == 'none':
      return None
    else:
      raise KeyError('Unsupported loss types: {:s}'.format(loss_types))

  def predictions(self, datas, targets={}):
    """Predict semantic segmentation by Softmax Classifier.
    """

    # Predict semantic labels.
    semantic_pred = datas['div1']
    targets['semantic_pred'] = semantic_pred
    
    return semantic_pred

  def embedding_to_rgb(self, datas):
    embeddings = datas['cluster_embedding'].detach()
    emb_h = emb_w = self.img_size // 4
    embeddings = embeddings.view(self.batch_size, emb_h, emb_w, -1).permute(0, 3, 1, 2).contiguous()
    vis_rgb = vis_utils.embedding_to_rgb(embeddings.data.cpu(), 'pca').cuda() / 255
    vis_rgb = self.transform_norm(vis_rgb)
    return vis_rgb

  def losses(self, datas, targets={}):
    """Compute losses.
    """

    # get prediction saliency map
    semantic_pred = datas['div1']
    contour_pred = datas['contour_div1']
    targets['semantic_pred'] = semantic_pred
    targets['contour_pred'] = contour_pred

    # cal parted ce loss, ssc loss, l1 crf loss
    partedCE_loss = None
    ssc_loss = None
    l1_crf_loss = None
    contour_loss = None

    # Ssc loss
    if self.ssc_loss_weight > 0.0:
      ssc_loss = self.SaliencyStructureConsistency(datas['div1_aug'], semantic_pred, 0.85)
      ssc_loss *= self.ssc_loss_weight

    # compute layers pCE and L1 CRF loss
    images = targets['image']
    out_div1 = torch.cat((1-semantic_pred, semantic_pred), dim=1)
    out_div2 = torch.cat((1-datas['div2'], datas['div2']), dim=1)
    out_div4 = torch.cat((1-datas['div4'], datas['div4']), dim=1)
    out_div8 = torch.cat((1-datas['div8'], datas['div8']), dim=1)
    out_div16 = torch.cat((1-datas['div16'], datas['div16']), dim=1)
    bef_out_div16 = torch.cat((1-datas['bef_div16'], datas['bef_div16']), dim=1)

    # perpare fg bg label
    semantic_labels = targets.get('semantic_label', None)
    semantic_labels = semantic_labels.masked_fill(
      semantic_labels >= self.num_classes, self.semantic_ignore_index)

    # parted ce loss
    if self.pCE_loss is not None:
      semantic_labels_mask = semantic_labels.squeeze(1).long()
      bg_label = semantic_labels_mask.clone()
      fg_label = semantic_labels_mask.clone()
      bg_label[semantic_labels_mask != 0] = 255
      fg_label[semantic_labels_mask == 0] = 255
      partedCE_loss = (self.pCE_loss(out_div1, fg_label) + self.pCE_loss(out_div1, bg_label)) * 1
      partedCE_loss += (self.pCE_loss(out_div2, fg_label) + self.pCE_loss(out_div2, bg_label)) * 0.8
      partedCE_loss += (self.pCE_loss(out_div4, fg_label) + self.pCE_loss(out_div4, bg_label)) * 0.8
      partedCE_loss += (self.pCE_loss(out_div8, fg_label) + self.pCE_loss(out_div8, bg_label)) * 0.6
      partedCE_loss += (self.pCE_loss(out_div16, fg_label) + self.pCE_loss(out_div16, bg_label)) * 0.4
      partedCE_loss += (self.pCE_loss(bef_out_div16, fg_label) + self.pCE_loss(bef_out_div16, bg_label)) * 0.4
      partedCE_loss *= self.parted_ce_loss_weight

    if self.contourCE_loss is not None:
      contour_loss = self.contourCE_loss(contour_pred, targets.get('contour_sum_label', None).unsqueeze(dim=1))
    '''
    # l1 crf loss
    if self.loss_lsc is not None:
      # perpare embedding rgb
      loss_lsc_radius = 5
      rgb_weight = self.l1_crf_loss_weight
      feat_weight = self.feat_crf_loss_weight
      lsc_h = lsc_w = self.img_size // 4
      sample = {}
      loss_lsc_kernels_desc_defaults = {"weight": 1, "xy": 20}
      if rgb_weight > 0.0:
        sample['rgb'] = F.interpolate(images, scale_factor=0.25, mode='bilinear', align_corners=True)
        loss_lsc_kernels_desc_defaults['rgb'] = rgb_weight

      sample_feat = {}
      loss_feat_lsc_kernels_desc_defaults = {"weight": 1, "xy": 6}
      if rgb_weight > 0.0:
        sample_feat['rgb'] = F.interpolate(images, scale_factor=0.25, mode='bilinear', align_corners=True)
        loss_feat_lsc_kernels_desc_defaults['rgb'] = rgb_weight
      if feat_weight > 0.0:
        sample_feat['vis_rgb'] = self.embedding_to_rgb(datas)
        loss_feat_lsc_kernels_desc_defaults['vis_rgb'] = feat_weight

      # emb layer div1
      emb_out2_ = F.interpolate(out_contour_div1[:, 1:2], scale_factor=0.25, mode='bilinear', align_corners=True)
      loss2_feat_lsc = self.loss_lsc(emb_out2_, [loss_feat_lsc_kernels_desc_defaults], loss_lsc_radius, sample_feat, lsc_h, lsc_w)['loss']

      # # layer div1
      # out2_ = F.interpolate(out_div1[:, 1:2], scale_factor=0.25, mode='bilinear', align_corners=True)
      # loss2_lsc = self.loss_lsc(out2_, [loss_lsc_kernels_desc_defaults], loss_lsc_radius, sample, lsc_h, lsc_w)['loss']

      # # layer div4
      # out3_ = F.interpolate(out_div4[:, 1:2], scale_factor=0.25, mode='bilinear', align_corners=True)
      # for k in loss_lsc_kernels_desc_defaults.keys():
      #   if k not in ["weight", "xy"]:
      #     loss_lsc_kernels_desc_defaults[k] /= 10
      # loss3_lsc = self.loss_lsc(out3_, [loss_lsc_kernels_desc_defaults], loss_lsc_radius, sample, lsc_h, lsc_w)['loss']

      # # layer div8
      # out4_ = F.interpolate(out_div8[:, 1:2], scale_factor=0.25, mode='bilinear', align_corners=True)
      # for k in loss_lsc_kernels_desc_defaults.keys():
      #   if k not in ["weight", "xy"]:
      #     loss_lsc_kernels_desc_defaults[k] /= 10
      # loss4_lsc = self.loss_lsc(out4_, [loss_lsc_kernels_desc_defaults], loss_lsc_radius, sample, lsc_h, lsc_w)['loss']

      # # layer div16
      # out5_ = F.interpolate(out_div16[:, 1:2], scale_factor=0.25, mode='bilinear', align_corners=True)
      # for k in loss_lsc_kernels_desc_defaults.keys():
      #   if k not in ["weight", "xy"]:
      #     loss_lsc_kernels_desc_defaults[k] /= 10
      # loss5_lsc = self.loss_lsc(out5_, [loss_lsc_kernels_desc_defaults], loss_lsc_radius, sample, lsc_h, lsc_w)['loss']
      ##l1_crf_loss = loss2_feat_lsc * 1 + loss2_lsc * 1 + loss3_lsc * 0.8 + loss4_lsc * 0.6 + loss5_lsc * 0.4

      loss2_lsc = self.loss_smo(out_div1[:, 1:2], images)
      l1_crf_loss = loss2_feat_lsc * 1 + loss2_lsc * 1
      l1_crf_loss *= 0.3
    '''
    # lsmo
    smo_loss = self.loss_smo(out_div1[:, 1:2], images)
    smo_loss *= 0.3

    # cal segsort loss
    sem_ann_loss = None
    sem_occ_loss = None
    img_sim_loss = None
    sem_ann_acc = None
    semantic_labels = semantic_labels.squeeze_(1).long()
    # Compute semantic annotation and semantic co-occurrence loss.
    if self.sem_ann_loss is not None or self.sem_occ_loss is not None:
      cluster_indices = datas['cluster_index']
      embeddings = datas['cluster_embedding']
      cluster_indices_aug = datas['cluster_index_aug'] + cluster_indices.max() + 1
      embeddings_aug = datas['cluster_embedding_aug']
      embeddings = torch.cat((embeddings, embeddings_aug), dim=0)
      semantic_labels = torch.cat((datas['cluster_semantic_label'], datas['cluster_semantic_label']), dim=0)
      cluster_indices = torch.cat((cluster_indices, cluster_indices_aug), dim=0)

      prototypes = targets['prototype']
      prototype_semantic_labels = targets['prototype_semantic_label']

      prototypes_aug = targets['prototype_aug']
      prototype_semantic_labels_aug = targets['prototype_semantic_label_aug']

      prototypes = torch.cat((prototypes, prototypes_aug), dim=0)
      prototype_semantic_labels = torch.cat((prototype_semantic_labels, prototype_semantic_labels_aug), dim=0)

      # Add prototypes in the memory bank.
      memory_prototypes = targets.get(
          'memory_prototype', [])
      memory_prototype_semantic_labels = targets.get(
          'memory_prototype_semantic_label', [])
      memory_prototypes_aug = targets.get(
          'memory_prototype_aug', [])
      memory_prototype_semantic_labels_aug = targets.get(
          'memory_prototype_semantic_label_aug', [])
      if (memory_prototypes
          and memory_prototype_semantic_labels
          and memory_prototypes_aug
          and memory_prototype_semantic_labels_aug):
        memory_prototypes += memory_prototypes_aug
        memory_prototype_semantic_labels += memory_prototype_semantic_labels_aug
        memory_prototypes = torch.cat(memory_prototypes, dim=0)
        memory_prototype_semantic_labels = torch.cat(memory_prototype_semantic_labels, dim=0)
        # memory_prototype_semantic_labels += (self.num_classes + 1) # memory semantic only use to faraway prototype

        prototypes = torch.cat((prototypes, memory_prototypes), dim=0)
        prototype_semantic_labels = torch.cat((prototype_semantic_labels, memory_prototype_semantic_labels), dim=0)

        pixel_inds = (semantic_labels != self.num_classes).logical_and(semantic_labels != (self.num_classes*2+1)).nonzero().view(-1)
        proto_inds = (prototype_semantic_labels != self.num_classes).logical_and(prototype_semantic_labels != (self.num_classes*2+1)).nonzero().view(-1)
        c_inds = torch.arange(
            prototypes.shape[0], dtype=torch.long,
            device=prototypes.device)
        c_inds = c_inds.masked_fill(
            (prototype_semantic_labels == self.num_classes).logical_or(prototype_semantic_labels == (self.num_classes*2+1)),
            c_inds.max() + 1)
        _, c_inds = torch.unique(c_inds, return_inverse=True)
        new_cluster_indices = torch.gather(
            c_inds, 0, cluster_indices)
      else:
        pixel_inds = (semantic_labels < self.num_classes).nonzero().view(-1)
        proto_inds = (prototype_semantic_labels < self.num_classes).nonzero().view(-1)
        c_inds = torch.arange(
            prototypes.shape[0], dtype=torch.long,
            device=prototypes.device)
        c_inds = c_inds.masked_fill(
            prototype_semantic_labels >= self.num_classes,
            c_inds.max() + 1)
        _, c_inds = torch.unique(c_inds, return_inverse=True)
        new_cluster_indices = torch.gather(
            c_inds, 0, cluster_indices)

      sem_ann_loss = self.sem_ann_loss (
          torch.index_select(embeddings, 0, pixel_inds),
          torch.index_select(semantic_labels, 0, pixel_inds),
          torch.index_select(new_cluster_indices, 0, pixel_inds),
          torch.index_select(prototypes, 0, proto_inds),
          torch.index_select(prototype_semantic_labels, 0, proto_inds))
      sem_ann_loss *= self.sem_ann_loss_weight

      sem_ann_acc, _ = segsort_eval.top_k_ranking(
          prototypes,
          prototype_semantic_labels,
          prototypes,
          prototype_semantic_labels,
          5)

    # Compute low-level image similarity loss.
    if self.img_sim_loss is not None:
      cluster_indices = datas['cluster_index']
      embeddings = datas['cluster_embedding_with_loc']
      instance_labels = datas['cluster_instance_label']
      batch_indices = datas['cluster_batch_index']

      img_sim_loss = []
      for batch_ind in torch.unique(batch_indices):
        batch_mask = batch_indices == batch_ind
        inds = batch_mask.nonzero().view(-1)
        embs = torch.index_select(embeddings, 0, inds)
        labs = torch.index_select(instance_labels, 0, inds)
        c_inds = torch.index_select(cluster_indices, 0, inds)
        p_labs, c_inds = segsort_common.prepare_prototype_labels(
            labs, c_inds, labs.max() + 1)
        protos = (
          segsort_common.calculate_prototypes_from_labels(embs, c_inds))
        img_sim_loss.append(self.img_sim_loss(
            embs, labs, c_inds, protos, p_labs))
      img_sim_loss = sum(img_sim_loss) / len(img_sim_loss)
      img_sim_loss *= self.img_sim_loss_weight

    # Aug Compute low-level image similarity loss.
    if self.img_sim_loss is not None:
      cluster_indices = datas['cluster_index_aug']
      embeddings = datas['cluster_embedding_with_loc_aug']
      instance_labels = datas['cluster_instance_label']
      batch_indices = datas['cluster_batch_index']

      img_aug_sim_loss = []
      for batch_ind in torch.unique(batch_indices):
        batch_mask = batch_indices == batch_ind
        inds = batch_mask.nonzero().view(-1)
        embs = torch.index_select(embeddings, 0, inds)
        labs = torch.index_select(instance_labels, 0, inds)
        c_inds = torch.index_select(cluster_indices, 0, inds)
        p_labs, c_inds = segsort_common.prepare_prototype_labels(
            labs, c_inds, labs.max() + 1)
        protos = (
          segsort_common.calculate_prototypes_from_labels(embs, c_inds))
        img_aug_sim_loss.append(self.img_sim_loss(
            embs, labs, c_inds, protos, p_labs))
      img_aug_sim_loss = sum(img_aug_sim_loss) / len(img_aug_sim_loss)
      img_aug_sim_loss *= self.img_sim_loss_weight
      img_sim_loss += img_aug_sim_loss
    return sem_ann_loss, img_sim_loss, partedCE_loss, ssc_loss, smo_loss, contour_loss, sem_ann_acc

  def forward(self, datas, targets=None,
              with_loss=True, with_prediction=False):
    """Compute loss and predictions.
    """
    targets = targets if targets is not None else {}
    outputs = {}

    if with_prediction:
      # Predict semantic and instance labels.
      semantic_pred = self.predictions(datas, targets)

      outputs.update({'semantic_prediction': semantic_pred})

    if with_loss:
      sem_ann_loss, img_sim_loss, partedCE_loss, ssc_loss, smo_loss, contour_loss, sem_ann_acc = (
          self.losses(datas, targets))

      outputs.update(
          {'sem_ann_loss': sem_ann_loss,
           'img_sim_loss': img_sim_loss,
           'partedCE_loss': partedCE_loss,
           'ssc_loss': ssc_loss,
           'smo_loss': smo_loss,
           'contour_loss': contour_loss,
           'accuracy': sem_ann_acc})

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


def segsort(config):
  """Paramteric prototype predictor.
  """
  return SegsortSoftmax(config)
