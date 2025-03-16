"""Build segmentation model with PSPNet."""

import torch
import torch.nn as nn
import torch.nn.functional as F

# debug
'''
import sys
sys.path.append("/home/liuzhy/SaliencyTrain/SPML")
from spml.config.default import config
'''

import spml.models.utils as model_utils
import spml.utils.general.common as common_utils
import spml.utils.segsort.common as segsort_common
from spml.models.embeddings.swin_fuse_net import SFNet
from spml.models.embeddings.local_model import LocationColorNetwork


class LocalVIT_SwinFuseNet(nn.Module):
    
  def __init__(self, config):
    """Build PSPNet using ResNet as backbone network.

    Args:
      backbone_depth: A list of integers indicate the number
        of residual layers in each block.
      strides: A list of intergers indicate the stride.
      dilations: A list of integers indicate the dilations.
      config: An easydict of the network configurations.
    """

    super(LocalVIT_SwinFuseNet, self).__init__()

    # Build Backbone Network.
    self.network_type = config.network.backbone_types
    if self.network_type == 'rgbd_sod_sfnet':
      self.sfnet = SFNet(need_depth_estimate=False)
    else:
      self.sfnet = SFNet(need_depth_estimate=True)

    # Build Local Feature Network.
    self.lfn = LocationColorNetwork(use_color=False, use_location=True,
                                    norm_color=False, smooth_ksize=None)

    # Parameters for VMF clustering.
    self.label_divisor = config.network.label_divisor
    self.num_classes = config.dataset.num_classes

    self.semantic_ignore_index = config.dataset.semantic_ignore_index

    self.kmeans_num_clusters = config.network.kmeans_num_clusters
    self.kmeans_iterations = config.network.kmeans_iterations

    # need segsort loss
    self.use_segsort_flag = False
    if ((config.train.sem_ann_loss_types != 'none') or (config.train.img_sim_loss_types != 'none')):
      self.use_segsort_flag = True

  def generate_embeddings(self, datas, targets=None, resize_as_input=False):
    """Feed-forward segmentation model to generate pixel-wise embeddings
    and location & RGB features.

    Args:
      datas: A dict with an entry `image`, which is a 4-D float tensor
        of shape `[batch_size, channels, height, width]`.
      targets: A dict with an entry `semantic_label` and `instance_label`,
        which are 3-D long tensors of shape `[batch_size, height, width]`.
      resize_as_input: enable/disable resize_as_input to upscale the 
        embeddings to the same size as the input image.

    Return:
      A dict with entry `embedding` and `local_feature` of shape
      `[batch_size, channels, height, width]`.
    """

    # Generate embeddings.
    div1_aug = None
    embeddings_aug = None
    if self.network_type == 'rgbd_sod_sfnet':
      before_fuse_div16, div16, div8, div4, div2, div1, embeddings, contour_div1 = self.sfnet(datas['image'], datas['depth'])
      if self.training:
        # if self.training:
        #   image_scale = F.interpolate(datas['image'], scale_factor=0.5, mode='bilinear', align_corners=True)
        #   depth_scale = F.interpolate(datas['depth'], scale_factor=0.5, mode='bilinear', align_corners=True)
        #   image_restore = F.interpolate(image_scale, scale_factor=2, mode='bilinear', align_corners=True)
        #   depth_restore = F.interpolate(depth_scale, scale_factor=2, mode='bilinear', align_corners=True)
        #   div1_r, _, _, _, _ = self.sfnet(image_restore, depth_restore)
        # else:
        #   div1_r = None
        image_scale = F.interpolate(datas['image'], scale_factor=0.5, mode='bilinear', align_corners=True)
        depth_scale = F.interpolate(datas['depth'], scale_factor=0.5, mode='bilinear', align_corners=True)
        image_restore = F.interpolate(image_scale, scale_factor=2, mode='bilinear', align_corners=True)
        depth_restore = F.interpolate(depth_scale, scale_factor=2, mode='bilinear', align_corners=True)
        _, _, _, _, _, div1_aug, embeddings_aug, _ = self.sfnet(image_restore, depth_restore)
      else:
        div1_aug = None
    else:
      div1, div4, div8, div16, embeddings = self.sfnet(datas['image'])
      if self.use_ssc_flag:
        image_scale = F.interpolate(datas['image'], scale_factor=0.5, mode='bilinear', align_corners=True)
        image_restore = F.interpolate(image_scale, scale_factor=2, mode='bilinear', align_corners=True)
        div1_r, _, _, _, _ = self.sfnet(image_restore)
      else:
        div1_r = None

    if resize_as_input:
      input_size = datas['image'].shape[-2:]
      embeddings = F.interpolate(
          embeddings, size=input_size, mode='bilinear')

    size = embeddings.shape[-2:]
    local_features = self.lfn(datas['image'], size=size)

    return {'embedding': embeddings, 'embedding_aug': embeddings_aug, 'div1': div1, 'div2': div2, 'div4': div4, 'div8': div8, 'div16': div16, 'bef_div16': before_fuse_div16, 'contour_div1': contour_div1, 'div1_aug': div1_aug, 'local_feature': local_features}

  def generate_clusters(self, embeddings,
                        embeddings_aug,
                        semantic_labels,
                        instance_labels,
                        local_features=None):
    """Perform Spherical KMeans clustering within each image.

    Args:
      embeddings: A a 4-D float tensor of shape
        `[batch_size, channels, height, width]`.
      semantic_labels: A 3-D long tensor of shape
        `[batch_size, height, width]`.
      instance_labels: A 3-D long tensor of shape
        `[batch_size, height, width]`.
      local_features: A 4-D float tensor of shape
        `[batch_size, height, width, channels]`.

    Return:
      A dict with entry `cluster_embedding`, `cluster_embedding_with_loc`,
      `cluster_semantic_label`, `cluster_instance_label`, `cluster_index`
      and `cluster_batch_index`.
    """

    if semantic_labels is not None and instance_labels is not None:
      labels = semantic_labels * self.label_divisor + instance_labels
      ignore_index = labels.max() + 1
      labels = labels.masked_fill(
          semantic_labels == self.semantic_ignore_index,
          ignore_index)
    else:
      labels = None
      ignore_index = None

    # Spherical KMeans clustering.
    (cluster_embeddings,
     cluster_embeddings_with_loc,
     cluster_labels,
     cluster_indices,
     cluster_batch_indices) = (
       segsort_common.segment_by_kmeans(
           embeddings,
           labels,
           self.kmeans_num_clusters,
           local_features=local_features,
           ignore_index=ignore_index,
           iterations=self.kmeans_iterations))

    cluster_semantic_labels = cluster_labels // self.label_divisor
    cluster_instance_labels = cluster_labels % self.label_divisor

    # Aug Spherical KMeans clustering.
    (cluster_embeddings_aug,
     cluster_embeddings_with_loc_aug,
     _,
     cluster_indices_aug,
     _) = (
       segsort_common.segment_by_kmeans(
           embeddings_aug,
           labels,
           self.kmeans_num_clusters,
           local_features=local_features,
           ignore_index=ignore_index,
           iterations=self.kmeans_iterations))

    outputs = {
      'cluster_embedding': cluster_embeddings,
      'cluster_embedding_with_loc': cluster_embeddings_with_loc,
      'cluster_embedding_aug': cluster_embeddings_aug,
      'cluster_embedding_with_loc_aug': cluster_embeddings_with_loc_aug,
      'cluster_semantic_label': cluster_semantic_labels,
      'cluster_instance_label': cluster_instance_labels,
      'cluster_index': cluster_indices,
      'cluster_index_aug': cluster_indices_aug,
      'cluster_batch_index': cluster_batch_indices,
    }

    return outputs

  def forward(self, datas, targets=None, resize_as_input=None):
    """Generate pixel-wise embeddings and Spherical Kmeans clustering
    within each image.
    """
    targets = targets if targets is not None else {}

    # Generaet embeddings.
    outputs = self.generate_embeddings(datas, targets, resize_as_input)

    if self.use_segsort_flag:
      if self.training:
          # Resize labels to embedding size.
          semantic_labels = targets.get('semantic_label', None)
          if semantic_labels is not None:
            semantic_labels = common_utils.resize_labels(
                semantic_labels, outputs['embedding'].shape[-2:])

          instance_labels = targets.get('instance_label', None)
          if instance_labels is not None:
            instance_labels = common_utils.resize_labels(
                instance_labels, outputs['embedding'].shape[-2:])

          # Generate clusterings.
          cluster_embeddings = self.generate_clusters(
              outputs['embedding'],
              outputs['embedding_aug'],
              semantic_labels,
              instance_labels,
              outputs['local_feature'])

          outputs.update(cluster_embeddings)

    return outputs

  def get_params_lr(self):
    ret = []
    return ret


def localvit_swinfusenet(config):
  """PSPNet with resnet50 backbone.
  """
  return LocalVIT_SwinFuseNet(config)

if __name__ == '__main__':
  model = LocalVIT_SwinFuseNet(config).cuda()
  model.eval()
  img = {}
  img['image'] = torch.rand(8, 3, 224, 224).cuda()
  img['depth'] = torch.rand(8, 3, 224, 224).cuda()
  for i in range(100):
    out = model(img)
  print('output.keys()=', out.keys())
  print('out.embedding.shape=', out['embedding'].shape)

