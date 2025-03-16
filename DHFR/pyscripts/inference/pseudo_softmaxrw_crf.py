"""Genereate pseudo labels by softmax classifier, random walk and CRF.
"""
from __future__ import print_function, division
import os
import math

import PIL.Image as Image
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
import argparse

import spml.utils.general.vis as vis_utils
from spml.data.datasets.base_dataset import ListDataset
from spml.config.default import config, update_config
from spml.models.embeddings.resnet_pspnet import resnet_50_pspnet, resnet_101_pspnet
from spml.models.embeddings.resnet_deeplab import resnet_50_deeplab, resnet_101_deeplab
from spml.models.predictions.softmax_classifier import softmax_classifier
from spml.models.embeddings.localvit_swinfusenet import localvit_swinfusenet

from tqdm import tqdm
from torchvision import transforms
from spml.models.crf import DenseCRF

cudnn.enabled = True
cudnn.benchmark = True


def parse_args(description=''):
  """Parse CLI arguments.
  """
  parser = argparse.ArgumentParser(description=description)
  # Misc parameters.
  parser.add_argument('--snapshot_dir', required=True, type=str,
                      help='/path/to/snapshot/dir.')
  parser.add_argument('--save_dir', type=str,
                      help='/path/to/save/dir.')
  parser.add_argument('--cfg_path', required=True, type=str,
                      help='/path/to/specific/config/file.')
  parser.add_argument('--data_dir', type=str, default=None,
                      help='/root/dir/to/data.')
  parser.add_argument('--test_list_root', type=str, default=None,
                      help='/root/dir/to/data.')
  # Network parameters.
  parser.add_argument('--kmeans_num_clusters', type=str,
                      help='H,W')
  parser.add_argument('--label_divisor', type=int,
                      help=2048)
  # DenseCRF parameters.
  parser.add_argument('--crf_iter_max', type=int, default=10,
                      help='number of iteration for crf.')
  parser.add_argument('--crf_pos_xy_std', type=int, default=1,
                      help='hyper paramter of crf.')
  parser.add_argument('--crf_pos_w', type=int, default=3,
                      help='hyper paramter of crf.')
  parser.add_argument('--crf_bi_xy_std', type=int, default=67,
                      help='hyper paramter of crf.')
  parser.add_argument('--crf_bi_w', type=int, default=4,
                      help='hyper paramter of crf.')
  parser.add_argument('--crf_bi_rgb_std', type=int, default=3,
                      help='hyper paramter of crf.')
  args, rest = parser.parse_known_args()

  # Update config with arguments.
  update_config(args.cfg_path)

  args = parser.parse_args()

  return args


def separate_comma(str_comma):
  ints = [int(i) for i in str_comma.split(',')]
  return ints


def main():
  """Inference for semantic segmentation.
  """
  # Retreve experiment configurations.
  args = parse_args('benchmark for sod.')
  to_pil = transforms.ToPILImage()
  gt_resize = transforms.Resize(config.test.crop_size)

  # Create models.
  if config.network.backbone_types == 'panoptic_pspnet_101':
    embedding_model = resnet_101_pspnet(config).cuda()
  elif config.network.backbone_types == 'panoptic_deeplab_101':
    embedding_model = resnet_101_deeplab(config).cuda()
  elif config.network.backbone_types in ['rgb_sod_sfnet', 'rgbd_sod_sfnet']:
    embedding_model = localvit_swinfusenet(config).cuda()
  else:
    raise ValueError('Not support ' + config.network.backbone_types)

  # Define CRF.
  postprocessor = DenseCRF(
    iter_max=args.crf_iter_max,
    pos_xy_std=args.crf_pos_xy_std,
    pos_w=args.crf_pos_w,
    bi_xy_std=args.crf_bi_xy_std,
    bi_rgb_std=args.crf_bi_rgb_std,
    bi_w=args.crf_bi_w, )

  prediction_model = softmax_classifier(config).cuda()
  embedding_model.eval()
  prediction_model.eval()

  # Load trained weights.
  model_path_template = os.path.join(args.snapshot_dir, 'model-{:d}.pth')
  save_iter = config.train.max_iteration - 1
  embedding_model.load_state_dict(
    torch.load(model_path_template.format(save_iter))['embedding_model'])

  # test
  test_datasets = ['test_in_train']
  for dataset in test_datasets:
    save_semantic_path = os.path.join(args.save_dir, 'semantic')
    save_gray_path = os.path.join(args.save_dir, 'gray')
    if not os.path.exists(save_semantic_path):
      os.makedirs(save_semantic_path)
    if not os.path.exists(save_gray_path):
      os.makedirs(save_gray_path)

    # test data list
    data_list = os.path.join(args.test_list_root, dataset + '.txt')

    # Create data loaders.
    test_dataset = ListDataset(
      data_dir=args.data_dir,
      data_list=data_list,
      img_mean=config.network.pixel_means,
      img_std=config.network.pixel_stds,
      size=config.test.crop_size,
      random_crop=False,
      random_scale=False,
      random_mirror=False,
      training=False)
    test_image_paths = test_dataset.image_paths
    have_depth_flag = 0
    if 'depth' in test_dataset[0][0].keys():
      have_depth_flag = 1

    # Start inferencing.
    test_tensor = torch.ones(len(test_dataset))
    tqdm_iter = tqdm(enumerate(test_tensor), total=len(test_dataset), leave=False)
    for data_index, temp1 in tqdm_iter:
      tqdm_iter.set_description(f"{dataset}:" f"te=>{data_index + 1}")

      # Image path.
      image_path = test_image_paths[data_index]
      base_name = os.path.basename(image_path).replace('.jpg', '.png')

      # Image resolution.
      image_batch, semantic_batch, _ = test_dataset[data_index]

      # Resize the input image.
      if have_depth_flag:
        resize_image_h, resize_image_w = image_batch['image'].shape[-2:]
        image_batch['image'] = torch.from_numpy(image_batch['image']).cuda().unsqueeze(dim=0)
        image_batch['depth'] = torch.from_numpy(image_batch['depth']).cuda().unsqueeze(dim=0)
      else:
        resize_image_h, resize_image_w = image_batch['image'].shape[-2:]
        image_batch['image'] = torch.from_numpy(image_batch['image']).cuda().unsqueeze(dim=0)

      # Feed-forward.
      embeddings = embedding_model(image_batch)
      outputs = prediction_model(embeddings)

      # Save semantic predictions.
      semantic_pred = outputs.get('semantic_pred', None)
      if semantic_pred is not None:
        semantic_pred = semantic_pred.view(1, resize_image_h, resize_image_w)
        semantic_pred = torch.cat((semantic_pred, 1 - semantic_pred), dim=0)
        semantic_pred = (semantic_pred
                         .cpu()
                         .detach()
                         .numpy()
                         .astype(np.float32))

        # CRF
        image = image_batch['image'].squeeze().permute(1, 2, 0).contiguous().data.cpu().numpy().astype(np.float32)
        image *= np.reshape(config.network.pixel_stds, (1, 1, 3))
        image += np.reshape(config.network.pixel_means, (1, 1, 3))
        image = image * 255
        image = image.astype(np.uint8)
        semantic_pred_crf = postprocessor(image, semantic_pred)
        semantic_pred_crf = torch.from_numpy(semantic_pred_crf[0]).squeeze()

        # save
        semantic_pred_name = os.path.join(save_semantic_path, base_name)
        gray_pred_name = os.path.join(save_gray_path, base_name)
        ori_w, ori_h = semantic_batch['semantic_label'].shape
        pred_pil = to_pil(semantic_pred_crf).resize((ori_h, ori_w), resample=Image.NEAREST)
        pred_pil.save(gray_pred_name)
        pred_data = np.array(pred_pil).astype('float')
        pred_data = np.round(pred_data / 255)
        pred_pil = Image.fromarray(pred_data.astype('uint8'))
        pred_pil.save(semantic_pred_name)

        # Clean GPU memory cache to save more space.
        outputs = {}
        torch.cuda.empty_cache()

if __name__ == '__main__':
  main()
