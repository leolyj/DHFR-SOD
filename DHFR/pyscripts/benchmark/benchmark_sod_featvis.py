from __future__ import print_function, division
import os
import argparse
from spml.config.default import config, update_config

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
  parser.add_argument('--init_gpu_id', type=int, default=0,
                      help='init gpu id.')
  args, rest = parser.parse_known_args()

  # Update config with arguments.
  update_config(args.cfg_path)

  args = parser.parse_args()

  return args


global_args = parse_args('benchmark for sod.')
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(global_args.init_gpu_id)

import torch
print("total gpu cards: ", torch.cuda.device_count())

import math

import PIL.Image as Image
import numpy as np
import cv2
import torch.backends.cudnn as cudnn

import spml.utils.general.vis as vis_utils
from spml.data.datasets.base_dataset import ListDataset
from spml.models.embeddings.resnet_pspnet import resnet_50_pspnet, resnet_101_pspnet
from spml.models.embeddings.resnet_deeplab import resnet_50_deeplab, resnet_101_deeplab
from spml.models.predictions.softmax_classifier import softmax_classifier
from spml.models.embeddings.localvit_swinfusenet import localvit_swinfusenet


from metric import CalTotalMetric
from tqdm import tqdm
from misc import make_log
from torchvision import transforms


cudnn.enabled = True
cudnn.benchmark = True


def separate_comma(str_comma):
  ints = [int(i) for i in str_comma.split(',')]
  return ints


def main():
    """Inference for semantic segmentation.
    """
    # Retreve experiment configurations.
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

    prediction_model = softmax_classifier(config).cuda()
    embedding_model.eval()
    prediction_model.eval()

    # Load trained weights.
    model_path_template = os.path.join(global_args.snapshot_dir, 'model-{:d}.pth')
    save_iter = config.train.max_iteration - 1
    embedding_model.load_state_dict(
        torch.load(model_path_template.format(save_iter))['embedding_model'])

    #test
    test_datasets = ['LFSD', 'NJU2K','NLPR', 'DES', 'SIP', 'SSD', 'STERE']
    for dataset in test_datasets:
        save_path = os.path.join(global_args.save_dir, dataset)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # test data list
        data_list = os.path.join(global_args.test_list_root, dataset+'.txt')
        
        # Create data loaders.
        test_dataset = ListDataset(
            data_dir=global_args.data_dir,
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

        # create metrics
        cal_total_metrics = CalTotalMetric(num=len(test_dataset), beta_for_wfm=1)

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

          vis_rgb = vis_utils.embedding_to_rgb(embeddings['embedding'], 'pca') / 255
          pred_pil = to_pil(vis_rgb.squeeze(dim=0).data.cpu()).resize(config.test.crop_size, resample=Image.NEAREST)
          pred_pil.save(os.path.join(save_path, base_name))

if __name__ == '__main__':
  main()

