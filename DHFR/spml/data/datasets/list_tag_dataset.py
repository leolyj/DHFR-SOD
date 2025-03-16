"""Classes for Dataset with image-level tags.
"""

import cv2
import numpy as np

from spml.data.datasets.base_dataset import ListDataset
import spml.data.transforms as transforms
from PIL import Image


class ListTagDataset(ListDataset):
  """Class of dataset which takes a file of paired list of 
  images and labels. This class returns image-level tags
  from semantic labels.
  """
  def __init__(self,
               data_dir,
               data_list,
               img_mean=(0, 0, 0),
               img_std=(1, 1, 1),
               size=None,
               random_crop=False,
               random_scale=False,
               random_mirror=False,
               training=False):
    """Class for Image-level Tags Dataset.

    Args:
      data_dir: A string indicates root directory of images and labels.
      data_list: A list of strings which indicate path of paired images
        and labels. 'image_path semantic_label_path instance_label_path'.
      img_mean: A list of scalars indicate the mean image value per channel.
      img_std: A list of scalars indicate the std image value per channel.
      size: A tuple of scalars indicate size of output image and labels.
        The output resolution remain the same if `size` is None.
      random_crop: enable/disable random_crop for data augmentation.
        If True, adopt randomly cropping as augmentation.
      random_scale: enable/disable random_scale for data augmentation.
        If True, adopt adopt randomly scaling as augmentation.
      random_mirror: enable/disable random_mirror for data augmentation.
        If True, adopt adopt randomly mirroring as augmentation.
      training: enable/disable training to set dataset for training and
        testing. If True, set to training mode.
    """
    super(ListTagDataset, self).__init__(
        data_dir,
        data_list,
        img_mean,
        img_std,
        size,
        random_crop,
        random_scale,
        random_mirror,
        training)

  def _get_datas_by_index(self, idx):
    """Return image_path, semantic_label_path, instance_label_path
    by the given index.
    """
    image_path = self.image_paths[idx]
    image = self._read_image(image_path)

    if self.depth_paths is not None:
      depth_path = self.depth_paths[idx]
      depth = self._read_depth(depth_path)
    else:
      depth = None

    if len(self.semantic_label_paths) > 0:
      semantic_label_path = self.semantic_label_paths[idx]
      semantic_label = self._read_label(semantic_label_path)
    else:
      semantic_label = None

    if len(self.instance_label_paths) > 0:
      instance_label_path = self.instance_label_paths[idx]
      instance_label = self._read_label(instance_label_path)
    else:
      instance_label = None

    if len(self.contours_sum_paths) > 0:
      contour_sum_label_path = self.contours_sum_paths[idx]
      contour_sum_label = self._read_label(contour_sum_label_path)
    else:
      contour_sum_label = None

    if depth is not None:
        if semantic_label is not None:
          cats = np.unique(semantic_label)
          semantic_tags = np.zeros((256, ), dtype=np.uint8)
          semantic_tags[cats] = 1
        else:
          semantic_tags = None

        return image, depth, semantic_label, instance_label, semantic_tags, contour_sum_label
    else:
        if semantic_label is not None:
          cats = np.unique(semantic_label)
          semantic_tags = np.zeros((256, ), dtype=np.uint8)
          semantic_tags[cats] = 1
        else:
          semantic_tags = None

        return image, semantic_label, instance_label, semantic_tags
        
  def _read_image(self, image_path):
    """Read BGR uint8 image.
    """
    img = np.array(Image.open(image_path).convert(mode='RGB'))
    return img

  def _read_depth(self, depth_path):
    """Read L uint8 image.
    """
    img = np.array(Image.open(depth_path).convert(mode='L'))
    img = np.repeat(np.expand_dims(img, 2), 3, axis=2)
    return img
    
  def _training_preprocess(self, idx):
    """Data preprocessing for training.
    """
    assert(self.size is not None)
    if self.depth_paths is not None:
      image, depth, semantic_label, instance_label, semantic_tags, contour_sum_label = self._get_datas_by_index(idx)

      image = Image.fromarray(image.astype('uint8'))
      depth = Image.fromarray(depth.astype('uint8'))

      # sematic bg, unk, fg: 0 2 1 -> 0 128 255
      semantic_label_mask = semantic_label.copy()
      semantic_label[semantic_label_mask == 2] = 128 # unk
      semantic_label[semantic_label_mask == 1] = 255 # fg
      semantic_label = Image.fromarray(semantic_label.astype('uint8'))

      # instance
      instance_label = instance_label.astype(np.float32)
      instance_label = instance_label + 1
      instance_label_max = instance_label.max()
      instance_label = instance_label / instance_label_max * 255
      instance_label = Image.fromarray(instance_label.astype('uint8'))

      # contours
      contour_sum_label = Image.fromarray(contour_sum_label.astype('uint8'))

      # augmentation
      image,semantic_label,instance_label,depth,contour_sum_label = transforms.cv_random_flip(image,semantic_label,instance_label,depth,contour_sum_label)
      image,semantic_label,instance_label,depth,contour_sum_label = transforms.randomCrop(image, semantic_label,instance_label,depth,contour_sum_label)
      image,semantic_label,instance_label,depth,contour_sum_label = transforms.randomRotation(image, semantic_label,instance_label,depth,contour_sum_label)
      image = transforms.colorEnhance(image)
      #semantic_label = transforms.randomPeper(semantic_label)

      # to float
      new_size = (self.size[0], self.size[1])
      image = transforms.resize_to_scale(np.array(image).astype(np.float32), new_size, 'nearest') / 255
      depth = transforms.resize_to_scale(np.array(depth).astype(np.float32), new_size, 'nearest') / 255
      semantic_label = transforms.resize_to_scale(np.array(semantic_label).astype(np.float32), new_size, 'nearest') / 255
      instance_label = transforms.resize_to_scale(np.array(instance_label).astype(np.float32), new_size, 'nearest') / 255
      contour_sum_label = transforms.resize_to_scale(np.array(contour_sum_label).astype(np.float32), new_size, 'nearest') / 255
      image = image - np.array(self.img_mean, dtype=image.dtype)
      image = image / np.array(self.img_std, dtype=image.dtype)
 
      # semantic ignore uncertainty points
      # 0 255/3 255*2/3 255 -> 0 1 2
      semantic_label_ori = semantic_label.copy()
      semantic_label[semantic_label_ori >= 2 / 3] = 1 # fg
      semantic_label[(semantic_label_ori >= 1 / 3) & (semantic_label_ori < 2 /3)] = 2 # unk
      semantic_label[semantic_label_ori < 1 / 3] = 0 # bg
      semantic_label = semantic_label.astype('uint8')
      
      # instance round
      instance_label = np.round(instance_label * instance_label_max).astype('uint8')

      # debug
      '''
      img = Image.fromarray((image * 128 / np.abs(image).max() + 128).astype('uint8'))
      img.save('/home/liuzhy/test/img.png')
      img = Image.fromarray((depth * 128 / np.abs(depth).max() + 128).astype('uint8'))
      img.save('/home/liuzhy/test/depth.png')
      img = Image.fromarray((semantic_label.astype(np.float32) * 128 / np.abs(semantic_label).max() + 128).astype('uint8'))
      img.save('/home/liuzhy/test/semantic_label.png')
      img = Image.fromarray((instance_label.astype(np.float32) * 128 / np.abs(instance_label).max() + 128).astype('uint8'))
      img.save('/home/liuzhy/test/instance_label.png')
      '''
      return image, depth, semantic_label, instance_label, semantic_tags, contour_sum_label
    else:
      image, semantic_label, instance_label, semantic_tags = self._get_datas_by_index(idx)

      image = Image.fromarray(image.astype('uint8'))

      # sematic bg, unk, fg: 2 0 1 -> 0 128 255
      semantic_label_mask = semantic_label.copy()
      semantic_label[semantic_label_mask == 0] = 128 # unk
      semantic_label[semantic_label_mask == 1] = 255 # fg
      semantic_label[semantic_label_mask == 2] = 0  # bg
      semantic_label = Image.fromarray(semantic_label.astype('uint8'))

      # instance
      instance_label = instance_label.astype(np.float32)
      instance_label = instance_label + 1
      instance_label_max = instance_label.max()
      instance_label = instance_label / instance_label_max * 255
      instance_label = Image.fromarray(instance_label.astype('uint8'))
      
      # augmentation
      image,semantic_label,instance_label = transforms.cv_random_flip(image,semantic_label,instance_label)
      image,semantic_label,instance_label = transforms.randomCrop(image, semantic_label,instance_label)
      image,semantic_label,instance_label = transforms.randomRotation(image, semantic_label,instance_label)
      image = transforms.colorEnhance(image)
      semantic_label = transforms.randomPeper(semantic_label)

      # to float
      new_size = (self.size[0], self.size[1])
      image = transforms.resize_to_scale(np.array(image).astype(np.float32), new_size, 'nearest') / 255
      semantic_label = transforms.resize_to_scale(np.array(semantic_label).astype(np.float32), new_size, 'nearest') / 255
      instance_label = transforms.resize_to_scale(np.array(instance_label).astype(np.float32), new_size, 'nearest') / 255
      image = image - np.array(self.img_mean, dtype=image.dtype)
      image = image / np.array(self.img_std, dtype=image.dtype)
 
      # semantic ignore uncertainty points
      # 0 255/3 255*2/3 255 -> 0 1 2
      semantic_label_ori = semantic_label.copy()
      semantic_label[semantic_label_ori >= 2 / 3] = 1 # fg
      semantic_label[(semantic_label_ori >= 1 / 3) & (semantic_label_ori < 2 /3)] = 2 # unk
      semantic_label[semantic_label_ori < 1 / 3] = 0 # bg
      semantic_label = semantic_label.astype('uint8')
      
      # instance round
      instance_label = np.round(instance_label * instance_label_max).astype('uint8')

      return image, semantic_label, instance_label, semantic_tags

  def __getitem__(self, idx):
    """Retrive image and label by index.
    """
    if self.depth_paths is not None:
      if self.training:
        image, depth, semantic_label, instance_label, semantic_tag, contour_sum_label = self._training_preprocess(idx)
      else:
        raise NotImplementedError()

      inputs = {'image': image.transpose(2, 0, 1),
                'depth': depth.transpose(2, 0, 1)}
      labels = {'semantic_label': semantic_label,
                'instance_label': instance_label,
                'semantic_tag': semantic_tag,
                'image': image.transpose(2, 0, 1),
                'contour_sum_label': contour_sum_label}

      return inputs, labels, idx
    else:
      if self.training:
        image, semantic_label, instance_label, semantic_tag = self._training_preprocess(idx)
      else:
        raise NotImplementedError()

      inputs = {'image': image.transpose(2, 0, 1)}
      labels = {'semantic_label': semantic_label,
                'instance_label': instance_label,
                'semantic_tag': semantic_tag,
                'image': image.transpose(2, 0, 1)}

      return inputs, labels, idx
