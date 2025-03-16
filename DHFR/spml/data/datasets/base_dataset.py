"""Base classes for Dataset.
"""

import os

import torch
import torch.utils.data
import numpy as np
import PIL.Image as Image
import cv2

import spml.data.transforms as transforms


class ListDataset(torch.utils.data.Dataset):
  """Base class of dataset which takes a file of paired list of
  images, semantic labels and instance labels.
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
    """Base class for Dataset.

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
    read_result = (
      self._read_image_and_label_paths(data_dir, data_list))
    self.depth_paths = None
    self.contours_sum_paths = None
    if len(read_result) == 5:
      self.image_paths, self.depth_paths, self.semantic_label_paths, self.instance_label_paths, self.contours_sum_paths = read_result
    else:
      self.image_paths, self.semantic_label_paths, self.instance_label_paths = read_result

    self.training = training
    self.img_mean = img_mean
    self.img_std = img_std
    self.size = size
    self.random_crop = random_crop
    self.random_scale = random_scale
    self.random_mirror = random_mirror

  def eval(self):
    """Set the dataset to evaluation mode.
    """
    self.training = False

  def train(self):
    """Set the dataset to training mode.
    """
    self.training = True

  def _read_image_and_label_paths(self, data_dir, data_list):
    """Parse strings into lists of image, semantic label and
    instance label paths.

    Args:
      data_dir: A string indicates root directory of images and labels.
      data_list: A list of strings which indicate path of paired images
        and labels. 'image_path semantic_label_path instance_label_path'.

    Return:
      Threee lists of file paths.
    """
    images, depths, semantic_labels, instance_labels, contours_sum_labels = [], [], [], [], []
    with open(data_list, 'r') as list_file:
      for line in list_file:
        line = line.strip('\n')

        contours_sum = None
        try:
          img, depth, semantic_lab, instance_lab, contours_sum = line.split(' ') 
        except:
          try:
            img, depth, semantic_lab, instance_lab = line.split(' ') 
          except:
            try:
              if ('depth' in line) or ('Depth' in line):
                img, depth, semantic_lab = line.split(' ') 
                instance_lab = None
              else:
                img, semantic_lab, instance_lab = line.split(' ') 
                depth = None
            except:
              try:
                img, semantic_lab = line.split(' ') 
                depth = None
                instance_lab = None
              except:
                img = line
                depth = semantic_lab = instance_lab = None

        images.append(os.path.join(data_dir, img))

        if depth is not None:
          depths.append(os.path.join(data_dir, depth))

        if semantic_lab is not None:
          semantic_labels.append(os.path.join(data_dir, semantic_lab))

        if instance_lab is not None:
          instance_labels.append(os.path.join(data_dir, instance_lab))

        if contours_sum is not None:
          contours_sum_labels.append(os.path.join(data_dir, contours_sum))

    if len(depths) > 0:
      return images, depths, semantic_labels, instance_labels, contours_sum_labels
    else:
      return images, semantic_labels, instance_labels, contours_sum_labels

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

  def _read_label(self, label_path):
    """Read uint8 label.
    """
    return np.array(Image.open(label_path).convert(mode='L'))

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
      contours_sum_path = self.contours_sum_paths[idx]
      contour_sum_label = self._read_label(contours_sum_path)
    else:
      contour_sum_label = None

    if depth is not None:
      return image, depth, semantic_label, instance_label, contour_sum_label
    else:
      return image, semantic_label, instance_label, contour_sum_label

  def _training_preprocess(self, idx):
    """Data preprocessing for training.
    """
    assert(self.size is not None)
    if self.depth_paths is not None:
      image, depth, semantic_label, instance_label, contour_sum_label = self._get_datas_by_index(idx)

      label = np.stack([semantic_label, instance_label, contour_sum_label], axis=2)

      semantic_label, instance_label, contour_sum_label = label[..., 0], label[..., 1], label[..., 2]
      return image, depth, semantic_label, instance_label, contour_sum_label
    else:
      image, semantic_label, instance_label = self._get_datas_by_index(idx)

      label = np.stack([semantic_label, instance_label], axis=2)

      semantic_label, instance_label = label[..., 0], label[..., 1]
      return image, semantic_label, instance_label

  def _eval_preprocess(self, idx):
    """Data preprocessing for evaluationg.
    """
    if self.depth_paths is not None:
      image, depth, semantic_label, instance_label, contour_sum_label = self._get_datas_by_index(idx)
      
      if self.size is not None:
        image = Image.fromarray(image.astype('uint8'))
        depth = Image.fromarray(depth.astype('uint8'))

        # to float
        new_size = (self.size[0], self.size[1])
        image = transforms.resize_to_scale(np.array(image).astype(np.float32), new_size, 'nearest') / 255
        depth = transforms.resize_to_scale(np.array(depth).astype(np.float32), new_size, 'nearest') / 255
        image = image - np.array(self.img_mean, dtype=image.dtype)
        image = image / np.array(self.img_std, dtype=image.dtype)

      return image, depth, semantic_label, instance_label, contour_sum_label
    else:
      image, semantic_label, instance_label = self._get_datas_by_index(idx)

      if self.size is not None:
        image = Image.fromarray(image.astype('uint8'))

        # to float
        new_size = (self.size[0], self.size[1])
        image = transforms.resize_to_scale(np.array(image).astype(np.float32), new_size, 'nearest') / 255
        image = image - np.array(self.img_mean, dtype=image.dtype)
        image = image / np.array(self.img_std, dtype=image.dtype)

      return image, semantic_label, instance_label

  def __len__(self):
    """Total number of datas in the dataset.
    """
    return len(self.image_paths)

  def __getitem__(self, idx):
    """Retrive image and label by index.
    """
    contour_sum_label = None
    if self.depth_paths is not None:
      if self.training:
        image, depth, semantic_label, instance_label, contour_sum_label = self._training_preprocess(idx)
      else:
        image, depth, semantic_label, instance_label, contour_sum_label = self._eval_preprocess(idx)
      
      inputs = {'image': image.transpose(2, 0, 1),
                'depth': depth.transpose(2, 0, 1)}
      labels = {'semantic_label': semantic_label,
                'instance_label': instance_label,
                'image': image.transpose(2, 0, 1),
                'contour_sum_label': contour_sum_label}
      
      return inputs, labels, idx
    else:
      if self.training:
        image, semantic_label, instance_label = self._training_preprocess(idx)
      else:
        image, semantic_label, instance_label = self._eval_preprocess(idx)

      inputs = {'image': image.transpose(2, 0, 1)}
      labels = {'semantic_label': semantic_label,
                'instance_label': instance_label,
                'image': image.transpose(2, 0, 1)}

      return inputs, labels, idx

  def _collate_fn_dict_list(self, dict_list):
    """Helper function to collate a list of dictionaries.
    """
    outputs = {}
    for key in dict_list[0].keys():
      values = [d[key] for d in dict_list]
      if values[0] is None:
        values = None
      elif (values[0].dtype == np.uint8
           or values[0].dtype == np.int32
           or values[0].dtype == np.int64):
        values = torch.LongTensor(values)
      elif (values[0].dtype == np.float32
             or values[0].dtype == np.float64):
        values = torch.FloatTensor(values)
      else:
        raise ValueError('Unsupported data type')

      outputs[key] = values

    return outputs

  def collate_fn(self, batch):
    """Customized collate function to group datas into batch.
    """
    inputs, labels, indices = zip(*batch)

    inputs = self._collate_fn_dict_list(inputs)
    labels = self._collate_fn_dict_list(labels)
    indices = torch.LongTensor(indices)

    return inputs, labels, indices
