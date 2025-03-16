"""Utility functions to process images.
"""

import numpy as np
import cv2
from PIL import Image
from PIL import ImageEnhance
import random


def resize(image,
           label,
           ratio,
           image_method='bilinear',
           label_method='nearest',
           depth=None):
  """Rescale image and label to the same size by the specified ratio.
  The aspect ratio is remained the same after rescaling.

  Args:
    image: A 2-D/3-D tensor of shape `[height, width, channels]`.
    label: A 2-D/3-D tensor of shape `[height, width, channels]`.
    ratio: A float/integer indicates the scaling ratio.
    image_method: Image resizing method. bilinear/nearest.
    label_method: Image resizing method. bilinear/nearest.

  Return:
    Two tensors of shape `[new_height, new_width, channels]`.
  """
  h, w = image.shape[:2]
  new_h, new_w = int(ratio * h), int(ratio * w)

  inter_image = (cv2.INTER_LINEAR if image_method == 'bilinear'
                  else cv2.INTER_NEAREST)
  new_image = cv2.resize(image, (new_w, new_h), interpolation=inter_image)

  inter_label = (cv2.INTER_LINEAR if label_method == 'bilinear'
                  else cv2.INTER_NEAREST)
  new_label = cv2.resize(label, (new_w, new_h), interpolation=inter_label)

  if depth is not None:
    inter_depth = (cv2.INTER_LINEAR if image_method == 'bilinear'
                  else cv2.INTER_NEAREST)
    new_depth = cv2.resize(depth, (new_w, new_h), interpolation=inter_depth)
    return new_image, new_label, new_depth

  return new_image, new_label


def random_resize(image,
                  label,
                  scale_min=1.0,
                  scale_max=1.0,
                  image_method='bilinear',
                  label_method='nearest',
                  depth=None):
  """Randomly rescale image and label to the same size. The
  aspect ratio is remained the same after rescaling.

  Args:
    image: A 2-D/3-D tensor of shape `[height, width, channels]`.
    label: A 2-D/3-D tensor of shape `[height, width, channels]`.
    scale_min: A float indicates the minimum scaling ratio.
    scale_max: A float indicates the maximum scaling ratio.
    image_method: Image resizing method. bilinear/nearest.
    label_method: Image resizing method. bilinear/nearest.

  Return:
    Two tensors of shape `[new_height, new_width, channels]`.
  """
  assert(scale_max >= scale_min)
  ratio = np.random.uniform(scale_min, scale_max)
  return resize(image, label, ratio, image_method, label_method, depth=depth)


def mirror(image, label, depth=None):
  """Horizontally flipp image and label.

  Args:
    image: A 2-D/3-D tensor of shape `[height, width, channels]`.
    label: A 2-D/3-D tensor of shape `[height, width, channels]`.

  Return:
    Two tensors of shape `[new_height, new_width, channels]`.
  """

  image = image[:, ::-1, ...]
  label = label[:, ::-1, ...]
  if depth is not None:
    depth = depth[:, ::-1, ...]
    return image, label, depth
  return image, label


def random_mirror(image, label, depth=None):
  """Randomly horizontally flipp image and label.

  Args:
    image: A 2-D/3-D tensor of shape `[height, width, channels]`.
    label: A 2-D/3-D tensor of shape `[height, width, channels]`.

  Return:
    Two tensors of shape `[new_height, new_width, channels]`.
  """
  is_flip = np.random.uniform(0, 1.0) >= 0.5
  if is_flip:
    if depth is not None:
      image, label, depth = mirror(image, label, depth=depth)
      return image, label, depth
    else:
      image, label = mirror(image, label)
      return image, label


def resize_with_interpolation(image, larger_size, method='bilinear'):
  """Rescale image with larger size as `larger_size`. The aspect
  ratio is remained the same after rescaling.

  Args:
    image: A 2-D/3-D tensor of shape `[height, width, channels]`.
    larger_size: An interger indicates the target size of larger side.
    method: Image resizing method. bilinear/nearest.

  Return:
    A tensor of shape `[new_height, new_width, channels]`.
  """
  h, w = image.shape[:2]
  new_size = float(larger_size)
  ratio = np.minimum(new_size / h, new_size / w)
  new_h, new_w = int(ratio * h), int(ratio * w)

  inter = (cv2.INTER_LINEAR if method == 'bilinear'
                  else cv2.INTER_NEAREST)
  new_image = cv2.resize(image, (new_w, new_h), interpolation=inter)

  return new_image


def resize_to_scale(image, scale, method='bilinear'):
  """Rescale image with larger size as `larger_size`. The aspect
  ratio is remained the same after rescaling.

  Args:
    image: A 2-D/3-D tensor of shape `[height, width, channels]`.
    larger_size: An interger indicates the target size of larger side.
    method: Image resizing method. bilinear/nearest.

  Return:
    A tensor of shape `[new_height, new_width, channels]`.
  """
  inter = (cv2.INTER_LINEAR if method == 'bilinear'
                  else cv2.INTER_NEAREST)
  new_image = cv2.resize(image, scale, interpolation=inter)

  return new_image


def resize_with_pad(image, size, image_pad_value=0, pad_mode='left_top'):
  """Upscale image by pad to the width and height.

  Args:
    image: A 2-D/3-D tensor of shape `[height, width, channels]`.
    size: A tuple of integers indicates the target size.
    image_pad_value: An integer indicates the padding value.
    pad_mode: Padding mode. left_top/center.

  Return:
    A tensor of shape `[new_height, new_width, channels]`.
  """
  h, w = image.shape[:2]
  new_shape = list(image.shape)
  new_shape[0] = h if h > size[0] else size[0]
  new_shape[1] = w if w > size[1] else size[1]
  pad_image = np.zeros(new_shape, dtype=image.dtype)

  if isinstance(image_pad_value, int) or isinstance(image_pad_value, float):
    pad_image.fill(image_pad_value)
  else:
    for ind_ch, val in enumerate(image_pad_value):
      pad_image[:, :, ind_ch].fill(val)

  if pad_mode == 'center':
    s_y = (new_shape[0] - h) // 2
    s_x = (new_shape[1] - w) // 2
    pad_image[s_y:s_y+h, s_x:s_x+w, ...] = image
  elif pad_mode == 'left_top':
    pad_image[:h, :w, ...] = image
  else:
    raise ValueError('Unsupported padding mode')

  return pad_image


def random_crop_with_pad(image,
                         label,
                         crop_size,
                         image_pad_value=0,
                         label_pad_value=255,
                         pad_mode='left_top',
                         return_bbox=False, 
                         depth=None):
  """Randomly crop image and label, and pad them before cropping
  if the size is smaller than `crop_size`.

  Args:
    image: A 2-D/3-D tensor of shape `[height, width, channels]`.
    label: A 2-D/3-D tensor of shape `[height, width, channels]`.
    crop_size: A tuple of integers indicates the cropped size.
    image_pad_value: An integer indicates the padding value.
    label_pad_value: An integer indicates the padding value.
    pad_mode: Padding mode. left_top/center.

  Return:
    Two tensors of shape `[new_height, new_width, channels]`.
  """
  image = resize_with_pad(image, crop_size,
                          image_pad_value, pad_mode)
  label = resize_with_pad(label, crop_size,
                          label_pad_value, pad_mode)
  if depth is not None:
    depth = resize_with_pad(depth, crop_size,
                            image_pad_value, pad_mode)
                          

  h, w = image.shape[:2]
  start_h = int(np.floor(np.random.uniform(0, h - crop_size[0])))
  start_w = int(np.floor(np.random.uniform(0, w - crop_size[1])))
  end_h = start_h + crop_size[0]
  end_w = start_w + crop_size[1]

  crop_image = image[start_h:end_h, start_w:end_w, ...]
  crop_label = label[start_h:end_h, start_w:end_w, ...]
  if depth is not None:
    crop_depth = depth[start_h:end_h, start_w:end_w, ...]

  if return_bbox:
    bbox = [start_w, start_h, end_w, end_h]
    if depth is not None:
      return crop_image, crop_label, bbox, crop_depth  
    return crop_image, crop_label, bbox
  else:
    if depth is not None:
      return crop_image, crop_label, crop_depth
    return crop_image, crop_label

#several data augumentation strategies
def cv_random_flip(img, label, instance, depth, contour_sum_label):
  flip_flag = random.randint(0, 1)
  #left right flip
  if flip_flag == 1:
    img = img.transpose(Image.FLIP_LEFT_RIGHT)
    label = label.transpose(Image.FLIP_LEFT_RIGHT)
    instance = instance.transpose(Image.FLIP_LEFT_RIGHT)
    contour_sum_label = contour_sum_label.transpose(Image.FLIP_LEFT_RIGHT)
    if depth is not None:
      depth = depth.transpose(Image.FLIP_LEFT_RIGHT)
  if depth is not None:
    return img, label, instance, depth, contour_sum_label
  return img, label, instance, contour_sum_label
    
def randomCrop(image, label, instance, depth, contour_sum_label):
  border=30
  image_width = image.size[0]
  image_height = image.size[1]
  crop_win_width = np.random.randint(image_width-border , image_width)
  crop_win_height = np.random.randint(image_height-border , image_height)
  random_region = (
      (image_width - crop_win_width) >> 1, (image_height - crop_win_height) >> 1, (image_width + crop_win_width) >> 1,
      (image_height + crop_win_height) >> 1)
  if depth is not None:
    return image.crop(random_region), label.crop(random_region), instance.crop(random_region), depth.crop(random_region), contour_sum_label.crop(random_region)
  return image.crop(random_region), label.crop(random_region), instance.crop(random_region), contour_sum_label.crop(random_region)
  
def randomRotation(image,label,instance,depth,contour_sum_label):
  mode=Image.BICUBIC
  if random.random()>0.8:
    random_angle = np.random.randint(-15, 15)
    image=image.rotate(random_angle, mode)
    label=label.rotate(random_angle, mode)
    instance=instance.rotate(random_angle, mode)
    contour_sum_label=contour_sum_label.rotate(random_angle, mode)
    if depth is not None:
      depth=depth.rotate(random_angle, mode)
  if depth is not None:
    return image,label,instance,depth,contour_sum_label
  return image,label,instance,contour_sum_label

def colorEnhance(image):
  bright_intensity=random.randint(5,15)/10.0
  image=ImageEnhance.Brightness(image).enhance(bright_intensity)
  contrast_intensity=random.randint(5,15)/10.0
  image=ImageEnhance.Contrast(image).enhance(contrast_intensity)
  color_intensity=random.randint(0,20)/10.0
  image=ImageEnhance.Color(image).enhance(color_intensity)
  sharp_intensity=random.randint(0,30)/10.0
  image=ImageEnhance.Sharpness(image).enhance(sharp_intensity)
  return image

def randomGaussian(image, mean=0.1, sigma=0.35):
  def gaussianNoisy(im, mean=mean, sigma=sigma):
      for _i in range(len(im)):
          im[_i] += random.gauss(mean, sigma)
      return im
  img = np.asarray(image)
  width, height = img.shape
  img = gaussianNoisy(img[:].flatten(), mean, sigma)
  img = img.reshape([width, height])
  return Image.fromarray(np.uint8(img))

def randomPeper(img):
  img=np.array(img)
  noiseNum=int(0.0015*img.shape[0]*img.shape[1])
  for i in range(noiseNum):
    randX=random.randint(0,img.shape[0]-1)  
    randY=random.randint(0,img.shape[1]-1)  
    if random.randint(0,1)==0:  
        img[randX,randY]=0  # bg
    else:  
        img[randX,randY]=255 # fg
  return Image.fromarray(img)  

