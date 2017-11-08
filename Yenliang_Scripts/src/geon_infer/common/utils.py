import os
from os import makedirs
from os.path import splitext, basename, join, isdir
import sys
import json

from PIL import Image
import numpy as np
import rasterio
import pdb


def _makedirs(path):
    try:
        makedirs(path)
    except:
        pass

def load_rasterio(file_path, window=None):
    with rasterio.open(file_path, 'r+') as r:
        # r.read(window=window) (4,6000,6000)
        return np.transpose(r.read(window=window), axes=[1, 2, 0])

def load_pillow(file_path, window=None):
    im = Image.open(file_path)
    if window is not None:
        ((row_begin, row_end), (col_begin, col_end)) = window
        box = (col_begin, row_begin, col_end, row_end)
        im = im.crop(box)
    im = np.array(im)
    if len(im.shape) == 2:
        im = np.expand_dims(im, axis=2)
    return im


def load_img(file_path, window=None):
  ext = splitext(file_path)[1]
  if ext in ['.tif', '.tiff']:
    return load_rasterio(file_path, window)
  return load_pillow(file_path, window)

def get_rasterio_size(file_path):
  with rasterio.open(file_path, 'r+') as r:
      nb_rows, nb_cols = r.height, r.width
      return nb_rows, nb_cols

def get_pillow_size(file_path):
  im = Image.open(file_path)
  nb_cols, nb_rows = im.size
  return nb_cols, nb_rows

def get_img_size(file_path):
  ext = splitext(file_path)[1]
  if ext in ['.tif', '.tiff']:
    return get_rasterio_size(file_path)
  return get_pillow_size(file_path)  

def save_rasterio(im, file_path):
  height, width, count = im.shape
  with rasterio.open(file_path, 'w', driver='GTiff', height=height,
                     compression=rasterio.enums.Compression.none,
                     width=width, count=count, dtype=rasterio.uint8) as dst:
      for channel_ind in range(count):
          dst.write(im[:, :, channel_ind], channel_ind + 1)

def save_pillow(im, file_path):
  im = Image.fromarray(im)
  im.save(file_path)

def save_img(im, file_path):
  ext = splitext(file_path)[1]
  if ext in ['.tif', '.tiff']:
    save_rasterio(im, file_path)
  else:
    save_pillow(im, file_path)

def save_numpy_array(file_path, arr):
  np.save(file_path, arr.astype(np.uint8))

def expand_dims(func):
  def wrapper(self, batch):
      ndim = batch.ndim
      if ndim == 3:
          batch = np.expand_dims(batch, axis=0)
      batch = func(self, batch)
      if ndim == 3:
          batch = np.squeeze(batch, axis=0)
      return batch
  return wrapper

def safe_divide(a, b):
  with np.errstate(divide='ignore', invalid='ignore'):
      c = np.true_divide(a, b)
      c[c == np.inf] = 0
      c = np.nan_to_num(c)
      return c

def get_channel_stats(batch):
  nb_channels = batch.shape[3]
  channel_data = np.reshape(
      np.transpose(batch, [3, 0, 1, 2]), (nb_channels, -1))

  means = np.mean(channel_data, axis=1)
  stds = np.std(channel_data, axis=1)
  return (means, stds)

def setup_run(run_path):  
  _makedirs(run_path)
  return run_path
