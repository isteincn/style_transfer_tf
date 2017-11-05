""" A collection of utility functions for TensorFlow implementation of
"A Neural Algorithm of Artistic Style"


Originated from the assigment from Stanforod CS class 20SI.

Author: Yang Jiao (young0106@gmail.com)
"""

import os

import cv2
import numpy as np
from six.moves import urllib


def download_model(download_link, file_name, expected_bytes):
  """ Download the pretrained VGG-19 model, if necessary.
  Args:
    download_link: str, url for the model.
    file_name: str, name of the file
    expected_bytes: int, to check the downloaded model.

  Returns:
    None
  """

  # Check if the model exists:
  if os.path.exists(file_name):
    print("Model already exists.")
    return

  # If not, download the model
  print("Downloading the model...")
  model, _ = urllib.request.urlretrive(
    url=download_link, filename=file_name)
  file_stat = os.stat(model)
  if file_stat.st_size == expected_bytes:
    print("Successfully downloaded the pre-trained model {}".format(file_name))
  else:
    raise Exception("File {} mighted be corrupted. Try downloading again.")


def get_resized_image(image_path, height, width, save=True):
  """ Resize the image.

  Args:
    image_path: str, where the image is stored
    height: int, height of the resized image
    width: int, width of the resized image
    save: boolean, whether to save the image to disk

  Returns:
    resized_image: np.array with dtype=np.float32
  """
  image = cv2.imread(image_path)
  resized_image = cv2.resize(image, (width, height))

  # cv2 reads image as "GBR" color, and we need to convert it to "RGB" first
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

  if save:
    # create the output path
    image_dirs = image_path.split("/")
    image_dirs[-1] = "resized_" + image_dirs[-1]
    out_path = os.path.join(*image_dirs)
    if not os.path.exists(out_path):
      print("Save resized image to disk...")
      cv2.imwrite(out_path, resized_image)

  return resized_image.astype(np.float32)[np.newaxis, ]


def generate_noise_image(content_image, height, width, noise_ratio=0.5):
  """ Generate noise image from the given image.

  Args:
    content_image: np.array, array representation of the content image
    height: int, height of the output image
    wdith: int, width of the output image

  Returns:
    noise_image: np.array, noised image
  """
  rand_image = np.random.uniform(
    low=-20, high=20, size=(1, height, width, 3)).astype(np.float32)

  noise_image = noise_ratio * rand_image + (1 - noise_ratio) * content_image

  return noise_image


def save_image(out_path, image):
  """ Save the clipped image to the disk.

  Args:
    out_path: str, path for output image
    image: np.array, the image to save

  Returns:
    None
  """
  # image has four dimension, [batch (1), height, weight, channel]
  # but to write to disk, we only need the latter three dimensions
  image = image[0]
  out_image = np.clip(image, a_min=0, a_max=255).astype("uint8")
  cv2.imwrite(out_path, out_image)


def make_dir(path):
  """ Create a directory if there isn't one already.

  Args:
    path: str, path of the directoy
  """
  try:
    os.mkdir(path)
  except FileExistsError:
    pass
