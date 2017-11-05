""" An implementation of the paper "A Neural Algorith of Artistic Style"
by Gatys et al. in TensorFlow.

Originated from the assigment of Stanforod CS class 20SI.

"""

import os
import time
import argparse

import numpy as np
import tensorflow as tf

import vgg_model
import utils


# Parameters for style transfer
ITERS = 500
LR = 1

# Parameters for tracking model training
SAVE_EVERY = 100

# Mean pixels used in their paper to 0-center the image, see
# "https://gist.github.com/ksimonyan/211839e770f7b538e2d8" for details
MEAN_PIXELS = np.array([123.68, 116.779, 103.939]).reshape((1, 1, 1, 3))

# Layers used for style features
# Refer to the paper, in general, we may want to give higher weight to
# upper(deeper) layers
STYLE_LAYERS = ["conv1_1", "conv2_1", "conv3_1", "conv4_1", "conv5_1"]
STYLE_W = [0.5, 1.0, 1.5, 3.0, 4.0]

# Layer used for content features
CONTENT_LAYER = "conv4_2"

# Weight used in loss definition
CONTENT_WEIGHT = 0.01
STYLE_WEIGHT = 1

# VGG model file
VGG_DOWNLOAD_LINK ='http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat'
VGG_MODEL = 'imagenet-vgg-verydeep-19.mat'
EXPECTED_BYTES = 534904783


def _create_content_loss(p, f):
  """ Caculate the loss between the feature representation of the content
  and the generated image.

  Args:
    p: tf.tensor, feature representation of content image
    f: tf.tensor, feature representation of generated image

  Returns:
    float, content loss

  Note:
    loss_content = \sum_{i, j}(F_ij^l - P_ij^l)^2
    F: feature representation of generated image
    P: feature representation of content image
  """
  # To make the loss converge faster, we use 1/(4*s) as coef,
  # where s = product of dimension of p
  coef = 1 / (4 * p.size)
  return coef * tf.reduce_sum(tf.squared_difference(p, f))


def _gram_matrix(f, n, m):
  """ Create and return the gram matrix for tensor F.

  Args:
    f: tensor
    n: int, third dimenison of the feature map, i.e., depth
    m: int, 1st * 2nd dimenion of feature map, i.e., height * width

  Returns:
    tf.tensor, 2D, gram matrix

  Note:
    tf.nn.conv2d only accept 4d input, so we need to reshape input first
  """
  f = tf.reshape(f, (m, n))
  return tf.matmul(tf.transpose(f), f)


def _single_style_loss(a, g):
  """ Calculate the style loss at a certain layer.

  Args:
    a: feature representation of the real image
    g: feature representation of the generated image

  Returns:
    float, style loss at one layer
  """
  # number of filters
  n = a.shape[3]
  # height * width
  m = a.shape[1] * a.shape[2]
  a_gram = _gram_matrix(a, n, m)
  g_gram = _gram_matrix(g, n, m)
  coef = np.float32(1 / (4 * (m * n) ** 2))

  return coef * tf.reduce_sum(tf.squared_difference(a_gram, g_gram))


def _create_style_loss(a, model):
  """ Return the total style loss.

  Args:
    a: gram matrix of a for certain layers
    model: Model

  Returns:
    float, total style loss
  """
  n_layers = len(STYLE_LAYERS)
  e = [_single_style_loss(a[i], model[STYLE_LAYERS[i]])
       for i in range(n_layers)]
  return sum([STYLE_W[i] * e[i] for i in range(n_layers)])


def _create_losses(model, input_image, content_image, style_image):
  """ Create all the losses.

  Args:
    model: model object
    input_image: np.array
    content_image: np.array
    style_image: np.array

  Returns:
    content_loss, float
    style_loss, float
    total_loss, float
  """
  # Content loss
  # We have f already, just need to calculate p
  with tf.variable_scope("loss"):
    with tf.Session() as sess:
      # Assign content image to the variable input_image
      sess.run(input_image.assign(content_image))
      # Calculate feature representation at the content layer
      p = sess.run(model[CONTENT_LAYER])
    content_loss = _create_content_loss(p, model[CONTENT_LAYER])

    # Style loss
    with tf.Session() as sess:
      sess.run(input_image.assign(style_image))
      a = sess.run([model[i] for i in STYLE_LAYERS])
    style_loss = _create_style_loss(a, model)

    # Total loss
    total_loss = CONTENT_WEIGHT * content_loss + STYLE_WEIGHT * style_loss

  return content_loss, style_loss, total_loss


def _create_summary(model):
  """ Create summary ops to track the model training.
  """
  with tf.name_scope("summaries"):
    tf.summary.scalar("content loss", model["content_loss"])
    tf.summary.scalar("style loss", model["style_loss"])
    tf.summary.scalar("total loss", model["total_loss"])
    tf.summary.histogram("histogram content loss", model["content_loss"])
    tf.summary.histogram("histogram style loss", model["style_loss"])
    tf.summary.histogram("histogram total loss", model["total_loss"])
    return tf.summary.merge_all()


def train(model, generated_image, initial_image):
  """ Train the model.

  Args:
    model: dict, with all defined ops for model
    generated_image: tf.tensor
    initial_image: np.array

  Returns:
    None
  """
  skip_step = 50
  with tf.Session() as sess:
    # Create Saver object
    saver = tf.train.Saver()
    # Variable Initialization
    sess.run(tf.global_variables_initializer())
    # Create writer object
    writer = tf.summary.FileWriter("./graphs/", sess.graph)

    #
    sess.run(generated_image.assign(initial_image))
    # Create checkpoint regularly
    ckpt = tf.train.get_checkpoint_state(
      os.path.dirname("./checkpoints/checkpoint"))
    if ckpt and ckpt.model_checkpoint_path:
      saver.restore(sess, ckpt.model_checkpoint_path)
    # Get the last step of the most recent checkpoint as the startint point
    initial_step = model['global_step'].eval()

    start_time = time.time()
    # Every time the program is called, it will run additional # ITERS steps
    for index in range(initial_step, initial_step + ITERS):
      # Run the model (minimize the loss)
      sess.run(model["optimizer"])

      # Check the summarize the results
      if (index + 1) % skip_step == 0:
        gen_image, total_loss, summary = (sess.run([generated_image,
                                                    model["total_loss"],
                                                    model["summary_op"]]))

        # Add the pixels back
        gen_image = gen_image + MEAN_PIXELS
        writer.add_summary(summary, global_step=index)
        print("Step {}\n Sum: {:5.1f}".format(index + 1, np.sum(gen_image)))
        print("     Loss: {:5.1f}".format(total_loss))
        print("     Time: {:5.1f}".format(time.time() - start_time))
        # Reset time
        start_time = time.time()

        file_name = "./outputs/{}.jpeg".format(index)
        utils.save_image(file_name, gen_image)

        # Save model
        if (index + 1) % SAVE_EVERY == 0:
          saver.save(sess, "./checkpoints/checkpoints", index)


def main():
  """ main function to run the style transfer.
  """
  parser = argparse.ArgumentParser()
  parser.add_argument("-s", "--style", type=str, help="style image")
  parser.add_argument("-c", "--content", type=str, help="content image")
  parser.add_argument("--height", type=int, default=255,
                      help="image height")
  parser.add_argument("--width", type=int, default=333,
                      help="image width")
  parser.add_argument("-nr", "--noise_ratio", type=int, default=0.5,
                      help="noise ratio when combing images")

  args = parser.parse_args()
  CONTENT_IMAGE = args.content
  STYLE_IMAGE = args.style
  IMAGE_HEIGHT = args.height
  IMAGE_WIDTH = args.width
  NR = args.noise_ratio

  with tf.variable_scope("input"):
    # Define input image as variable, so we can directly
    # modify it to get the output image
    input_image = tf.Variable(np.zeros([1, IMAGE_HEIGHT, IMAGE_WIDTH, 3]),
                              dtype=np.float32)

  # Download the model and make new directories
  utils.download_model(VGG_DOWNLOAD_LINK, VGG_MODEL, EXPECTED_BYTES)
  utils.make_dir("./checkpoints")
  utils.make_dir("./graphs")
  utils.make_dir("./outputs")

  # Initialize the model with given input image
  vgg = vgg_model.VGG_loader(file_path=VGG_MODEL)
  model = vgg.load_image(input_image)
  # Initialize the global step
  model["global_step"] = tf.Variable(
    0, dtype=tf.int32, trainable=False, name="global_step")

  content_image = utils.get_resized_image(
    CONTENT_IMAGE, IMAGE_HEIGHT, IMAGE_WIDTH)
  content_image -= MEAN_PIXELS

  style_image = utils.get_resized_image(
    STYLE_IMAGE, IMAGE_HEIGHT, IMAGE_WIDTH)
  style_image -= MEAN_PIXELS

  model["content_loss"], model["style_loss"], model["total_loss"] = (
    _create_losses(model, input_image, content_image, style_image)
  )

  # Define the optimizer
  model["optimizer"] = (
    tf.train.AdamOptimizer(LR).minimize(model["total_loss"])
  )

  # Define the summary op
  model["summary_op"] = _create_summary(model)

  # Generate initial image
  initial_image = (
    utils.generate_noise_image(content_image, IMAGE_HEIGHT, IMAGE_WIDTH, NR)
  )

  # Finally, run the optimizer to get the style transferred image
  train(model, input_image, initial_image)


if __name__ == "__main__":
  main()
