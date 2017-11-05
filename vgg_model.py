""" Load VGG model weights needed for the implementaion of the paper
"A Neural Algorithm of Artistic Style"


Originated from the assigment from Stanforod CS class 20SI.

Author: Yang Jiao (young0106@gmail.com)
"""

import tensorflow as tf
import scipy.io
import cv2


class VGG_loader(object):
  """ Class to help load VGG model into TensorFLow.
  """

  def __init__(self, file_path):
    """ Initializer.

    Args:
      file_path: path to the model file
    """
    self.vgg_layers = scipy.io.loadmat(file_path)["layers"]

  def _weights(self, layer, expected_layer_name):
    """ Return the weights and biases from VGG model

    Args:
      vgg_layers: np.array, with all layers for VGG
      layer: int, the requested layer
      expected_layer_name: str, the requested layer name

    Returns:
      weights: np.array, weights at the requested layer
      biases: np.array, biases at the requested layer

    """
    weights = self.vgg_layers[0][layer][0][0][2][0][0]
    biases = self.vgg_layers[0][layer][0][0][2][0][1]

    layer_name = self.vgg_layers[0][layer][0][0][0][0]
    assert layer_name == expected_layer_name

    return weights, biases.reshape(biases.size)

  def _conv2d_relu(self, prev_layer, layer, layer_name):
    """ Return the Conv2D layer with RELU using the weights,
    biases from VGG model at the request layer

    Args:
      prev_layer: tf.tensor, the input tensor from previous layer
      layer: int, the index to the current layer in self.vgg_layers
      layer_name: str, name of the current layer

    Returns:
      tf.tensor
    """
    with tf.name_scope(layer_name):
      weights, biases = self._weights(layer, layer_name)
      weights = tf.constant(weights, name="weights")
      biases = tf.constant(biases, name="biases")
      conv2d = tf.nn.conv2d(
        input=prev_layer,
        filter=weights,
        strides=[1, 1, 1, 1],
        padding="SAME",
        name="conv2d")

    return tf.nn.relu(conv2d + biases)

  def _avgpool(self, prev_layer):
    """ Return the average pooling layer.

    Args:
      prev_layer: tf.tensor, the output tensor from the previous layer

    Returns:
      out_tensor: tf.tensor, with average pooling applied
    """
    out_tensor = tf.nn.avg_pool(
      value=prev_layer,
      ksize=[1, 2, 2, 1],
      strides=[1, 2, 2, 1],
      padding="SAME",
      name="avg_pool")

    return out_tensor

  def load_image(self, input_image):
    """ forward propogate the input image through the network

    Args:
      input_image:

    Returns:
      out_graph: dict, output graph
    """
    # Create a dict to store the graph
    graph = {}

    # Details of the model architecture can be found here
    graph["conv1_1"] = self._conv2d_relu(input_image, 0, "conv1_1")
    graph["conv1_2"] = self._conv2d_relu(graph["conv1_1"], 2, "conv1_2")
    graph["avgpool1"] = self._avgpool(graph["conv1_2"])
    graph["conv2_1"] = self._conv2d_relu(graph["avgpool1"], 5, "conv2_1")
    graph["conv2_2"] = self._conv2d_relu(graph["conv2_1"], 7, "conv2_2")
    graph["avgpool2"] = self._avgpool(graph["conv2_2"])
    graph["conv3_1"] = self._conv2d_relu(graph["avgpool2"], 10, "conv3_1")
    graph["conv3_2"] = self._conv2d_relu(graph["conv3_1"], 12, "conv3_2")
    graph["conv3_3"] = self._conv2d_relu(graph["conv3_2"], 14, "conv3_3")
    graph["conv3_4"] = self._conv2d_relu(graph["conv3_3"], 16, "conv3_4")
    graph["avgpool3"] = self._avgpool(graph["conv3_4"])
    graph["conv4_1"] = self._conv2d_relu(graph["avgpool3"], 19, "conv4_1")
    graph["conv4_2"] = self._conv2d_relu(graph["conv4_1"], 21, "conv4_2")
    graph["conv4_3"] = self._conv2d_relu(graph["conv4_2"], 23, "conv4_3")
    graph["conv4_4"] = self._conv2d_relu(graph["conv4_3"], 25, "conv4_4")
    graph["avgpool4"] = self._avgpool(graph["conv4_4"])
    graph["conv5_1"] = self._conv2d_relu(graph["avgpool4"], 28, "conv5_1")
    graph["conv5_2"] = self._conv2d_relu(graph["conv5_1"], 30, "conv5_2")
    graph["conv5_3"] = self._conv2d_relu(graph["conv5_2"], 32, "conv5_3")
    graph["conv5_4"] = self._conv2d_relu(graph["conv5_3"], 34, "conv5_4")
    graph["avgpool5"] = self._avgpool(graph["conv5_4"])

    return graph
