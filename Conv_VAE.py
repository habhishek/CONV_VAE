
# imports
from IPython import display
import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow.keras as tfk
import tensorflow.keras.layers as tfkl
import time


# Define the VAE Model Class
class VAE(tfk.Model):
    """
    Uses keras model subclassing to implement the VAE model consisting of the Encoder and Decoder sub networks.
    KL_Divergence term of the loss is added to the model via the add_loss() method
    A forward pass through the model is defined by the keras Functional API call()
    """
    def __init__(self, latent_dim, **kwargs):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = self.encoder_z()
        self.decoder = self.decoder_x()

    def encoder_z(self):
        layers = [tfkl.InputLayer(input_shape=[28, 28, 1]),
                  tfkl.Conv2D(
                      filters=32, kernel_size=3, strides=(2, 2), padding='valid', activation='relu'),
                  tfkl.Conv2D(
                      filters=64, kernel_size=3, strides=(2, 2), padding='valid', activation='relu'),
                  tfkl.Flatten(),
                  # 2 times the latent dim because we concatenate the means and logvar
                  # no activation
                  tfkl.Dense(self.latent_dim*2)]
        return tfk.Sequential(layers)

    def encode(self, x_input):
        mean, logvar = tf.split(self.encoder_z(x_input), num_or_size_splits=2, axis=1)
        eps = tf.random.normal(shape=mean.shape)
        z_sample = eps * tf.exp(logvar * .5) + mean
        return z_sample, mean, logvar

    def decoder_x(self):
        layers = [
            tfkl.InputLayer(input_shape=(self.latent_dim,)),
            # Expand the dimensions using Dense layer and prepare for de-convolution
            tfkl.Dense(7*7*32, activation='relu'),
            # Reshape the flattened representation into a 3D, so that de-convolution
            # can be successfully applied
            tfkl.Reshape(target_shape=(7, 7, 32)),
            tfkl.Conv2DTranspose(
                filters=64, kernel_size=3, strides=2, padding='same', activation='relu'),
            tfkl.Conv2DTranspose(
                filters=32, kernel_size=3, strides=2, padding='same', activation='relu'),
            # note that no activation is required in the final layer as we will
            # take care of that in the loss function.
            tfkl.Conv2DTranspose(filters=1, kernel_size=3, strides=1, padding='same')]
        return tfk.Sequential(layers)

    @tf.function
    def sample(self, eps=None):
        """Generation, after training - Sample from Gaussian Prior and use decoder
        to generate logits"""
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.decoder_x(eps)

    # Functional
    def call(self, x_input):
        """
        Forward pass through the encoder-decoder model.
        Calculates the KL_Divergence D_KL(q(z|x)||p(z)) analytically
        :param x_input: Input image of shape [28,28,1]
        :return: the logits of reconstructed image
        """
        z_sample, mean, logvar = self.encode(x_input)
        kl_divergence = tf.math.reduce_mean(- 0.5 *
                                            tf.math.reduce_sum(logvar - tf.math.square(mean) -
                                                               tf.exp(logvar) + 1, axis=1))
        x_logits = self.decoder_x(z_sample)
        # VAE is inherited from tfk.Model, thus have class method add_loss()
        self.add_loss(kl_divergence)
        return x_logits

