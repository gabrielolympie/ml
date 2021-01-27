from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D, Concatenate, AveragePooling2D, Conv2DTranspose
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam, RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.constraints import Constraint
from keras.initializers import RandomNormal
import cv2
from custom_layer import *

import matplotlib.pyplot as plt

import sys

import numpy as np

class ClipConstraint(Constraint):
	# set clip value when initialized
	def __init__(self, clip_value):
		self.clip_value = clip_value
 
	# clip model weights to hypercube
	def __call__(self, weights):
		return backend.clip(weights, -self.clip_value, self.clip_value)
 
	# get the config
	def get_config(self):
		return {'clip_value': self.clip_value}


class DCGAN():
    def __init__(self, item_list = np.array([0])):
        # Input shape
        self.img_rows = 128
        self.img_cols = 128
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 256
        
        self.item_list = item_list
        
#        optimizer = Adam(0.0002, 0.5)
#        optimizer = Adam(5e-4)
        optimizer = RMSprop(0.00005)
        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        
        loss = 'binary_crossentropy'
#        loss = wasserstein_loss
        
        self.discriminator.compile(loss=loss,
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generates imgs
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        valid = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, valid)
#        optimizer = Adam(0.0001, 0.5)
#        optimizer = Adam(1e-3)
        self.combined.compile(loss=loss, optimizer=optimizer)

#    def build_generator(self):
#        
#        
#        noise = Input(shape=(self.latent_dim,))
#        
#        base_4 = Dense(self.latent_dim * 4 * 4, activation="relu", input_dim=self.latent_dim)(noise)
#        base_4 = Reshape(( 4, 4, self.latent_dim))(base_4)
#            
#        base_4 = Conv2DTranspose(512, kernel_size = 4, padding="same")(base_4)
#        base_4 = LeakyReLU(alpha=0.2)(base_4)
#        
#        base_4 = Conv2DTranspose(256, kernel_size=3, padding="same")(base_4)
#        base_4 = PixelNormalization()(base_4)
#        base_4 = LeakyReLU(alpha=0.2)(base_4)
##        base_4 = BatchNormalization(momentum=0.8)(base_4)
##        base_4 = Conv2D(256, kernel_size=3, padding="same")(base_4)
##        base_4 = PixelNormalization()(base_4)
##        base_4 = LeakyReLU(alpha=0.2)(base_4)
###        base_4 = BatchNormalization(momentum=0.8)(base_4)
##        base_4 = Conv2D(256, kernel_size=3, padding="same")(base_4)
##        base_4 = PixelNormalization()(base_4)
##        base_4 = LeakyReLU(alpha=0.2)(base_4)
#        
#        
#        base_8 = UpSampling2D()(base_4)
#        base_8 = Conv2DTranspose(512, kernel_size=3, padding="same")(base_8)
#        base_8 = PixelNormalization()(base_8)
#        base_8 = LeakyReLU(alpha=0.2)(base_8)
##        base_8 = BatchNormalization(momentum=0.8)(base_8)
#        base_8 = Conv2DTranspose(512, kernel_size=3, padding="same")(base_8)
#        base_8 = PixelNormalization()(base_8)
#        base_8 = LeakyReLU(alpha=0.2)(base_8)
##        base_8 = BatchNormalization(momentum=0.8)(base_8)
#        
#        
#        base_16 = UpSampling2D()(base_8)
#        base_16 = Conv2DTranspose(128, kernel_size=3, padding="same")(base_16)
#        base_16 = PixelNormalization()(base_16)
#        base_16 = LeakyReLU(alpha=0.2)(base_16)
##        base_16 = BatchNormalization(momentum=0.8)(base_16)
#        base_16 = Conv2DTranspose(128, kernel_size=3, padding="same")(base_16)
#        base_16 = PixelNormalization()(base_16)
#        base_16 = LeakyReLU(alpha=0.2)(base_16)
##        base_16 = BatchNormalization(momentum=0.8)(base_16)
#        
#        
#        base_32 = UpSampling2D()(base_16)
#        base_32 = Conv2DTranspose(64, kernel_size=3, padding="same")(base_32)
#        base_32 = PixelNormalization()(base_32)
#        base_32 = LeakyReLU(alpha=0.2)(base_32)
##        base_32 = BatchNormalization(momentum=0.8)(base_32)
#        base_32 = Conv2DTranspose(64, kernel_size=3, padding="same")(base_32)
#        base_32 = PixelNormalization()(base_32)
#        base_32 = LeakyReLU(alpha=0.2)(base_32)
##        base_32 = BatchNormalization(momentum=0.8)(base_32)
#        
#        
#        base_64 = UpSampling2D()(base_32)
#        base_64 = Conv2DTranspose(32, kernel_size=3, padding="same")(base_64)
#        base_64 = PixelNormalization()(base_64)
#        base_64 = LeakyReLU(alpha=0.2)(base_64)
##        base_64 = BatchNormalization(momentum=0.8)(base_64)
#        base_64 = Conv2DTranspose(32, kernel_size=3, padding="same")(base_64)
#        base_64 = PixelNormalization()(base_64)
#        base_64 = LeakyReLU(alpha=0.2)(base_64)
##        base_64 = BatchNormalization(momentum=0.8)(base_64)  
#        
#        
#        base_128 = UpSampling2D()(base_64)
#        base_128 = Conv2DTranspose(16, kernel_size=3, padding="same")(base_128)
#        base_128 = PixelNormalization()(base_128)
#        base_128 = LeakyReLU(alpha=0.2)(base_128)
##        base_128 = BatchNormalization(momentum=0.8)(base_128)
#        base_128 = Conv2DTranspose(16, kernel_size=3, padding="same")(base_128)
#        base_128 = PixelNormalization()(base_128)
#        base_128 = LeakyReLU(alpha=0.2)(base_128)
##        base_128 = BatchNormalization(momentum=0.8)(base_128)
#        
#        
#        out_4 =  Conv2DTranspose(3, kernel_size=3, padding="same", activation = 'tanh')(base_4)
##        out_4 = LeakyReLU(alpha=0.2)(out_4)
#        out_8 =  Conv2DTranspose(3, kernel_size=3, padding="same", activation = 'tanh')(base_8)
##        out_8 = LeakyReLU(alpha=0.2)(out_8)
#        out_16 =  Conv2DTranspose(3, kernel_size=3, padding="same", activation = 'tanh')(base_16)
##        out_16 = LeakyReLU(alpha=0.2)(out_16)
#        out_32 =  Conv2DTranspose(3, kernel_size=3, padding="same", activation = 'tanh')(base_32)
##        out_32 = LeakyReLU(alpha=0.2)(out_32)
#        out_64 =  Conv2DTranspose(3, kernel_size=3, padding="same", activation = 'tanh')(base_64)
##        out_64 = LeakyReLU(alpha=0.2)(out_64)
#        out_128 =  Conv2DTranspose(3, kernel_size=3, padding="same", activation = 'tanh')(base_128)
##        out_128 = LeakyReLU(alpha=0.2)(out_128)
#        
#        out = [out_4, out_8,out_16,out_32,out_64,out_128]
#
#        model = Model(noise, out)
#        
#        model.summary()
#        
#        return model
    
    def build_generator(self):
        outputs = []

        z_in = Input(shape=(self.latent_dim,))
        x = Dense(8*8*256)(z_in)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Reshape((8, 8, 256))(x)

        for i in range(5):
            if i == 0:
                x = Conv2DTranspose(128, (5, 5), strides=(1, 1),
                    padding='same')(x)
                x = BatchNormalization()(x)
                x = LeakyReLU()(x)
            else:
                x = Conv2DTranspose(128, (5, 5), strides=(2, 2),
                    padding='same')(x)
                x = BatchNormalization()(x)
                x = LeakyReLU()(x)

            x = Conv2D(128, (5, 5), strides=(1, 1), padding='same')(x)
            x = BatchNormalization()(x)
            x = LeakyReLU()(x)

            outputs.append(Conv2DTranspose(3, (5, 5), strides=(1, 1),
                padding='same', activation='tanh')(x))

        model = Model(inputs=z_in, outputs=outputs)
        model.summary()
        return model


#    def build_discriminator(self):
#        
#        
#        in_4 = Input(shape=(4,4,3))
#        in_8 = Input(shape=(8,8,3))
#        in_16 = Input(shape=(16,16,3))
#        in_32 = Input(shape=(32,32,3))
#        in_64 = Input(shape=(64,64,3))
#        in_128 = Input(shape=(128,128,3))
#        
#        inputs = [in_4, in_8, in_16,in_32,in_64,in_128]
#        
#        inp_4 = Conv2D(3, kernel_size=1, padding="same")(in_4)
#        inp_4 = LeakyReLU(alpha=0.2)(inp_4)
#        inp_8 = Conv2D(3, kernel_size=1, padding="same")(in_8)
#        inp_8 = LeakyReLU(alpha=0.2)(inp_8)
#        inp_16 = Conv2D(3, kernel_size=1, padding="same")(in_16)
#        inp_16 = LeakyReLU(alpha=0.2)(inp_16)
#        inp_32 = Conv2D(3, kernel_size=1, padding="same")(in_32)
#        inp_32 = LeakyReLU(alpha=0.2)(inp_32)
#        inp_64 = Conv2D(3, kernel_size=1, padding="same")(in_64)
#        inp_64 = LeakyReLU(alpha=0.2)(inp_64)
#        inp_128 = Conv2D(3, kernel_size=1, padding="same")(in_128)
#        inp_128 = LeakyReLU(alpha=0.2)(inp_128)
##        
##        inp_4 = in_4
##        inp_8 = in_8
##        inp_16 = in_16
##        inp_32 = in_32
##        inp_64 = in_64
##        inp_128 = in_128
#        
#        const = ClipConstraint(0.01)
#        init = RandomNormal(stddev=0.02)
#        
#        base_128 = MinibatchStdev()(inp_128)
#        base_128 = Conv2D(16, kernel_size=3, padding="same", kernel_initializer=init, kernel_constraint=const)(base_128)
#        base_128 = LeakyReLU(alpha=0.2)(base_128)
##        base_128 = BatchNormalization()(base_128)
#        base_128 = Conv2D(32, kernel_size=3, padding="same", kernel_initializer=init, kernel_constraint=const)(base_128)
#        base_128 = LeakyReLU(alpha=0.2)(base_128)
#        base_128 = BatchNormalization()(base_128)
#        base_128 = AveragePooling2D(pool_size=(2, 2))(base_128)
##        base_128 = Dropout(0.25)(base_128)
#        
#        base_64 = Concatenate()([base_128, inp_64])
#        base_64 = MinibatchStdev()(base_64)
##        base_64 = BatchNormalization()(base_64)
#        base_64 = Conv2D(32, kernel_size=3, padding="same", kernel_initializer=init, kernel_constraint=const)(base_64)
#        base_64 = LeakyReLU(alpha=0.2)(base_64)
##        base_64 = BatchNormalization()(base_64)
#        base_64 = Conv2D(64, kernel_size=3, padding="same", kernel_initializer=init, kernel_constraint=const)(base_64)
#        base_64 = LeakyReLU(alpha=0.2)(base_64)
#        base_64 = BatchNormalization()(base_64)
#        base_64 = AveragePooling2D(pool_size=(2, 2))(base_64)
##        base_64 = Dropout(0.25)(base_64)
#        
#        base_32 = Concatenate()([base_64, inp_32])
#        base_32 = MinibatchStdev()(base_32)
#        base_32 = Conv2D(64, kernel_size=3, padding="same", kernel_initializer=init, kernel_constraint=const)(base_32)
#        base_32 = LeakyReLU(alpha=0.2)(base_32)
##        base_32 = BatchNormalization()(base_32)
#        base_32 = Conv2D(128, kernel_size=3, padding="same", kernel_initializer=init, kernel_constraint=const)(base_32)
#        base_32 = LeakyReLU(alpha=0.2)(base_32)
#        base_32 = BatchNormalization()(base_32)
#        base_32 = AveragePooling2D(pool_size=(2, 2))(base_32)
#        
##        base_32 = Dropout(0.25)(base_32)
#        
#        base_16 = Concatenate()([base_32, inp_16])
#        base_16 = MinibatchStdev()(base_16)
#        base_16 = Conv2D(128, kernel_size=3, padding="same", kernel_initializer=init, kernel_constraint=const)(base_16)
#        base_16 = LeakyReLU(alpha=0.2)(base_16)
##        base_16 = BatchNormalization()(base_16)
#        base_16 = Conv2D(128, kernel_size=3, padding="same", kernel_initializer=init, kernel_constraint=const)(base_16)
#        base_16 = LeakyReLU(alpha=0.2)(base_16)
#        base_16 = BatchNormalization()(base_16)
#        base_16 = AveragePooling2D(pool_size=(2, 2))(base_16)
##        base_16 = Dropout(0.25)(base_16)
#        
#        base_8 = Concatenate()([base_16, inp_8])
#        base_8 = MinibatchStdev()(base_8)
#        base_8 = Conv2D(256, kernel_size=3, padding="same", kernel_initializer=init, kernel_constraint=const)(base_8)
#        base_8 = LeakyReLU(alpha=0.2)(base_8)
##        base_8 = BatchNormalization()(base_8)
#        base_8 = Conv2D(256, kernel_size=3, padding="same", kernel_initializer=init, kernel_constraint=const)(base_8)
#        base_8 = LeakyReLU(alpha=0.2)(base_8)
#        base_8 = BatchNormalization()(base_8)
#        base_8 = AveragePooling2D(pool_size=(2, 2))(base_8)
##        base_8 = Dropout(0.25)(base_8)
#        
#        base_4 = Concatenate()([base_8, inp_4])
#        base_4 = MinibatchStdev()(base_4)
#        base_4 = Conv2D(512, kernel_size=3, strides=1, padding="same", kernel_initializer=init, kernel_constraint=const)(base_4)
#        base_4 = LeakyReLU(alpha=0.2)(base_4)
##        base_4 = BatchNormalization()(base_4)
#        base_4 = Conv2D(512, kernel_size=4, strides=1, padding="same", kernel_initializer=init, kernel_constraint=const)(base_4)
#        base_4 = LeakyReLU(alpha=0.2)(base_4)
##        base_4 = BatchNormalization()(base_4)
#        
##        base_4 = Dropout(0.25)(base_4)
#        
#        base = Flatten()(base_4)
#        validity = Dense(1, activation='sigmoid')(base)
#
#        img = Input(shape=self.img_shape)
#        
#        model = Model(inputs, validity)
#        model.summary()
#        
#        return model
#    
    
#    def get_batch(self,batch_size):
#        import os
#        import imageio
#        import random
#        
#        if self.item_list.sum() == 0:
#            imgs = os.listdir('./image_clean_64')
#        else:
#            imgs = self.item_list
#        
#        sizes = [4,8,16,32,64,128]
#        
#        vect = [list(np.zeros(batch_size)) for i in range(len(sizes))]
#        
#        for i in range(batch_size):
#            r = random.randint(0,len(imgs)-1)
#            for j in range(len(sizes)):
#                path = './image_clean_'+str(sizes[j])+'/'+imgs[r]
#                img = imageio.imread(path)/127.5 - 1
#                vect[j][i] = np.array(img).astype('float32')
#        
#        for i in range(len(vect)):
#            vect[i] = np.array(vect[i])
#            
##        vect = np.array(vect)
#        return vect
    def build_discriminator(self):
    # we have multiple inputs to make a real/fake decision from
        inputs = [
            Input(shape=(128, 128, 3)),
            Input(shape=(64, 64, 3)),
            Input(shape=(32, 32, 3)),
            Input(shape=(16, 16, 3)),
            Input(shape=(8, 8, 3)),
        ]

        x = None
        for image_in in inputs:
            if x is None:
                # for the first input we don't have features to append to
                x = Conv2D(64, (5, 5), strides=(2, 2),
                    padding='same')(image_in)
                x = LeakyReLU()(x)
                x = Dropout(0.3)(x)
            else:
                # every additional input gets its own conv layer then appended
                y = Conv2D(64, (5, 5), strides=(2, 2),
                    padding='same')(image_in)
                y = LeakyReLU()(y)
                y = Dropout(0.3)(y)
                x = Concatenate()([x, y])

            x = Conv2D(128, (3, 3), strides=(1, 1), padding='same')(x)
            x = LeakyReLU()(x)
            x = Dropout(0.3)(x)

            x = Conv2D(64, (5, 5), strides=(2, 2), padding='same')(x)
            x = LeakyReLU()(x)
            x = Dropout(0.3)(x)

        x = Flatten()(x)
        out = Dense(1, activation = 'sigmoid')(x)
        inputs = inputs[::-1] # reorder the list to be smallest resolution first
        model = Model(inputs=inputs, outputs=out)
        model.summary()
        return model
  
        
    def get_batch(self,batch_size):
        import os
        import imageio
        import random
        
        if self.item_list.sum() == 0:
            imgs = os.listdir('./celeba_128')
        else:
            imgs = self.item_list
        
        sizes = [8,16,32,64,128]
#        sizes = [128,64,32,16,8]
        
        vect = [list(np.zeros(batch_size)) for i in range(len(sizes))]
        
        for i in range(batch_size):
            r = random.randint(0,len(imgs)-1)
            
            path = './celeba_128/'+imgs[r]
            img = imageio.imread(path)/127.5 - 1
            
            for j in range(len(sizes)):
                img1 = cv2.resize(img, (sizes[j],sizes[j]))
                vect[j][i] = np.array(img1).astype('float32')
        
        for i in range(len(vect)):
            vect[i] = np.array(vect[i])
            
#        vect = np.array(vect)
        return vect
    
    def train(self, epochs, batch_size=128, save_interval=50):

        # Load the dataset
#        (X_train, _), (_, _) = mnist.load_data()
#
#        # Rescale -1 to 1
#        X_train = X_train / 127.5 - 1.
#        X_train = np.expand_dims(X_train, axis=3)
        
        X_train = self.get_batch(batch_size)
                
        
        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
#        fake = np.zeros((batch_size, 1))
        fake = np.zeros((batch_size, 1)) 
    
#        valid = np.random.uniform(0.90,1,(batch_size,1))
#        fake = np.random.uniform(0,0.1,(batch_size,1))
        
        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half of images
            idx = np.random.randint(0, len(X_train[0]), batch_size)
#            imgs = X_train[:,idx]
            
            imgs = [elt[idx] for elt in X_train]
            
            # Sample noise and generate a batch of new images
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            gen_imgs = self.generator.predict(noise)

            # Train the discriminator (real classified as ones and generated as zeros)
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Train the generator (wants discriminator to mistake images as real)
#            for _ in range(3):
#                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            g_loss = self.combined.train_on_batch(noise, valid)
            
            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                self.save_imgs(epoch)

    def save_imgs(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
#        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[j][cnt, :,:,:]*0.5+0.5)
                axs[i,j].axis('off')
            cnt += 1
            
        fig.savefig("images_gen/mnist_%d.png" % epoch)
        plt.close()
        
        return gen_imgs


if __name__ == '__main__':
    dcgan = DCGAN()
    dcgan.train(epochs=4000, batch_size=32, save_interval=50)
