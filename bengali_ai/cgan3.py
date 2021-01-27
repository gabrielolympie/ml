from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D, Concatenate, Conv2DTranspose, Add
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam, SGD, Adadelta

import matplotlib.pyplot as plt

import numpy as np

class CGAN():
    def __init__(self):
        # Input shape
        self.img_rows = 84
        self.img_cols = 84
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.num_classes = 10
        self.num_classes_gr = 168
        self.num_classes_vd = 11
        self.num_classes_cd = 7
        self.latent_dim = 100

        optimizer = Adam(0.0002, 0.5)
#        optimizer = SGD(0.1)
#        optimizer = Adadelta()

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=['binary_crossentropy'],
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise and the target label as input
        # and generates the corresponding digit of that label
        noise = Input(shape=(self.latent_dim,))
#        label_gr = Input(shape=(1,))
#        label_vd = Input(shape=(1,))
#        label_cd = Input(shape=(1,))
        label_gr = Input(shape=self.img_shape)
        label_vd = Input(shape=self.img_shape)
        label_cd = Input(shape=self.img_shape)
        img = self.generator([noise, label_gr, label_vd, label_cd])

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated image as input and determines validity
        # and the label of that image
        valid = self.discriminator([img, label_gr, label_vd, label_cd])
        
        optimizer = Adam(0.0002, 0.5)
        # The combined model  (stacked generator and discriminator)
        # Trains generator to fool discriminator
        self.combined = Model([noise, label_gr, label_vd, label_cd], valid)
        self.combined.compile(loss=['binary_crossentropy'],
            optimizer=optimizer)

    def build_generator(self):
        
        input_noise = Input(shape=(self.latent_dim,))
        noise = Dense(128 * 11 * 11, activation="relu")(input_noise)
        noise = Reshape((11, 11, 128))(noise)
        
#        label_gr = Input(shape=(1,), dtype='int32')
#        label_embedding_gr = Flatten()(Embedding(self.num_classes_gr, self.num_classes_gr)(label_gr))
#        label_embedding_gr = Dense(11*11*1, activation="relu")(label_embedding_gr)
#        label_embedding_gr = Reshape((11,11,1))(label_embedding_gr)
#        
#        label_vd = Input(shape=(1,), dtype='int32')
#        label_embedding_vd = Flatten()(Embedding(self.num_classes_vd, self.num_classes_vd)(label_vd))
#        label_embedding_vd = Dense(11*11*32, activation="relu")(label_embedding_vd)
#        label_embedding_vd = Reshape((11,11,32))(label_embedding_vd)
#        
#        label_cd = Input(shape=(1,), dtype='int32')
#        label_embedding_cd = Flatten()(Embedding(self.num_classes_cd, self.num_classes_cd)(label_cd))
#        label_embedding_cd = Dense(11*11*32, activation="relu")(label_embedding_cd)
#        label_embedding_cd = Reshape((11,11,32))(label_embedding_cd)
        
        label_gr = Input(shape=self.img_shape)
        label_vd = Input(shape=self.img_shape)
        label_cd = Input(shape=self.img_shape)
        
        label_embedding_gr = Conv2D(128, kernel_size=3, strides=2, input_shape = (84,84,1), padding='same', activation = 'relu')(label_gr)
        label_embedding_gr = Conv2D(64, kernel_size=3, strides=2, padding='same', activation = 'relu')(label_embedding_gr)
        label_embedding_gr = Conv2D(32, kernel_size=3, strides=2, padding='same', activation = 'relu')(label_embedding_gr)
        label_embedding_gr = Conv2D(8, kernel_size=3, strides=1, padding='same', activation = 'relu')(label_embedding_gr)
        
        label_embedding_vd = Conv2D(128, kernel_size=3, strides=2, input_shape = (84,84,1), padding='same', activation = 'relu')(label_vd)
        label_embedding_vd = Conv2D(64, kernel_size=3, strides=2, padding='same', activation = 'relu')(label_embedding_vd)
        label_embedding_vd = Conv2D(32, kernel_size=3, strides=2, padding='same', activation = 'relu')(label_embedding_vd)
        label_embedding_vd = Conv2D(2, kernel_size=3, strides=1, padding='same', activation = 'relu')(label_embedding_vd)
        
        label_embedding_cd = Conv2D(128, kernel_size=3, strides=2, input_shape = (84,84,1), padding='same', activation = 'relu')(label_cd)
        label_embedding_cd = Conv2D(64, kernel_size=3, strides=2, padding='same', activation = 'relu')(label_embedding_cd)
        label_embedding_cd = Conv2D(32, kernel_size=3, strides=2, padding='same', activation = 'relu')(label_embedding_cd)
        label_embedding_cd = Conv2D(2, kernel_size=3, strides=1, padding='same', activation = 'relu')(label_embedding_cd)
             
        conc = Concatenate()([noise, label_embedding_gr, label_embedding_vd, label_embedding_cd])
        
        img = Conv2D(512, kernel_size=3, padding="same")(conc)
        img = BatchNormalization(momentum=0.8)(img)
        img = Activation("relu")(img)
        
        img = UpSampling2D()(img)
        img = Conv2D(256, kernel_size=3, padding="same")(img)
        img = BatchNormalization(momentum=0.8)(img)
        img = Activation("relu")(img)
        
        img = UpSampling2D()(img)
        img = Conv2D(128, kernel_size=3, padding="same")(img)
        img = BatchNormalization(momentum=0.8)(img)
        img = Activation("relu")(img)
        
        img = UpSampling2D()(img)
        img = Conv2D(64, kernel_size=3)(img)
        img = BatchNormalization(momentum=0.8)(img)
        img = Activation("relu")(img)
        
        img = Conv2D(self.channels, kernel_size=3)(img)
        img = Activation("tanh")(img)
        
#        def residual_block(layer_input, filters):
#            """Residual block described in paper"""
#            d = Conv2D(filters, kernel_size=3, strides=1, padding='same')(layer_input)
#            d = Activation('relu')(d)
#            d = BatchNormalization(momentum=0.8)(d)
#            d = Conv2D(filters, kernel_size=3, strides=1, padding='same')(d)
#            d = BatchNormalization(momentum=0.8)(d)
#            d = Add()([d, layer_input])
#            return d
#
#        def deconv2d(layer_input, padding = 'same'):
#            """Layers used during upsampling"""
#            u = UpSampling2D(size=2)(layer_input)
#            u = Conv2D(256, kernel_size=3, strides=1, padding=padding)(u)
#            u = Activation('relu')(u)
#            return u
#
#        # Low resolution image input
##        img_lr = Input(shape=(11,11,1))
#
#        # Pre-residual block
#        c1 = Conv2D(64, kernel_size=9, strides=1, padding='same')(conc)
#        c1 = Activation('relu')(c1)
#
#        # Propogate through residual blocks
#        r = residual_block(c1, 64)
#        for _ in range(16 - 1):
#            r = residual_block(r, 64)
#
#        # Post-residual block
#        c2 = Conv2D(64, kernel_size=3, strides=1, padding='same')(r)
#        c2 = BatchNormalization(momentum=0.8)(c2)
#        c2 = Add()([c2, c1])
#
#        # Upsampling
#        u1 = deconv2d(c2)
#        u2 = deconv2d(u1, padding = 'valid')
#        u2 = deconv2d(u2)
#
#        # Generate high resolution output
#        img = Conv2D(1, kernel_size=9, strides=1, padding='same', activation='tanh')(u2)
        
        
        
        model = Model([input_noise, label_gr, label_vd, label_cd], img)
        model.summary()
        
        return model

    def build_discriminator(self):
            
        img = Input(shape=self.img_shape)
        
#        label_gr = Input(shape=(1,), dtype='int32')
#        label_embedding_gr = Flatten()(Embedding(self.num_classes_gr, self.num_classes_gr)(label_gr))
#        label_embedding_gr = Dense(np.prod(self.img_shape), activation='relu')(label_embedding_gr)
#        label_embedding_gr = Reshape(self.img_shape)(label_embedding_gr)
#        
#        label_vd = Input(shape=(1,), dtype='int32')
#        label_embedding_vd = Flatten()(Embedding(self.num_classes_vd, self.num_classes_vd)(label_vd))
#        label_embedding_vd = Dense(np.prod(self.img_shape), activation='relu')(label_embedding_vd)
#        label_embedding_vd = Reshape(self.img_shape)(label_embedding_vd)
#        
#        label_cd = Input(shape=(1,), dtype='int32')
#        label_embedding_cd = Flatten()(Embedding(self.num_classes_cd, self.num_classes_cd)(label_cd))
#        label_embedding_cd = Dense(np.prod(self.img_shape), activation='relu')(label_embedding_cd)
#        label_embedding_cd = Reshape(self.img_shape)(label_embedding_cd)
        
        label_gr = Input(shape=self.img_shape)
        label_vd = Input(shape=self.img_shape)
        label_cd = Input(shape=self.img_shape)
        
        
        concat = Concatenate()([img, label_gr, label_vd, label_cd])
        
        concat = Conv2D(32, kernel_size=3, strides=2, padding="same")(concat)
        concat = LeakyReLU(alpha=0.2)(concat)
        concat = Dropout(0.25)(concat)
        
        concat = Conv2D(64, kernel_size=3, strides=2, padding="same")(concat)
        concat = ZeroPadding2D(padding=((0,1),(0,1)))(concat)
        concat = BatchNormalization(momentum=0.8)(concat)
        concat = LeakyReLU(alpha=0.2)(concat)
        concat = Dropout(0.25)(concat)
        
        concat = Conv2D(128, kernel_size=3, strides=2, padding="same")(concat)
        concat = BatchNormalization(momentum=0.8)(concat)
        concat = LeakyReLU(alpha=0.2)(concat)
        concat = Dropout(0.25)(concat)
        
        concat = Conv2D(256, kernel_size=3, strides=2, padding="same")(concat)
        concat = BatchNormalization(momentum=0.8)(concat)
        concat = LeakyReLU(alpha=0.2)(concat)
        concat = Dropout(0.25)(concat)
        
        concat = Conv2D(512, kernel_size=3, strides=2, padding="same")(concat)
        concat = BatchNormalization(momentum=0.8)(concat)
        concat = LeakyReLU(alpha=0.2)(concat)
        concat = Dropout(0.25)(concat)
        
        concat = Flatten()(concat)
        validity = Dense(1, activation='sigmoid')(concat)
        
#        def d_block(layer_input, filters, strides=1, bn=True):
#            """Discriminator layer"""
#            d = Conv2D(filters, kernel_size=3, strides=strides, padding='same')(layer_input)
#            d = LeakyReLU(alpha=0.2)(d)
#            if bn:
#                d = BatchNormalization(momentum=0.8)(d)
#            return d
#
#        # Input img
##        d0 = Input(shape=(84,84,1))
#
#        d1 = d_block(concat, 64, bn=False)
#        d2 = d_block(d1, 64, strides=2)
#        d3 = d_block(d2, 64*2)
#        d4 = d_block(d3, 64*2, strides=2)
#        d5 = d_block(d4, 64*4)
#        d6 = d_block(d5, 64*4, strides=2)
#        d7 = d_block(d6, 64*8)
#        d8 = d_block(d7, 64*8, strides=2)
#        
#        d8 = Flatten()(d8)
#        
#        d9 = Dense(64*16)(d8)
#        d10 = LeakyReLU(alpha=0.2)(d9)
#        validity = Dense(1, activation='sigmoid')(d10)
#        
        
#        import resnet
#        build = resnet.ResnetBuilder()
#        model_conv = build.build_resnet_18((4,84,84),1)
#        
#        validity = model_conv(concat)
        
        model = Model([img, label_gr, label_vd, label_cd], validity)
        model.summary()

        return model

    def train(self, epochs, X_train, ygr, yvd, ycd,dico_gr, dico_cd, dico_vd,  batch_size=128, sample_interval=50):

        # Load the dataset
#        (X_train, y_train), (_, _) = mnist.load_data()

        # Configure input
#        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        X_train = np.expand_dims(X_train, axis=3)
        
        ygr = np.expand_dims(ygr, axis=3)
        yvd = np.expand_dims(yvd, axis=3)
        ycd = np.expand_dims(ycd, axis=3)
        
    
#        ygr = ygr.reshape(-1, 1)
#        yvd = yvd.reshape(-1, 1)
#        ycd = ycd.reshape(-1, 1)

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half batch of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs, labels_gr, labels_vd, labels_cd = X_train[idx], ygr[idx], yvd[idx], ycd[idx]

            # Sample noise as generator input
            noise = np.random.normal(0, 1, (batch_size, 100))

            # Generate a half batch of new images
            gen_imgs = self.generator.predict([noise, labels_gr, labels_vd, labels_cd])

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch([imgs, labels_gr, labels_vd, labels_cd], valid)
            d_loss_fake = self.discriminator.train_on_batch([gen_imgs, labels_gr, labels_vd, labels_cd], fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------
            def build_image(X, dico_X):
                Y = list(np.zeros(X.shape[0]))
    
                for elt in range(len(X)):
                    Y[elt] = dico_X[X[elt]]
    
                Y = np.array(Y)
                return Y
        
            # Condition on labels
            sampled_labels_gr = np.random.randint(0, self.num_classes_gr, batch_size)
            sampled_labels_vd = np.random.randint(0, self.num_classes_vd, batch_size)
            sampled_labels_cd = np.random.randint(0, self.num_classes_cd, batch_size)
            
            sampled_labels_gr = build_image(sampled_labels_gr, dico_gr)
            sampled_labels_vd = build_image(sampled_labels_vd, dico_vd)
            sampled_labels_cd = build_image(sampled_labels_cd, dico_cd)
            
            sampled_labels_gr = np.expand_dims(sampled_labels_gr, axis=3)
            sampled_labels_vd = np.expand_dims(sampled_labels_vd, axis=3)
            sampled_labels_cd = np.expand_dims(sampled_labels_cd, axis=3)
            
            # Train the generator
            g_loss = self.combined.train_on_batch([noise, sampled_labels_gr, sampled_labels_vd, sampled_labels_cd], valid)

            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_images(epoch, ygr, yvd, ycd)

    def sample_images(self, epoch, ygr, yvd, ycd):
        r, c = 2, 5
        noise = np.random.normal(0, 1, (r * c, 100))
        sampled_labels_gr = ygr[:10] #np.arange(0, 6).reshape(-1, 1)
        sampled_labels_vd = yvd[:10] #np.arange(0, 6).reshape(-1, 1)
        sampled_labels_cd = ycd[:10] #np.arange(0, 6).reshape(-1, 1)

        gen_imgs = self.generator.predict([noise, sampled_labels_gr, sampled_labels_vd, sampled_labels_cd])

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5
#        print(np.array(gen_imgs).shape)
        fig, axs = plt.subplots(r, c)   
#        print(axs.shape)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt,:,:,0], cmap='gray')
#                axs[i,j].set_title("Digit: %d" % sampled_labels_gr[cnt])
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("images/%d.png" % epoch)
        plt.close()


if __name__ == '__main__':
    cgan = CGAN()
    cgan.train(epochs=20000, batch_size=32, sample_interval=200)
