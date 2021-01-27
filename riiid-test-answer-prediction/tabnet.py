import tensorflow as tf
from tensorflow_addons.activations import sparsemax
from typing import List, Tuple

## Utils
def glu(x, n_units=None):
    """Generalized linear unit nonlinear activation."""
    return x[:, :n_units] * tf.nn.sigmoid(x[:, n_units:])

## Batch_normalization
class GhostBatchNormalization(tf.keras.layers.Layer):
    def __init__(
        self, virtual_divider: int = 1, momentum: float = 0.9, epsilon: float = 1e-5
    ):
        super(GhostBatchNormalization, self).__init__()
        self.virtual_divider = virtual_divider
        self.bn = BatchNormInferenceWeighting(momentum=momentum)

    def call(self, x, training: bool = None, alpha: float = 0.0):
        if training:
            chunks = tf.split(x, self.virtual_divider)
            x = [self.bn(x, training=True) for x in chunks]
            return tf.concat(x, 0)
        return self.bn(x, training=False, alpha=alpha)

#    @property
#    def moving_mean(self):
#        return self.bn.moving_mean
#
#    @property
#    def moving_variance(self):
#        return self.bn.moving_variance


class BatchNormInferenceWeighting(tf.keras.layers.Layer):
    def __init__(self, momentum: float = 0.9, epsilon: float = None):
        super(BatchNormInferenceWeighting, self).__init__()
        self.momentum = momentum
        self.epsilon = tf.keras.backend.epsilon() if epsilon is None else epsilon

    def build(self, input_shape):
        channels = input_shape[-1]

        self.gamma = tf.Variable(
            initial_value=tf.ones((channels,), tf.float32), trainable=True,
        )
        self.beta = tf.Variable(
            initial_value=tf.zeros((channels,), tf.float32), trainable=True,
        )

        self.moving_mean = tf.Variable(
            initial_value=tf.zeros((channels,), tf.float32), trainable=False,
        )
        self.moving_mean_of_squares = tf.Variable(
            initial_value=tf.zeros((channels,), tf.float32), trainable=False,
        )

#    def __update_moving(self, var, value):
#        var.assign(var * self.momentum + (1 - self.momentum) * value)

#    def __apply_normalization(self, x, mean, variance):
#        return self.gamma * (x - mean) / tf.sqrt(variance + self.epsilon) + self.beta

    def call(self, x, training: bool = None, alpha: float = 0.0):
        mean = tf.reduce_mean(x, axis=0)
        mean_of_squares = tf.reduce_mean(tf.pow(x, 2), axis=0)
        if training:
            # update moving stats
#            self.__update_moving(self.moving_mean, mean)
            self.moving_mean.assign(self.moving_mean * self.momentum + (1 - self.momentum) * mean)
#            self.__update_moving(mean, mean_of_squares)
            self.moving_mean_of_squares.assign(self.moving_mean_of_squares * self.momentum + (1 - self.momentum) * mean_of_squares)

            variance = mean_of_squares - tf.pow(mean, 2)
        
            x = self.gamma * (x - mean) / tf.sqrt(variance + self.epsilon) + self.beta
        else:
            mean = alpha * mean + (1 - alpha) * self.moving_mean
            variance = (
                alpha * mean_of_squares + (1 - alpha) * self.moving_mean_of_squares
            ) - tf.pow(mean, 2)
            x = self.gamma * (x - mean) / tf.sqrt(variance + self.epsilon) + self.beta

        return x

## Features Transformations
class FeatureBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        feature_dim: int,
        apply_glu: bool = True,
        bn_momentum: float = 0.9,
        bn_virtual_divider: int = 32,
        fc: tf.keras.layers.Layer = None,
        epsilon: float = 1e-5,
    ):
        super(FeatureBlock, self).__init__()
        self.apply_gpu = apply_glu
        self.feature_dim = feature_dim
        units = feature_dim * 2 if apply_glu else feature_dim

        self.fc = tf.keras.layers.Dense(units, use_bias=False) if fc is None else fc
        self.bn = GhostBatchNormalization(
            virtual_divider=bn_virtual_divider, momentum=bn_momentum
        )

    def call(self, x, training: bool = None, alpha: float = 0.0):
        x = self.fc(x)
        x = self.bn(x, training=training, alpha=alpha)
        if self.apply_gpu:
            return glu(x, self.feature_dim)
        return x


class AttentiveTransformer(tf.keras.layers.Layer):
    def __init__(self, feature_dim: int, bn_momentum: float, bn_virtual_divider: int):
        super(AttentiveTransformer, self).__init__()
        self.block = FeatureBlock(
            feature_dim,
            bn_momentum=bn_momentum,
            bn_virtual_divider=bn_virtual_divider,
            apply_glu=False,
        )

    def call(self, x, prior_scales, training=None, alpha: float = 0.0):
        x = self.block(x, training=training, alpha=alpha)
        return sparsemax(x * prior_scales)


class FeatureTransformer(tf.keras.layers.Layer):
    def __init__(
        self,
        feature_dim: int,
        fcs = [],
        n_total: int = 4,
        n_shared: int = 2,
        bn_momentum: float = 0.9,
        bn_virtual_divider: int = 1,
    ):
        super(FeatureTransformer, self).__init__()
        self.n_total, self.n_shared = n_total, n_shared

        kargs = {
            "feature_dim": feature_dim,
            "bn_momentum": bn_momentum,
            "bn_virtual_divider": bn_virtual_divider,
        }

        # build blocks
        self.blocks: List[FeatureBlock] = []
        for n in range(n_total):
            # some shared blocks
            if fcs and n < len(fcs):
                self.blocks.append(FeatureBlock(**kargs, fc=fcs[n]))
            # build new blocks
            else:
                self.blocks.append(FeatureBlock(**kargs))

    def call(
        self, x: tf.Tensor, training: bool = None, alpha: float = 0.0
    ) -> tf.Tensor:
        x = self.blocks[0](x, training=training, alpha=alpha)
        for n in range(1, self.n_total):
            x = x * tf.sqrt(0.5) + self.blocks[n](x, training=training, alpha=alpha)
        return x

    @property
    def shared_fcs(self):
        return [self.blocks[i].fc for i in range(self.n_shared)]

## Tabnet
class TabNet(tf.keras.layers.Layer):
    def __init__(
        self,
        num_features: int,
        feature_dim: int,
        output_dim: int,
        feature_columns = None,
        n_step: int = 1,
        n_total: int = 4,
        n_shared: int = 2,
        relaxation_factor: float = 1.5,
        bn_epsilon: float = 1e-5,
        bn_momentum: float = 0.7,
        bn_virtual_divider: int = 1,
    ):
        """TabNet
        Will output a vector of size output_dim.
        Args:
            num_features (int): Number of features.
            feature_dim (int): Embedding feature dimention to use.
            output_dim (int): Output dimension.
            feature_columns (List, optional): If defined will add a DenseFeatures layer first. Defaults to None.
            n_step (int, optional): Total number of steps. Defaults to 1.
            n_total (int, optional): Total number of feature transformer blocks. Defaults to 4.
            n_shared (int, optional): Number of shared feature transformer blocks. Defaults to 2.
            relaxation_factor (float, optional): >1 will allow features to be used more than once. Defaults to 1.5.
            bn_epsilon (float, optional): Batch normalization, epsilon. Defaults to 1e-5.
            bn_momentum (float, optional): Batch normalization, momentum. Defaults to 0.7.
            bn_virtual_divider (int, optional): Batch normalization. Full batch will be divided by this.
        """
        super(TabNet, self).__init__()
        self.output_dim, self.num_features = output_dim, num_features
        self.n_step, self.relaxation_factor = n_step, relaxation_factor
        self.feature_columns = feature_columns

        if feature_columns is not None:
            self.input_features = tf.keras.layers.DenseFeatures(feature_columns)

        # ? Switch to Ghost Batch Normalization
        self.bn = tf.keras.layers.BatchNormalization(
            momentum=bn_momentum, epsilon=bn_epsilon
        )

        kargs = {
            "feature_dim": feature_dim + output_dim,
            "n_total": n_total,
            "n_shared": n_shared,
            "bn_momentum": bn_momentum,
            "bn_virtual_divider": bn_virtual_divider,
        }

        # first feature transformer block is built first to get the shared blocks
        self.feature_transforms: List[FeatureTransformer] = [
            FeatureTransformer(**kargs)
        ]
        self.attentive_transforms: List[AttentiveTransformer] = []
        for i in range(n_step):
            self.feature_transforms.append(
                FeatureTransformer(**kargs, fcs=self.feature_transforms[0].shared_fcs)
            )
            self.attentive_transforms.append(
                AttentiveTransformer(num_features, bn_momentum, bn_virtual_divider)
            )

    def call(
        self, features: tf.Tensor, training: bool = None, alpha: float = 0.0
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        if self.feature_columns is not None:
            features = self.input_features(features)

        bs = tf.shape(features)[0]
        out_agg = tf.zeros((bs, self.output_dim))
        prior_scales = tf.ones((bs, self.num_features))
        masks = []

        features = self.bn(features, training=training)
        masked_features = features

        total_entropy = 0.0

        for step_i in range(self.n_step + 1):
            x = self.feature_transforms[step_i](
                masked_features, training=training, alpha=alpha
            )

            if step_i > 0:
                out = tf.keras.activations.relu(x[:, : self.output_dim])
                out_agg += out

            # no need to build the features mask for the last step
            if step_i < self.n_step:
                x_for_mask = x[:, self.output_dim :]

                mask_values = self.attentive_transforms[step_i](
                    x_for_mask, prior_scales, training=training, alpha=alpha
                )

                # relaxation factor of 1 forces the feature to be only used once.
                prior_scales *= self.relaxation_factor - mask_values

                masked_features = tf.multiply(mask_values, features)

                # entropy is used to penalize the amount of sparsity in feature selection
                total_entropy = tf.reduce_mean(
                    tf.reduce_sum(
                        tf.multiply(mask_values, tf.math.log(mask_values + 1e-15)),
                        axis=1,
                    )
                )

                masks.append(tf.expand_dims(tf.expand_dims(mask_values, 0), 3))

        loss = total_entropy / self.n_step
        
#        self.add_loss(loss)
        
        return out_agg, masks