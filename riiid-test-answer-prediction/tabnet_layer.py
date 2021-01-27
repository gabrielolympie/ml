import tensorflow as tf
import numpy as np

def glu(act, n_units):
  """Generalized linear unit nonlinear activation."""
  return act[:, :n_units] * tf.keras.activations.sigmoid(act[:, n_units:])

@tf.keras.utils.register_keras_serializable(package="Addons")
def sparsemax(logits, axis: int = -1) -> tf.Tensor:
    """Sparsemax activation function [1].
    For each batch `i` and class `j` we have
      $$sparsemax[i, j] = max(logits[i, j] - tau(logits[i, :]), 0)$$
    [1]: https://arxiv.org/abs/1602.02068
    Args:
        logits: Input tensor.
        axis: Integer, axis along which the sparsemax operation is applied.
    Returns:
        Tensor, output of sparsemax transformation. Has the same type and
        shape as `logits`.
    Raises:
        ValueError: In case `dim(logits) == 1`.
    """
    logits = tf.convert_to_tensor(logits, name="logits")
    logits = tf.cast(logits, tf.float32)
    # We need its original shape for shape inference.
    shape = logits.get_shape()
    rank = shape.rank
    is_last_axis = (axis == -1) or (axis == rank - 1)

    if is_last_axis:
        output = _compute_2d_sparsemax(logits)
        output.set_shape(shape)
        return output
    
    rank_op = tf.rank(logits)
    axis_norm = axis % rank
    logits = _swap_axis(logits, axis_norm, tf.math.subtract(rank_op, 1))

    output = _compute_2d_sparsemax(logits)
    output = _swap_axis(output, axis_norm, tf.math.subtract(rank_op, 1))

    output.set_shape(shape)
    return output


def _swap_axis(logits, dim_index, last_index, **kwargs):
    return tf.transpose(
        logits,
        tf.concat(
            [
                tf.range(dim_index),
                [last_index],
                tf.range(dim_index + 1, last_index),
                [dim_index],
            ],
            0,
        ),
        **kwargs,
    )


def _compute_2d_sparsemax(logits):
    """Performs the sparsemax operation when axis=-1."""
    shape_op = tf.shape(logits)
    obs = tf.math.reduce_prod(shape_op[:-1])
    dims = shape_op[-1]
    z = tf.reshape(logits, [obs, dims])
    z_sorted, _ = tf.nn.top_k(z, k=dims)
    z_cumsum = tf.math.cumsum(z_sorted, axis=-1)
    k = tf.range(1, tf.cast(dims, logits.dtype) + 1, dtype=logits.dtype)
    z_check = 1 + k * z_sorted > z_cumsum
    k_z = tf.math.reduce_sum(tf.cast(z_check, tf.int32), axis=-1)
    k_z_safe = tf.math.maximum(k_z, 1)
    indices = tf.stack([tf.range(0, obs), tf.reshape(k_z_safe, [-1]) - 1], axis=1)
    tau_sum = tf.gather_nd(z_cumsum, indices)
    tau_z = (tau_sum - 1) / tf.cast(k_z, logits.dtype)
    p = tf.math.maximum(tf.cast(0, logits.dtype), z - tf.expand_dims(tau_z, -1))
    p_safe = tf.where(
        tf.expand_dims(
            tf.math.logical_or(tf.math.equal(k_z, 0), tf.math.is_nan(z_cumsum[:, -1])),
            axis=-1,
        ),
        tf.fill([obs, dims], tf.cast(float("nan"), logits.dtype)),
        p,
    )
    # Reshape back to original size
    p_safe = tf.reshape(p_safe, shape_op)
    return p_safe

@tf.keras.utils.register_keras_serializable(package="Addons")
class Sparsemax(tf.keras.layers.Layer):
    """Sparsemax activation function.
    The output shape is the same as the input shape.
    See [From Softmax to Sparsemax: A Sparse Model of Attention and Multi-Label Classification](https://arxiv.org/abs/1602.02068).
    Arguments:
        axis: Integer, axis along which the sparsemax normalization is applied.
    """
    def __init__(self, axis: int = -1, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.axis = axis

    def call(self, inputs):
        return sparsemax(inputs, axis=self.axis)

    def get_config(self):
        config = {"axis": self.axis}
        base_config = super().get_config()
        return {**base_config, **config}

    def compute_output_shape(self, input_shape):
        return input_shape

class TabNet(tf.keras.layers.Layer):
    def __init__(self,
#           columns,
           num_features,
           feature_dim,
           output_dim,
           num_decision_steps,
           relaxation_factor,
           batch_momentum,
           virtual_batch_size,
#           num_classes,
           epsilon=0.00001):
        """Initializes a TabNet instance.
        Args:
          columns: The Tensorflow column names for the dataset.
          num_features: The number of input features (i.e the number of columns for
            tabular data assuming each feature is represented with 1 dimension).
          feature_dim: Dimensionality of the hidden representation in feature
            transformation block. Each layer first maps the representation to a
            2*feature_dim-dimensional output and half of it is used to determine the
            nonlinearity of the GLU activation where the other half is used as an
            input to GLU, and eventually feature_dim-dimensional output is
            transferred to the next layer.
          output_dim: Dimensionality of the outputs of each decision step, which is
            later mapped to the final classification or regression output.
          num_decision_steps: Number of sequential decision steps.
          relaxation_factor: Relaxation factor that promotes the reuse of each
            feature at different decision steps. When it is 1, a feature is enforced
            to be used only at one decision step and as it increases, more
            flexibility is provided to use a feature at multiple decision steps.
          batch_momentum: Momentum in ghost batch normalization.
          virtual_batch_size: Virtual batch size in ghost batch normalization. The
            overall batch size should be an integer multiple of virtual_batch_size.
          num_classes: Number of output classes.
          epsilon: A small number for numerical stability of the entropy calcations.

        Returns:
          A TabNet instance.
        """
        super(TabNet, self).__init__()
        assert feature_dim > output_dim, 'feature dim must be strictly superior to output dim'
        self.num_features = num_features
        self.feature_dim = feature_dim
        self.output_dim = output_dim
        self.num_decision_steps = num_decision_steps
        self.relaxation_factor = relaxation_factor
        self.batch_momentum = batch_momentum
        self.virtual_batch_size = virtual_batch_size
        #        self.num_classes = num_classes
        self.epsilon = epsilon

        self.tr1 = tf.keras.layers.Dense(self.feature_dim * 2, use_bias=False)
        self.tr2 = tf.keras.layers.Dense(self.feature_dim * 2, use_bias=False)
        self.tr3 = [tf.keras.layers.Dense(self.feature_dim * 2, use_bias=False) for _ in range(num_decision_steps)]
        self.tr4 = [tf.keras.layers.Dense(self.feature_dim * 2, use_bias=False) for _ in range(num_decision_steps)]
        self.mask = [tf.keras.layers.Dense(self.num_features, use_bias=False) for _ in range(num_decision_steps - 1)]
        
        self.bnf = tf.keras.layers.BatchNormalization(momentum=self.batch_momentum)
        
        self.bn1 =  tf.keras.layers.BatchNormalization(momentum=self.batch_momentum, virtual_batch_size=self.virtual_batch_size)
        self.bn2 =  tf.keras.layers.BatchNormalization(momentum=self.batch_momentum, virtual_batch_size=self.virtual_batch_size)
        self.bn3 =  tf.keras.layers.BatchNormalization(momentum=self.batch_momentum, virtual_batch_size=self.virtual_batch_size)
        self.bn4 =  tf.keras.layers.BatchNormalization(momentum=self.batch_momentum, virtual_batch_size=self.virtual_batch_size)
        self.bnm = tf.keras.layers.BatchNormalization(momentum=self.batch_momentum, virtual_batch_size=self.virtual_batch_size)
        self.sparse_max = Sparsemax(axis = -1)
        
        
    def call(self, features, is_training = True):
        
        features = self.bnf(features)
        batch_size = tf.shape(features)[0]
        
#         print(batch_size.eval())
        
        output_aggregated = tf.zeros([batch_size, self.output_dim])
        masked_features = features
        mask_values = tf.zeros([batch_size, self.num_features])
        aggregated_mask_values = tf.zeros([batch_size, self.num_features])
        complemantary_aggregated_mask_values = tf.ones([batch_size, self.num_features])
        total_entropy = 0

        if is_training:
            v_b = self.virtual_batch_size
        else:
            v_b = 1
#        
        for ni in range(self.num_decision_steps):
            reuse_flag = (ni > 0)

            transform_f1 = self.tr1(masked_features)
            transform_f1 = self.bn1(transform_f1)
            transform_f1 = glu(transform_f1, self.feature_dim)

            transform_f2 = self.tr2(transform_f1)
            transform_f2 = self.bn2(transform_f2)
            transform_f2 = (glu(transform_f2, self.feature_dim) + transform_f1) * np.sqrt(0.5)

            transform_f3 = self.tr3[ni](transform_f2)
            transform_f3 = self.bn3(transform_f3)
            transform_f3 = (glu(transform_f3, self.feature_dim) + transform_f2) * np.sqrt(0.5)

            transform_f4 = self.tr4[ni](transform_f3)
            transform_f4 = self.bn4(transform_f4)
            transform_f4 = (glu(transform_f4, self.feature_dim) + transform_f3) * np.sqrt(0.5)
            
            
            if ni > 0:
                decision_out = tf.keras.activations.relu(transform_f4[:, :self.output_dim])
                output_aggregated += decision_out
                scale_agg = tf.math.reduce_sum(decision_out, axis=1, keepdims=True) / (self.num_decision_steps - 1)
                aggregated_mask_values += mask_values * scale_agg
           
            features_for_coef = (transform_f4[:, self.output_dim:])
#             print(features_for_coef)
            if ni < self.num_decision_steps - 1:

             # Determines the feature masks via linear and nonlinear
             # transformations, taking into account of aggregated feature use.
                mask_values = self.mask[ni](features_for_coef)
                mask_values = self.bnm(mask_values)

                mask_values *= complemantary_aggregated_mask_values

                mask_values = self.sparse_max(mask_values)
                
                complemantary_aggregated_mask_values *= (
                                       self.relaxation_factor - mask_values)


                total_entropy += tf.reduce_mean(
                       tf.reduce_sum(-mask_values * tf.math.log(mask_values + self.epsilon),axis=1)) / (self.num_decision_steps - 1)

                
                masked_features = tf.multiply(mask_values, features)
#                 print(mask_values)
#
#                tf.summary.image("Mask for step" + str(ni),
#                        tf.expand_dims(tf.expand_dims(mask_values, 0), 3), max_outputs=1)
            
        return output_aggregated, total_entropy