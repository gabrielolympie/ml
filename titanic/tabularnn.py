import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp

@tf.function
def sparsemoid(inputs: tf.Tensor):
    return tf.clip_by_value(0.5 * inputs + 0.5, 0., 1.)

@tf.function
def identity(x: tf.Tensor):
    return x



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


    
class GLUBlock(tf.keras.layers.Layer):
    def __init__(self, units = 64,
                 virtual_batch_size = 128, 
                 momentum = 0.02):
        super(GLUBlock, self).__init__()
        self.units = units
        self.virtual_batch_size = virtual_batch_size
        self.momentum = momentum
        self.fc_outout = tf.keras.layers.Dense(self.units, use_bias=False)
        self.bn_outout = tf.keras.layers.BatchNormalization(virtual_batch_size=self.virtual_batch_size, momentum=self.momentum)
        self.fc_gate = tf.keras.layers.Dense(self.units, use_bias=False)
        self.bn_gate = tf.keras.layers.BatchNormalization(virtual_batch_size=self.virtual_batch_size, momentum=self.momentum)
        
        
    def call(self, inputs, 
             training = True):
        output = self.bn_outout(self.fc_outout(inputs), training=training)
        gate = self.bn_gate(self.fc_gate(inputs),training=training)
        return output * tf.keras.activations.sigmoid(gate) # GLU

    
class FeatureTransformerBlock(tf.keras.layers.Layer):
    def __init__(self, units = 64, 
                 virtual_batch_size = 128, 
                 momentum = 0.02, 
                 skip=False):
        super(FeatureTransformerBlock, self).__init__()
        self.units = units
        self.virtual_batch_size = virtual_batch_size
        self.momentum = momentum
        self.skip = skip
        
        self.initial = GLUBlock(units = self.units,virtual_batch_size=self.virtual_batch_size, momentum=self.momentum)
        self.residual =  GLUBlock(units = self.units,virtual_batch_size=self.virtual_batch_size, momentum=self.momentum)
        
    def call(self, inputs, 
             training = True):
        
        initial = self.initial(inputs, training=training)
        
        if self.skip == True:
            initial += inputs

        residual = self.residual(initial, training=training) # skip
        return (initial + residual) * np.sqrt(0.5)
    
    
class AttentiveTransformer(tf.keras.layers.Layer):
    def __init__(self, units = 64, 
                 virtual_batch_size = 128, 
                 momentum = 0.02):
        super(AttentiveTransformer, self).__init__()
        self.units = units
        self.virtual_batch_size = virtual_batch_size
        self.momentum = momentum
        self.sparse_max = Sparsemax(axis = -1)
        
        self.fc = tf.keras.layers.Dense(self.units, use_bias=False)
        self.bn = tf.keras.layers.BatchNormalization(virtual_batch_size=self.virtual_batch_size, momentum=self.momentum)
        
        
    def call(self, inputs, 
             priors = None, 
             training = None):
        
        feature = self.bn(self.fc(inputs), 
                          training=training)
        if priors is None:
            output = feature
        else:
            output = feature * priors
            
        return self.sparse_max(output)
    
class TabNetStep(tf.keras.layers.Layer):
    def __init__(self, units = 64, 
                 n_features = 64,
                 virtual_batch_size = 128, 
                 momentum = 0.02):
        super(TabNetStep, self).__init__()
        self.units = units
        self.virtual_batch_size = virtual_batch_size
        self.momentum = momentum
        self.n_features = n_features
        
        self.unique = FeatureTransformerBlock(units = self.units, virtual_batch_size=self.virtual_batch_size, momentum=self.momentum, skip=True)
        self.attention = AttentiveTransformer(units = n_features, virtual_batch_size=self.virtual_batch_size,momentum=self.momentum)
        
        
    def call(self, inputs, shared, priors, training=None):  
        split = self.unique(shared, training=training)
        keys = self.attention(split, priors, training=training)
        masked = keys * inputs
        return split, masked, keys

class TabNetEncoder(tf.keras.layers.Layer):
    def __init__(self,                  
                 n_steps = 3, 
                 n_features = 8,
                 outputs = 1, 
                 gamma = 1.3,
                 epsilon = 1e-8, 
                 sparsity = 1e-5, 
                 virtual_batch_size=128, 
                 momentum =0.02):
        '''
        n_steps : the number of steps in the model
        n_features : the number of input features
        outputs : shape of output dimension
        gamma : indice of features penalisation accross steps
        sparsity : factor to penalise loss
        virtual_batch_size : size of virtual batch for normalisation
        momentum : momentum for batch normalization
        '''
        
        super(TabNetEncoder, self).__init__()
        
#        self.units = units
        self.n_steps = n_steps
        self.n_features = n_features
        self.virtual_batch_size = virtual_batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.momentum = momentum
        self.sparsity = sparsity
        
        self.bn = tf.keras.layers.BatchNormalization(virtual_batch_size=self.virtual_batch_size, momentum=self.momentum)
        self.shared_block = FeatureTransformerBlock(units = self.n_features, virtual_batch_size=self.virtual_batch_size, momentum=self.momentum) 
        self.initial_step = TabNetStep(units = self.n_features ,n_features = self.n_features,virtual_batch_size=self.virtual_batch_size, momentum=self.momentum)
        self.steps = [TabNetStep(units = self.n_features,n_features = self.n_features, virtual_batch_size=self.virtual_batch_size, momentum=self.momentum) for _ in range(self.n_steps)]
#        self.final = tf.keras.layers.Dense(units = self.units, use_bias=False)

    def call(self, X, 
             training = True):        
        entropy_loss = 0.
        encoded = 0.
        output = 0.
        importance = 0.
        prior = tf.reduce_mean(tf.ones_like(X), axis=0)
        
        B = prior * self.bn(X, training=training)
        
        shared = self.shared_block(B, training=training)
        
        _, masked, keys = self.initial_step(B, shared, prior, training=training)

        for step in self.steps:
            entropy_loss += tf.reduce_mean(tf.reduce_sum(-keys * tf.math.log(keys + self.epsilon), axis=-1)) / tf.cast(self.n_steps, tf.float32)
            prior *= (self.gamma - tf.reduce_mean(keys, axis=0))
            importance += keys
            
            shared = self.shared_block(masked, training=training)
            
            split, masked, keys = step(B, shared, prior, training=training)
            
            features = tf.keras.activations.relu(split)
            
            output += features
            encoded += split
            
        self.add_loss(self.sparsity * entropy_loss)
          
        return output,encoded, importance

    
class ODST(tf.keras.layers.Layer):
    def __init__(self, n_trees = 3, depth = 4, units = 1, threshold_init_beta = 1.):
        super(ODST, self).__init__()
        self.initialized = False
        self.n_trees = n_trees
        self.depth = depth
        self.units = units
        self.threshold_init_beta = threshold_init_beta
    
    def build(self, input_shape):
        feature_selection_logits_init = tf.zeros_initializer()
        self.feature_selection_logits = tf.Variable(initial_value=feature_selection_logits_init(shape=(input_shape[-1], self.n_trees, self.depth), dtype='float32'),
                                 trainable=True)        
        
        feature_thresholds_init = tf.zeros_initializer()
        self.feature_thresholds = tf.Variable(initial_value=feature_thresholds_init(shape=(self.n_trees, self.depth), dtype='float32'),
                                 trainable=True)
        
        log_temperatures_init = tf.ones_initializer()
        self.log_temperatures = tf.Variable(initial_value=log_temperatures_init(shape=(self.n_trees, self.depth), dtype='float32'),
                                 trainable=True)
        
        indices = tf.keras.backend.arange(0, 2 ** self.depth, 1)
        offsets = 2 ** tf.keras.backend.arange(0, self.depth, 1)
        bin_codes = (tf.reshape(indices, (1, -1)) // tf.reshape(offsets, (-1, 1)) % 2)
        bin_codes_1hot = tf.stack([bin_codes, 1 - bin_codes], axis=-1)
        self.bin_codes_1hot = tf.Variable(initial_value=tf.cast(bin_codes_1hot, 'float32'),
                                 trainable=False)
        
        response_init = tf.ones_initializer()
        self.response = tf.Variable(initial_value=response_init(shape=(self.n_trees, self.units, 2**self.depth), dtype='float32'),
                                 trainable=True)
                
    def initialize(self, inputs):        
        feature_values = self.feature_values(inputs)
        
        # intialize feature_thresholds
        percentiles_q = (100 * tfp.distributions.Beta(self.threshold_init_beta, 
                                                      self.threshold_init_beta)
                         .sample([self.n_trees * self.depth]))
        flattened_feature_values = tf.map_fn(tf.keras.backend.flatten, feature_values)
        init_feature_thresholds = tf.linalg.diag_part(tfp.stats.percentile(flattened_feature_values, percentiles_q, axis=0))
        
        self.feature_thresholds.assign(tf.reshape(init_feature_thresholds, self.feature_thresholds.shape))
        
        
        # intialize log_temperatures
        self.log_temperatures.assign(tfp.stats.percentile(tf.math.abs(feature_values - self.feature_thresholds), 50, axis=0))
        
        
        
    def feature_values(self, inputs, training = True):
        feature_selectors = Sparsemax()(self.feature_selection_logits)
        # ^--[in_features, n_trees, depth]

        feature_values = tf.einsum('bi,ind->bnd', inputs, feature_selectors)
        # ^--[batch_size, n_trees, depth]
        
        return feature_values
        
    def call(self, inputs, training = True):
        if not self.initialized:
            self.initialize(inputs)
            self.initialized = True
            
        feature_values = self.feature_values(inputs)
        
        threshold_logits = (feature_values - self.feature_thresholds) * tf.math.exp(-self.log_temperatures)

        threshold_logits = tf.stack([-threshold_logits, threshold_logits], axis=-1)
        # ^--[batch_size, n_trees, depth, 2]

        bins = sparsemoid(threshold_logits)
        # ^--[batch_size, n_trees, depth, 2], approximately binary

        bin_matches = tf.einsum('btds,dcs->btdc', bins, self.bin_codes_1hot)
        # ^--[batch_size, n_trees, depth, 2 ** depth]

        response_weights = tf.math.reduce_prod(bin_matches, axis=-2)
        # ^-- [batch_size, n_trees, 2 ** depth]

        response = tf.einsum('bnd,ncd->bnc', response_weights, self.response)
        # ^-- [batch_size, n_trees, units]
        
        return tf.reduce_sum(response, axis=1)
    
class NODE(tf.keras.Model):
    def __init__(self, units = 1, n_layers = 1, link = tf.identity, n_trees = 3, depth = 4, threshold_init_beta = 1., feature_column = None):
        super(NODE, self).__init__()
        self.units = units
        self.n_layers = n_layers
        self.n_trees = n_trees
        self.depth = depth
        self.units = units
        self.threshold_init_beta = threshold_init_beta
        self.feature_column = feature_column
        
        if feature_column is None:
            self.feature = tf.keras.layers.Lambda(identity)
        else:
            self.feature = feature_column
        
        self.bn = tf.keras.layers.BatchNormalization()
        self.ensemble = [ODST(n_trees = n_trees,
                              depth = depth,
                              units = units,
                              threshold_init_beta = threshold_init_beta) 
                         for _ in range(n_layers)]
        
        self.link = link
        
        
    def call(self, inputs, training=None):
        X = self.feature(inputs)
        X = self.bn(X, training=training)
        
        for tree in self.ensemble:
            H = tree(X)
            X = tf.concat([X, H], axis=1)
            
        return self.link(H)
    
    
    
