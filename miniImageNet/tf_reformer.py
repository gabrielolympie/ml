import tensorflow as tf
from tensorflow.keras.layers import Dropout, Dense, Embedding, LayerNormalization
from tensorflow.keras import layers

def mask_fill_inf(matrix, mask):
    negmask = 1 - mask
    num = 3.4 * math.pow(10, 38)
    return (matrix * mask) + (-((negmask * num + num) - num))

def make_unit_length(x, epsilon=1e-6):
    norm = tf.norm(x,  ord=2, axis=-1, keepdims=True)
    return tf.math.truediv(x, norm + epsilon)

def sort_key_val(t1, t2, dim=-1):
    values = tf.sort(t1, axis=dim)
    t2 = tf.broadcast_to(t2, t1.shape)
    return values, tf.gather(t2, tf.argsort(t1, axis=dim), axis=dim)

def batched_index_select(values, indices):
    last_dim = values.shape[-1]
    return tf.squeeze(tf.gather(values, indices[:, :, None], axis=1))

def process_inputs_chunk(fn, *args, chunks=1):
    chunked_inputs = list(map(lambda x: tf.split(x, chunks, axis=0), args))
    outputs = [fn(*input_pair) for input_pair in zip(*chunked_inputs)]
    return outputs

def chunked_sum(tensor, chunks=1):
    *orig_size, last_dim = tensor.shape
    tensor = tf.reshape(tensor,  [-1, last_dim])
    summed_tensors = [c.sum(axis=-1) for c in tf.chunk(tensor, chunks, axis=0)]
    return tf.reshape(torch.concat(summed_tensors, axis=0), orig_size)

def cache_fn(f):
    cache = None
    def cached_fn(*args, **kwargs):
        nonlocal cache
        if cache is not None:
            return cache
        cache = f(*args, **kwargs)
        return cache
    return cached_fn

class ScaleNorm(layers.Layer):
    def __init__(self, emb, eps):
        super(ScaleNorm, self).__init__()
        self.g = tf.Variable(initial_value=w_init(shape=(1,),
                        dtype='float32'),
                        trainable=True)
        self.eps = eps

    def call(self, inputs):
        n = tf.norm(inputs, axis=-1, keepdims=True).clip_by_value(min=self.eps)
        return x / n * self.g

class WithNorm(layers.Layer):
    def __init__(self, norm_class, emb, fn):
        super(WithNorm, self).__init__()
        self.emb = emb
        if isinstance(norm_class, ScaleNorm):
            self.norm = norm_class(emb)
        else:
            self.norm = norm_class()

        self.fn = fn

    def call(self, inputs):
        inputs = self.norm(inputs)
        return self.fn(inputs)

class Chunk(layers.Layer):
    def __init__(self, chunks, fn, along_axis = -1):
        super(Chunk, self).__init__()
        self.axis = along_axis
        self.chunks = chunks
        self.fn = fn

    def call(self, inputs):
        chunks = tf.split(inputs, self.chunks, axis= self.axis)
        return tf.concat([self.fn(c) for c in chunks], axis = self.axis)

class RevBlock(tf.keras.Model):
    """Single reversible block containing several `_Residual` blocks.
    Each `_Residual` block in turn contains two _ResidualInner blocks,
    corresponding to the `F`/`G` functions in the paper.
    """

    def __init__(self,
                n_res,
                filters,
                strides,
                input_shape,
                batch_norm_first=False,
                data_format="channels_first",
                bottleneck=False,
                fused=True,
                dtype=tf.float32):
        """Initialize RevBlock.
        Args:
        n_res: number of residual blocks
        filters: list/tuple of integers for output filter sizes of each residual
        strides: length 2 list/tuple of integers for height and width strides
        input_shape: length 3 list/tuple of integers
        batch_norm_first: whether to apply activation and batch norm before conv
        data_format: tensor data format, "NCHW"/"NHWC"
        bottleneck: use bottleneck residual if True
        fused: use fused batch normalization if True
        dtype: float16, float32, or float64
        """
        super(RevBlock, self).__init__()
        self.blocks = tf.train.checkpoint.List()
        for i in range(n_res):
            curr_batch_norm_first = batch_norm_first and i == 0
            curr_strides = strides if i == 0 else (1, 1)
            block = _Residual(
                filters,
                curr_strides,
                input_shape,
                batch_norm_first=curr_batch_norm_first,
                data_format=data_format,
                bottleneck=bottleneck,
                fused=fused,
                dtype=dtype)
            self.blocks.append(block)

        if data_format == "channels_first":
            input_shape = (filters, input_shape[1] // curr_strides[0],
                        input_shape[2] // curr_strides[1])
        else:
            input_shape = (input_shape[0] // curr_strides[0],
                        input_shape[1] // curr_strides[1], filters)

    def call(self, h, training=True):
        """Apply reversible block to inputs."""

        for block in self.blocks:
            h = block(h, training=training)
        return h

    def backward_grads_and_vars(self, x, y, dy, training=True):
        """Apply reversible block backward to outputs."""

        grads_all = []
        vars_all = []

        for i in reversed(range(len(self.blocks))):
            block = self.blocks[i]
            if i == 0:
                # First block usually contains downsampling that can't be reversed
                with tf.GradientTape() as tape:
                    x = tf.identity(x)
                    tape.watch(x)
                    y = block(x, training=training)

                    grads_combined = tape.gradient(
                        y, [x] + block.trainable_variables, output_gradients=dy)
                    dy = grads_combined[0]
                    grads_all += grads_combined[1:]
                    vars_all += block.trainable_variables
            else:
                y, dy, grads, vars_ = block.backward_grads_and_vars(
                    y, dy, training=training)
                grads_all += grads
                vars_all += vars_

        return dy, grads_all, vars_all

class ReversibleSequence(tf.keras.Model):
    """Single reversible block containing several `_Residual` blocks.
    Each `_Residual` block in turn contains two _ResidualInner blocks,
    corresponding to the `F`/`G` functions in the paper.

    This is based on PyTorch's RevTorch - ReversibleSequence
    """

    def __init__(self,
                blocks):
        """Initialize RevBlock.
        Args:
        n_res: number of residual blocks
        filters: list/tuple of integers for output filter sizes of each residual
        strides: length 2 list/tuple of integers for height and width strides
        input_shape: length 3 list/tuple of integers
        batch_norm_first: whether to apply activation and batch norm before conv
        data_format: tensor data format, "NCHW"/"NHWC"
        bottleneck: use bottleneck residual if True
        fused: use fused batch normalization if True
        dtype: float16, float32, or float64
        """
        super(ReversibleSequence, self).__init__()
        self.blocks = blocks

    def call(self, h, training=True):
        """Apply reversible block to inputs."""
        for block in self.blocks:
            h = block(h, training=training)
        return h

    def backward_grads_and_vars(self, x, y, dy, training=True):
        """Apply reversible block backward to outputs."""

        grads_all = []
        vars_all = []

        for i in reversed(range(len(self.blocks))):
            block = self.blocks[i]
            if i == 0:
                # First block usually contains downsampling that can't be reversed
                with tf.GradientTape() as tape:
                    x = tf.identity(x)
                    tape.watch(x)
                    y = block(x, training=training)

                    grads_combined = tape.gradient(
                        y, [x] + block.trainable_variables, output_gradients=dy)
                    dy = grads_combined[0]
                    grads_all += grads_combined[1:]
                    vars_all += block.trainable_variables
            else:
                y, dy, grads, vars_ = block.backward_grads_and_vars(
                    y, dy, training=training)
                grads_all += grads
                vars_all += vars_

        return dy, grads_all, vars_all


# class _Residual(tf.keras.Model):
#     """Single residual block contained in a _RevBlock. Each `_Residual` object has
#     two _ResidualInner objects, corresponding to the `F` and `G` functions in the
#     paper.
#     Args:
#         filters: output filter size
#         strides: length 2 list/tuple of integers for height and width strides
#         input_shape: length 3 list/tuple of integers
#         batch_norm_first: whether to apply activation and batch norm before conv
#         data_format: tensor data format, "NCHW"/"NHWC",
#         bottleneck: use bottleneck residual if True
#         fused: use fused batch normalization if True
#         dtype: float16, float32, or float64
#     """
#     def __init__(self,
#                 filters,
#                 strides,
#                 input_shape,
#                 batch_norm_first=True,
#                 data_format="channels_first",
#                 bottleneck=False,
#                 fused=True,
#                 dtype=tf.float32):
#         super(_Residual, self).__init__()

#         self.filters = filters
#         self.strides = strides
#         self.axis = 1 if data_format == "channels_first" else 3
#         if data_format == "channels_first":
#             f_input_shape = (input_shape[0] // 2,) + input_shape[1:]
#             g_input_shape = (filters // 2, input_shape[1] // strides[0],
#                             input_shape[2] // strides[1])
#         else:
#             f_input_shape = input_shape[:2] + (input_shape[2] // 2,)
#             g_input_shape = (input_shape[0] // strides[0],
#                             input_shape[1] // strides[1], filters // 2)

#         factory = _BottleneckResidualInner if bottleneck else _ResidualInner
#         self.f = factory(
#             filters=filters // 2,
#             strides=strides,
#             input_shape=f_input_shape,
#             batch_norm_first=batch_norm_first,
#             data_format=data_format,
#             fused=fused,
#             dtype=dtype)
#         self.g = factory(
#             filters=filters // 2,
#             strides=(1, 1),
#             input_shape=g_input_shape,
#             batch_norm_first=batch_norm_first,
#             data_format=data_format,
#             fused=fused,
#             dtype=dtype)

#     def call(self, x, training=True, concat=True):
#         """Apply residual block to inputs."""

#         x1, x2 = tf.split(x, num_or_size_splits=2, axis=self.axis)
#         f_x2 = self.f(x2, training=training)
#         x1_down = ops.downsample(
#             x1, self.filters // 2, self.strides, axis=self.axis)
#         x2_down = ops.downsample(
#             x2, self.filters // 2, self.strides, axis=self.axis)
#         y1 = f_x2 + x1_down
#         g_y1 = self.g(y1, training=training)
#         y2 = g_y1 + x2_down
#         if not concat:  # For correct backward grads
#             return y1, y2

#         return tf.concat([y1, y2], axis=self.axis)

#     def backward_grads_and_vars(self, y, dy, training=True):
#         """Manually compute backward gradients given input and output grads."""
#         dy1, dy2 = tf.split(dy, num_or_size_splits=2, axis=self.axis)

#         with tf.GradientTape(persistent=True) as tape:
#             y = tf.identity(y)
#             tape.watch(y)
#             y1, y2 = tf.split(y, num_or_size_splits=2, axis=self.axis)
#             z1 = y1
#             gz1 = self.g(z1, training=training)
#             x2 = y2 - gz1
#             fx2 = self.f(x2, training=training)
#             x1 = z1 - fx2

#             grads_combined = tape.gradient(
#                 gz1, [z1] + self.g.trainable_variables, output_gradients=dy2)
#             dz1 = dy1 + grads_combined[0]
#             dg = grads_combined[1:]
#             dx1 = dz1

#             grads_combined = tape.gradient(
#                 fx2, [x2] + self.f.trainable_variables, output_gradients=dz1)
#             dx2 = dy2 + grads_combined[0]
#             df = grads_combined[1:]

#             del tape

#         grads = df + dg
#         vars_ = self.f.trainable_variables + self.g.trainable_variables

#         x = tf.concat([x1, x2], axis=self.axis)
#         dx = tf.concat([dx1, dx2], axis=self.axis)

#         return x, dx, grads, vars_

class ReversibleBlock(tf.keras.Model):
    """Single residual block contained in a _RevBlock. Each `_Residual` object has
    two _ResidualInner objects, corresponding to the `F` and `G` functions in the
    paper. This version takes in the F and G block directly, instead of constructing them. 

    This implementation is based on PyTorch's RevTorch - ReversibleBlock
    Args:
        f_block: The first residual block
        g_block: the second residual block
        split_along_axis: axis for splitting, defaults to 1
    """

    def __init__(self,
                f_block,
                g_block,
                split_along_axis=1):
        super(ReversibleBlock, self).__init__()

        self.axis = split_along_axis        
        self.f = f_block
        self.g = g_block

    def call(self, x, training=True, concat=True):
        """Apply residual block to inputs."""

        x1, x2 = tf.split(x, num_or_size_splits=2, axis=self.axis)
        f_x2 = self.f(x2, training=training)
        y1 = f_x2 + x1
        g_y1 = self.g(y1, training=training)
        y2 = g_y1 + x2
        if not concat:  # For correct backward grads
            return y1, y2

        return tf.concat([y1, y2], axis=self.axis)

    def backward_grads_and_vars(self, y, dy, training=True):
        """Manually compute backward gradients given input and output grads."""
        dy1, dy2 = tf.split(dy, num_or_size_splits=2, axis=self.axis)

        with tf.GradientTape(persistent=True) as tape:
            y = tf.identity(y)
            tape.watch(y)
            y1, y2 = tf.split(y, num_or_size_splits=2, axis=self.axis)
            z1 = y1
            gz1 = self.g(z1, training=training)
            x2 = y2 - gz1
            fx2 = self.f(x2, training=training)
            x1 = z1 - fx2

            grads_combined = tape.gradient(
                gz1, [z1] + self.g.trainable_variables, output_gradients=dy2)
            dz1 = dy1 + grads_combined[0]
            dg = grads_combined[1:]
            dx1 = dz1

            grads_combined = tape.gradient(
                fx2, [x2] + self.f.trainable_variables, output_gradients=dz1)
            dx2 = dy2 + grads_combined[0]
            df = grads_combined[1:]

            del tape

        grads = df + dg
        vars_ = self.f.trainable_variables + self.g.trainable_variables

        x = tf.concat([x1, x2], axis=self.axis)
        dx = tf.concat([dx1, dx2], axis=self.axis)

        return x, dx, grads, vars_


def _BottleneckResidualInner(filters,
                             strides,
                             input_shape,
                             batch_norm_first=True,
                             data_format="channels_first",
                             fused=True,
                             dtype=tf.float32):
    """Single bottleneck residual inner function contained in _Resdual.
    Corresponds to the `F`/`G` functions in the paper.
    Suitable for training on ImageNet dataset.
    Args:
        filters: output filter size
        strides: length 2 list/tuple of integers for height and width strides
        input_shape: length 3 list/tuple of integers
        batch_norm_first: whether to apply activation and batch norm before conv
        data_format: tensor data format, "NCHW"/"NHWC"
        fused: use fused batch normalization if True
        dtype: float16, float32, or float64
    Returns:
        A keras model
    """

    axis = 1 if data_format == "channels_first" else 3
    model = tf.keras.Sequential()
    if batch_norm_first:
        model.add(
            tf.keras.layers.BatchNormalization(
                axis=axis, input_shape=input_shape, fused=fused, dtype=dtype))
        model.add(tf.keras.layers.Activation("relu"))
    model.add(
        tf.keras.layers.Conv2D(
            filters=filters // 4,
            kernel_size=1,
            strides=strides,
            input_shape=input_shape,
            data_format=data_format,
            use_bias=False,
            padding="SAME",
            dtype=dtype))

    model.add(
        tf.keras.layers.BatchNormalization(axis=axis, fused=fused, dtype=dtype))
    model.add(tf.keras.layers.Activation("relu"))
    model.add(
        tf.keras.layers.Conv2D(
            filters=filters // 4,
            kernel_size=3,
            strides=(1, 1),
            data_format=data_format,
            use_bias=False,
            padding="SAME",
            dtype=dtype))

    model.add(
        tf.keras.layers.BatchNormalization(axis=axis, fused=fused, dtype=dtype))
    model.add(tf.keras.layers.Activation("relu"))
    model.add(
        tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=1,
            strides=(1, 1),
            data_format=data_format,
            use_bias=False,
            padding="SAME",
            dtype=dtype))

    return model


def _ResidualInner(filters,
                   strides,
                   input_shape,
                   batch_norm_first=True,
                   data_format="channels_first",
                   fused=True,
                   dtype=tf.float32):
    """Single residual inner function contained in _ResdualBlock.
      Corresponds to the `F`/`G` functions in the paper.
      Args:
        filters: output filter size
        strides: length 2 list/tuple of integers for height and width strides
        input_shape: length 3 list/tuple of integers
        batch_norm_first: whether to apply activation and batch norm before conv
        data_format: tensor data format, "NCHW"/"NHWC"
        fused: use fused batch normalization if True
        dtype: float16, float32, or float64
      Returns:
        A keras model
      """

    axis = 1 if data_format == "channels_first" else 3
    model = tf.keras.Sequential()
    if batch_norm_first:
        model.add(
            tf.keras.layers.BatchNormalization(
                axis=axis, input_shape=input_shape, fused=fused, dtype=dtype))
   
        model.add(tf.keras.layers.Activation("relu"))
        model.add(
          tf.keras.layers.Conv2D(
              filters=filters,
              kernel_size=3,
              strides=strides,
              input_shape=input_shape,
              data_format=data_format,
              use_bias=False,
              padding="SAME",
              dtype=dtype))

    model.add(
      tf.keras.layers.BatchNormalization(axis=axis, fused=fused, dtype=dtype))
    model.add(tf.keras.layers.Activation("relu"))
    model.add(
      tf.keras.layers.Conv2D(
          filters=filters,
          kernel_size=3,
          strides=(1, 1),
          data_format=data_format,
          use_bias=False,
          padding="SAME",
          dtype=dtype))

    return model    
    

class MultiHeadAttention(tf.keras.layers.Layer):

    def __init__(self, d_model, num_heads, name="multi_head_attention"):
        super(MultiHeadAttention, self).__init__(name=name)
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0
        self.depth = d_model // self.num_heads
        self.query_dense = Dense(units=d_model)
        self.key_dense = Dense(units=d_model)
        self.value_dense = Dense(units=d_model)
        self.dense = Dense(units=d_model)

    def split_heads(self, inputs, batch_size):
        inputs = tf.reshape(
            inputs, shape=(batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(inputs, perm=[0, 2, 1, 3])

    def call(self, inputs):
        query, key, value, mask = inputs['query'], inputs['key'], inputs[
            'value'], inputs['mask']
        batch_size = tf.shape(query)[0]

        # linear layers
        query = self.query_dense(query)
        key = self.key_dense(key)
        value = self.value_dense(value)

        # split heads
        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        scaled_attention = scaled_dot_product_attention(query, key, value, mask)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention,
                                    (batch_size, -1, self.d_model))

        outputs = self.dense(concat_attention)
        return outputs


class TFSelfAttention(tf.keras.Model):
    def __init__(self, emb, heads = 8, causal = False):
        super().__init__()
        assert emb % heads == 0, 'dimensions must be divisible by number of heads'
        self.attn = MultiheadAttention(emb, heads)
        self.to_out = Dense(emb, emb)
        self.causal = causal

    def call(self, inputs):
        b, t, e = inputs.shape
        inputs = tf.transpose(inputs, (0, 1))

        attn_mask = tf.zeros(t, t)
        if self.causal:
            causal_mask = tf.triu(tf.ones(t, t) == 1, 1)
            mask_fill_inf(attn_mask, causal_mask)

        output = self.attn({'query' : x, 'key' : x, 'value' : x, 'mask' : attn_mask})
        return self.to_out(tf.transpose(output, (0, 1)))


class TFFeedForward(tf.keras.Model):
    def __init__(self, emb, mult = 4):
        super().__init__()
        self.emb = emb
        self.proj_in = Dense(emb * mult)
        self.proj_out = Dense(emb)

    def call(self, inputs):
        inputs = self.proj_in(inputs)
        inputs = tf.keras.activations.relu(inputs)
        inputs = self.proj_out(inputs)
        return inputs


class TFLSHAttention(tf.keras.Model):
    def __init__( self,
                  dropout = 0.,
                  bucket_size = 64,
                  n_hashes = 8,
                  causal = False,
                  allow_duplicate_attention = True,
                  attend_across_buckets = True,
                  rehash_each_round = True,
                  drop_for_hash_rate = 0.0,
                  random_rotations_per_head = False):
        super(TFLSHAttention, self).__init__()
        if dropout >= 1.0:
            raise ValueError('Dropout rates must be lower than 1.')

        self.dropout = Dropout(dropout)
        self.dropout_for_hash = Dropout(dropout)

        assert rehash_each_round or allow_duplicate_attention, (
            'The setting {allow_duplicate_attention=False, rehash_each_round=False}'
            ' is not implemented.')

        self.causal = causal
        self.n_hashes = n_hashes
        self.bucket_size = bucket_size

        self._allow_duplicate_attention = allow_duplicate_attention
        self._attend_across_buckets = attend_across_buckets
        self._rehash_each_round = rehash_each_round
        self._random_rotations_per_head = random_rotations_per_head

    def hash_vectors(self, n_buckets, vecs):
        batch_size = vecs.shape[0]
        device = vecs.device

        # See https://arxiv.org/pdf/1509.02897.pdf
        # We sample a different random rotation for each round of hashing to
        # decrease the probability of hash misses.
        assert n_buckets % 2 == 0

        rot_size = n_buckets

        rotations_shape = (
            batch_size if self._random_rotations_per_head else 1,
            vecs.shape[-1],
            self.n_hashes if self._rehash_each_round else 1,
            rot_size // 2)

        random_rotations = tf.broadcast_to(tf.random.normal(rotations_shape), (batch_size, vecs.shape[-1], self.n_hashes if self._rehash_each_round else 1, rot_size // 2))

        dropped_vecs = self.dropout_for_hash(vecs)
        rotated_vecs = tf.einsum('btf,bfhi->bhti', dropped_vecs, random_rotations)

        if self._rehash_each_round:
            rotated_vecs = tf.concat([rotated_vecs, -rotated_vecs], axis=-1)
            buckets = tf.math.argmax(rotated_vecs, axis=-1)
            # buckets is now (self.n_hashes, seqlen). Next we add offsets so that
            # bucket numbers from different hashing rounds don't overlap.
            offsets = tf.range(self.n_hashes)
            offsets = tf.reshape(offsets * n_buckets, (1, -1, 1))
            offsets = tf.cast(offsets, tf.int64)
            buckets = tf.reshape(buckets + offsets, (batch_size, -1,))
        else:
            rotated_vecs = tf.concat([rotated_vecs, -rotated_vecs], axis=-1)
            # In this configuration, we map each item to the top self.n_hashes buckets
            rotated_vecs = tf.squeeze(rotated_vecs, axis=0)
            bucket_range = tf.range(rotated_vecs.shape[-1])
            bucket_range = tf.reshape(bucket_range, (1, -1))
            bucket_range = tf.broadcast_to(bucket_range, rotated_vecs.shape)

            _, buckets = sort_key_val(rotated_vecs, bucket_range, axis=-1)
            buckets = buckets[:, -self.n_hashes:]

            h, *_ = buckets.shape 
            buckets = tf.reshape(buckets.permute((*_, h)), (-1,))

        return buckets

    def call(self, qk, v):
        batch_size, seqlen, _ = qk.shape
        device = qk.device

        n_buckets = seqlen // self.bucket_size
        n_bins = n_buckets

        buckets = self.hash_vectors(n_buckets, qk)
        # We use the same vector as both a query and a key.
        assert int(buckets.shape[1]) == self.n_hashes * seqlen

        ticker = tf.expand_dims(tf.range(self.n_hashes * seqlen), axis=0)
        buckets_and_t = seqlen * buckets + tf.cast((ticker % seqlen), tf.int64)
        buckets_and_t = tf.stop_gradient(buckets_and_t)

        # Hash-based sort ("s" at the start of variable names means "sorted")
        sbuckets_and_t, sticker = sort_key_val(buckets_and_t, ticker, dim=-1)
        _, undo_sort = sort_key_val(sticker, ticker, dim=-1)
        del ticker

        sbuckets_and_t = tf.stop_gradient(sbuckets_and_t)
        sticker = tf.stop_gradient(sticker)
        undo_sort = tf.stop_gradient(undo_sort)

        st = (sticker % seqlen)
        sqk = batched_index_select(qk, st)
        sv = batched_index_select(v, st)

        # Split off a "bin" axis so that attention only occurs within chunks.
        bq_t = bkv_t = tf.reshape(st, (batch_size, self.n_hashes * n_bins, -1))
        bqk = tf.reshape(sqk, (batch_size, self.n_hashes * n_bins, -1, sqk.shape[-1]))
        bv = tf.reshape(sv, (batch_size, self.n_hashes * n_bins, -1, sv.shape[-1]))
        bq_buckets = bkv_buckets = tf.reshape(sbuckets_and_t // seqlen, (batch_size, self.n_hashes * n_bins, -1))

        # Hashing operates on unit-length vectors. Unnormalized query vectors are
        # fine because they effectively provide a learnable temperature for the
        # attention softmax, but normalizing keys is needed so that similarity for
        # the purposes of attention correctly corresponds to hash locality.
        bq = bqk
        bk = make_unit_length(bqk)

        # Allow each chunk to attend within itself, and also one chunk back. Chunk
        # boundaries might occur in the middle of a sequence of items from the
        # same bucket, so this increases the chances of attending to relevant items.
        def look_one_back(x):
            x_extra = tf.concat([x[:, -1:, ...], x[:, :-1, ...]], axis=1)
            return tf.concat([x, x_extra], axis=2)

        bk = look_one_back(bk)
        bv = look_one_back(bv)
        bkv_t = look_one_back(bkv_t)
        bkv_buckets = look_one_back(bkv_buckets)

        # Dot-product attention.
        dots = tf.einsum('bhie,bhje->bhij', bq, bk) * (bq.shape[-1] ** -0.5)

        # Causal masking
        if self.causal:
            mask = bq_t[:, :, :, None] < bkv_t[:, :, None, :] 
            dots = tf.math.multiply(dots, tf.cast(mask, tf.float32)) + (1-tf.cast(mask, tf.float32)) * float('-inf')
            del mask

        # Mask out attention to self except when no other targets are available.
        self_mask = bq_t[:, :, :, None] == bkv_t[:, :, None, :]
        dots = tf.math.multiply(dots, tf.cast(self_mask, tf.float32)) + (1-tf.cast(self_mask, tf.float32)) * (- 1e5)
        del self_mask

        # Mask out attention to other hash buckets.
        if not self._attend_across_buckets:
            bucket_mask = bq_buckets[:, :, :, None] != bkv_buckets[:, :, None, :]
            dots = tf.math.multiply(dots, tf.cast(bucket_mask, tf.float32)) + (1-tf.cast(bucket_mask, tf.float32)) * float('-inf')
            del bucket_mask

        # Don't double-count query-key pairs across multiple rounds of hashing.
        # There are two possible strategies here. (1) The default is to count how
        # many times a query-key pair is repeated, and to lower its log-prob
        # correspondingly at each repetition. (2) When hard_k is set, the code
        # instead masks all but the first occurence of each query-key pair.
        if not self._allow_duplicate_attention:
            locs1 = undo_sort // bq_t.shape[-1]
            locs2 = (locs1 + 1) % (self.n_hashes * n_bins)
            if not self._attend_across_buckets:
                locs1 = buckets * (self.n_hashes * n_bins) + locs1
                locs2 = buckets * (self.n_hashes * n_bins) + locs2
            locs = tf.transpose(
                tf.concat([
                    tf.reshape(locs1, (batch_size, self.n_hashes, seqlen)),
                    tf.reshape(locs2, (batch_size, self.n_hashes, seqlen)),
                ], 1),
            perm=[0, 2, 1]) 

            slocs = batched_index_select(locs, st)
            b_locs = tf.reshape(slocs, (batch_size, self.n_hashes * n_bins, -1, 2 * self.n_hashes))

            b_locs1 = b_locs[:, :, :, None, :self.n_hashes]

            bq_locs = b_locs1.expand(b_locs.shape[:3] + (2, self.n_hashes))
            bq_locs = tf.reshape(bq_locs, b_locs.shape)
            bkv_locs = look_one_back(b_locs)

            dup_counts = (bq_locs[:, :, :, None, :] == bkv_locs[:, :, None, :, :])
            # for memory considerations, chunk summation of last dimension for counting duplicates
            dup_counts = chunked_sum(dup_counts, chunks=(self.n_hashes * batch_size))
            dup_counts = tf.stop_gradient(dup_counts)
            assert dup_counts.shape == dots.shape
            dots = dots - tf.log(dup_counts + 1e-9)
            del dup_counts

        # Softmax.
        dots_logsumexp = tf.math.reduce_logsumexp(dots, axis=-1, keepdims=True)
        dots = tf.exp(dots - dots_logsumexp)
        dots = self.dropout(dots)

        bo = tf.einsum('buij,buje->buie', dots, bv)
        so = tf.reshape(bo, (batch_size, -1, bo.shape[-1]))
        slogits = tf.reshape(dots_logsumexp, (batch_size, -1,))

        class UnsortLogits(tf.keras.layers.Layer):
            def __init__(self):
                super(UnsortLogits, self).__init__()
            
            def call(self, so, slogits):
                so, slogits = tf.stop_gradient(so), tf.stop_gradient(slogits)
                o = batched_index_select(so, undo_sort)
                _, logits = sort_key_val(sticker, slogits, dim=-1)
                return o, logits

            
        unsortlogits = UnsortLogits()
        o, logits = unsortlogits(so, slogits)

        if self.n_hashes == 1:
            out = o
        else:
            o = tf.reshape(o, (batch_size, self.n_hashes, seqlen, o.shape[-1]))
            logits = tf.reshape(logits, (batch_size, self.n_hashes, seqlen, 1))
            probs = tf.exp(logits - tf.math.reduce_logsumexp(logits, axis=1, keepdims=True))
            out = tf.reduce_sum(o * probs, axis=1)

        assert out.shape == v.shape
        return out, buckets

class TFLSHSelfAttention(tf.keras.Model):
    def __init__(self, emb, heads = 8, bucket_size = 64, n_hashes = 8, causal = False, attn_chunks = None, random_rotations_per_head = False, attend_across_buckets = True, allow_duplicate_attention = True, **kwargs):
        super(TFLSHSelfAttention, self).__init__()
        assert emb % heads == 0, 'dimensions must be divisible by number of heads'

        self.emb = emb
        self.heads = heads
        self.attn_chunks = heads if attn_chunks is None else attn_chunks

        self.toqk = Dense(emb, use_bias = False)
        self.tov = Dense(emb, use_bias = False)
        self.to_out = Dense(emb)

        self.bucket_size = bucket_size
        self.lsh_attn = TFLSHAttention(bucket_size=bucket_size, causal=causal, random_rotations_per_head=random_rotations_per_head, attend_across_buckets = attend_across_buckets,  allow_duplicate_attention = allow_duplicate_attention, **kwargs)

    def call(self, inputs):
        b, t, e, h = *inputs.shape, self.heads
        assert t % self.bucket_size == 0, f'Sequence length needs to be divisible by target bucket size - {self.bucket_size}'

        qk = self.toqk(inputs)
        v = self.tov(inputs)

        def merge_heads(v):
            return tf.reshape(tf.transpose(tf.reshape(v, (b, t, h, -1)), perm=[0, 2, 1, 3]), (b * h, t, -1)) 

        def split_heads(v):
            return tf.transpose(tf.reshape(v, (b, t, h, -1)), perm=[0, 2, 1, 3])

        qk = merge_heads(qk)
        v = merge_heads(v)

        outputs = process_inputs_chunk(self.lsh_attn, qk, v, chunks=self.attn_chunks)
        attn_out = tf.concat([output for (output, _) in outputs], axis=0)

        out = tf.reshape(split_heads(attn_out), (b, t, e))

        return self.to_out(out)
    
    
class TFReformer(tf.keras.Model):
    def __init__(self, emb, depth, max_seq_len, heads = 8, bucket_size = 64, 
                 n_hashes = 8, ff_chunks = 100, attn_chunks = None, 
                 causal = False, weight_tie = False, lsh_dropout = 0., 
                 lsh_attend_across_buckets = True, lsh_allow_duplicate_attention = True, 
                 random_rotations_per_head = False, twin_attention = False, 
                 use_scale_norm = False, use_full_attn = False):
        super().__init__()
        self.emb = emb
        self.depth = depth

        get_full_attn = lambda: TFSelfAttention(emb, heads, causal = causal)
        get_lsh_attn = lambda: TFLSHSelfAttention(emb, heads, bucket_size, n_hashes, causal = causal, dropout = lsh_dropout, attn_chunks = attn_chunks, allow_duplicate_attention = lsh_allow_duplicate_attention, attend_across_buckets = lsh_attend_across_buckets, random_rotations_per_head = random_rotations_per_head)

        get_attn = get_full_attn if use_full_attn else get_lsh_attn
        get_ff = lambda: TFFeedForward(emb)

        if weight_tie:
            get_attn = cache_fn(get_attn)
            get_ff = cache_fn(get_ff)

        blocks = []
        norm_type = ScaleNorm if use_scale_norm else LayerNormalization

        for _ in range(depth):
            attn = get_attn()
            parallel_net = get_attn() if twin_attention else get_ff()
            f = WithNorm(norm_type, emb, attn)
            g = WithNorm(norm_type, emb, parallel_net)

            if not twin_attention and ff_chunks > 1:
                g = Chunk(ff_chunks, g, along_axis = -2)

            blocks.append(ReversibleBlock(f, g, split_along_axis=-1))

        self.model_layers = ReversibleSequence(blocks)

    def call(self, x):
        x = tf.concat([x, x], axis = -1)
        x = self.model_layers(x)
        return tf.stack(tf.reduce_sum(tf.split(x, 2, axis=-1), axis=0))

class TFReformerLM(tf.keras.Model):
    def __init__(self, num_tokens, emb, depth, max_seq_len, heads = 8, bucket_size = 64, 
                 n_hashes = 8, ff_chunks = 100, attn_chunks = None, causal = False, 
                 weight_tie = False, lsh_dropout = 0., random_rotations_per_head = False, 
                 twin_attention = False, use_scale_norm = False, use_full_attn = False):
        super().__init__()
        self.token_emb = Embedding(num_tokens, emb)
        self.pos_emb = Embedding(max_seq_len, emb)
        self.reformer = TFReformer(emb, depth, max_seq_len, heads = heads, bucket_size = bucket_size, n_hashes = n_hashes, ff_chunks = ff_chunks, attn_chunks = attn_chunks, causal = causal, weight_tie = weight_tie, lsh_dropout = lsh_dropout, random_rotations_per_head = random_rotations_per_head, twin_attention = twin_attention, use_scale_norm = use_scale_norm, use_full_attn = use_full_attn)
        self.to_logits = Dense(num_tokens)

    def call(self, inputs):
        print(inputs.shape)
        inputs = self.token_emb(inputs) + self.pos_emb(tf.range(inputs.shape[1]))
        inputs = self.reformer(inputs)
        return self.to_logits(inputs)