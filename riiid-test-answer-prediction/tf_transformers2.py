import tensorflow as tf
import numpy as np


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates

def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                          np.arange(d_model)[np.newaxis, :],
                          d_model)
  
  # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
  
  # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    
    pos_encoding = angle_rads[np.newaxis, ...]
    
    return tf.cast(pos_encoding, dtype=tf.float32)

def create_padding_mask(seq, add_dimension = True, pad_token = 0):
    seq = tf.cast(tf.math.equal(seq, pad_token), tf.float32)
  
  # add extra dimensions to add the padding
  # to the attention logits.
    if add_dimension:
        return seq[:, tf.newaxis, tf.newaxis, :]
    else:
        return seq[:, :]  # (batch_size, 1, 1, seq_len)

def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)

def scaled_dot_product_attention(q, k, v, mask):
    """Calculate the attention weights.
      q, k, v must have matching leading dimensions.
      k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
      The mask has different shapes depending on its type(padding or look ahead) 
      but it must be broadcastable for addition.

      Args:
        q: query shape == (..., seq_len_q, depth)
        k: key shape == (..., seq_len_k, depth)
        v: value shape == (..., seq_len_v, depth_v)
        mask: Float tensor with shape broadcastable 
              to (..., seq_len_q, seq_len_k). Defaults to None.

      Returns:
        output, attention_weights
      """

    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)
  
  # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

  # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)  

  # softmax is normalized on the last axis (seq_len_k) so that the scores
  # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)
        
    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention, 
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights
    


def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
          tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
          tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
      ])

class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
    
    def call(self, x, training, mask):

        attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2
   
    
class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)
    
    
    def call(self, x, enc_output, training, 
           look_ahead_mask, padding_mask):
    # enc_output.shape == (batch_size, input_seq_len, d_model)

        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)  # (batch_size, target_seq_len, d_model)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        attn2, attn_weights_block2 = self.mha2(
            enc_output, enc_output, out1, padding_mask)  # (batch_size, target_seq_len, d_model)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)

        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)

        return out3, attn_weights_block1, attn_weights_block2    

class GPTDecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(GPTDecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)
    
    
    def call(self, x, training, look_ahead_mask):
    # enc_output.shape == (batch_size, input_seq_len, d_model)

        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)  # (batch_size, target_seq_len, d_model)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

#        attn2, attn_weights_block2 = self.mha2(out1, out1, out1, look_ahead_mask)  # (batch_size, target_seq_len, d_model)
#        attn2 = self.dropout2(attn2, training=training)
#        out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out1)  # (batch_size, target_seq_len, d_model)

        return out3, attn_weights_block1  
    

class Encoder(tf.keras.layers.Layer):
    
    """
    Build an encoder block, 
    Padding mask are built automatically
    
    Inputs initialization:
        - num_layers : the number of encoder layers
        - d_model : size of representation (must be the same as the decoder)
        - num_heads : the number of attention heads
        - dff : size of representation for dense layer
        - input_vocab_size : size of the input vocabulary
        - maximum_position_encoding : maximum number of tokens in a sequence
        - num_types : the number of distinct ids types you can get
        - rate : percentage of dropout
        - bidirectional_encoder : bool, whether to build as bi directional or mono directional
        
    Inputs on call:
        - x : the inputs ids of the encoder
        - training : (optional) whether or not to activate dropout
        - token_types_ids : (optional) the token types ids of the decoder sequence
    
    Outputs on call:
        - x : the encoded sequence
        - mask : the padding mask of the encoder (to be used with the decoder)
    """
    
    
    def __init__(self, num_layers = 2, d_model = 512, num_heads = 8, dff = 1024, input_vocab_size = 10000, maximum_position_encoding = 512, num_types = 2, rate=0.1, bidirectional_encoder = True):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        
        self.token_types_embedding = tf.keras.layers.Embedding(num_types, d_model)
        
        self.pos_encoding = positional_encoding(maximum_position_encoding, 
                                                self.d_model)

 
        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate) 
                           for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)
        
        self.bidirectional_encoder = bidirectional_encoder
        
    def call(self, x, training, token_types_ids = None):
        """
        Two arguments to pass:
            x : the input sequence of the transformer
            training : bool, whether to train or not for dropout
        
        """
        seq_len = tf.shape(x)[1]
        
        if self.bidirectional_encoder == False:
            look_ahead_mask = create_look_ahead_mask(tf.shape(x)[1])
            dec_target_padding_mask = create_padding_mask(x)
            mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
        else:
            mask = create_padding_mask(x)
        
        # adding embedding and position encoding.
        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        
        if token_types_ids is not None:
            token_types_ids_emb = self.token_types_embedding(token_types_ids)
            x += token_types_ids_emb
        
        x = self.dropout(x, training=training)
        

        
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x, mask  # (batch_size, input_seq_len, d_model)
    

class GPTDecoder(tf.keras.layers.Layer):
    """
    Build a pure decoder block, bypassing the use of an encoder
    Attention masks for padding are automatically computed with tokens 0
    
    Inputs initialization:
        - num_layers : the number of decoder layers
        - d_model : size of representation (must be the same as the encoder)
        - num_heads : the number of attention heads
        - dff : size of representation for dense layer
        - target_vocab_size : size of the target vocabulary
        - maximum_position_encoding : maximum number of tokens in a sequence
        - num_types : the number of distinct ids types you can get
        - rate : percentage of dropout
        - bidirectional_decoder : bool, whether to build as bi directional or mono directional
    
    Inputs on call :
        - x : the input sequence of the decoder
        - enc_output : the output of a decoder block with the same d_model as the encoder ex : 768 for a bert encoder
        - training : (optional) whether or not to activate dropout
        - padding_mask : (optional) the padding masks of the encoder sequence
        - token_types_ids : (optional) the token types ids of the decoder sequence
    
    Outputs on call :
        - x : the embedded decoded (you can add a dense layer on top to make a classification)
        - attention_weights
        
    """
    
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size,
               maximum_position_encoding, num_types = 2, rate=0.1, bidirectional_decoder = False):
        super(GPTDecoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        
        self.token_types_embedding = tf.keras.layers.Embedding(num_types, d_model)
        
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)

        self.dec_layers = [GPTDecoderLayer(d_model, num_heads, dff, rate) 
                           for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)
        
        self.bidirectional_decoder = bidirectional_decoder
    
    def call(self, x, training = True, token_types_ids = None):

        seq_len = tf.shape(x)[1]
        attention_weights = {}
        
        if self.bidirectional_decoder == False:
            look_ahead_mask = create_look_ahead_mask(tf.shape(x)[1])
            dec_target_padding_mask = create_padding_mask(x)
            mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
        else:
            mask = create_padding_mask(x)
        
        x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        
        if token_types_ids is not None:
            token_types_ids_emb = self.token_types_embedding(token_types_ids)
            x += token_types_ids_emb
        
        x = self.dropout(x, training=training)
        

        for i in range(self.num_layers):
            x, block1 = self.dec_layers[i](x, training, look_ahead_mask = mask)

            attention_weights['decoder_layer{}_block1'.format(i+1)] = block1
#            attention_weights['decoder_layer{}_block2'.format(i+1)] = block2

        # x.shape == (batch_size, target_seq_len, d_model)
        return x, attention_weights  
    
    
class Decoder(tf.keras.layers.Layer):
    """
    build a decoder block to be used with the output of an encoder
    Attention masks for padding are automatically computed with tokens 0
    
    Inputs initialization:
        - num_layers : the number of decoder layers
        - d_model : size of representation (must be the same as the encoder)
        - num_heads : the number of attention heads
        - dff : size of representation for dense layer
        - target_vocab_size : size of the target vocabulary
        - maximum_position_encoding : maximum number of tokens in a sequence
        - num_types : the number of distinct ids types you can get
        - rate : percentage of dropout
        - bidirectional_decoder : bool, whether to build as bi directional or mono directional
    
    Inputs on call :
        - x : the input sequence of the decoder
        - enc_output : the output of a decoder block with the same d_model as the encoder ex : 768 for a bert encoder
        - training : (optional) whether or not to activate dropout
        - padding_mask : (optional) the padding masks of the encoder sequence
        - token_types_ids : (optional) the token types ids of the decoder sequence
    
    Outputs on call :
        - x : the embedded decoded (you can add a dense layer on top to make a classification)
        - attention_weights
        
    """
    
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size,
               maximum_position_encoding, num_types = 2, rate=0.1, bidirectional_decoder = False):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        
        self.token_types_embedding = tf.keras.layers.Embedding(num_types, d_model)
        
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)

        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate) 
                           for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)
        
        self.bidirectional_decoder = bidirectional_decoder
    
    def call(self, x, enc_output, training = True, padding_mask = None, token_types_ids = None):

        seq_len = tf.shape(x)[1]
        attention_weights = {}
        
        if self.bidirectional_decoder == False:
            look_ahead_mask = create_look_ahead_mask(tf.shape(x)[1])
            dec_target_padding_mask = create_padding_mask(x)
            mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
        else:
            mask = create_padding_mask(x)
        
        x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        
        if token_types_ids is not None:
            token_types_ids_emb = self.token_types_embedding(token_types_ids)
            x += token_types_ids_emb
        
        x = self.dropout(x, training=training)
        

        
        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, training,
                                                 mask, padding_mask)

            attention_weights['decoder_layer{}_block1'.format(i+1)] = block1
            attention_weights['decoder_layer{}_block2'.format(i+1)] = block2

        # x.shape == (batch_size, target_seq_len, d_model)
        return x, attention_weights   
    
    
class Transformer(tf.keras.Model):
    """
    Build an encoder decoder transformers architecture
    
    Inputs initialization :
        - num_layers : number of layer for the encoder block and for the decoder block
        - d_model : size of the embedding representation of the tokens (must be the same in encoder and decoder)
        - num_heads : number of attention head (d_model must be a multiple of num_heads)
        - dff : size of the dense layer in the feed forward network
        - input_vocab_size : size of the vocabulary inputed in the encoder
        - target_vocab_size : size of the vocabulary of the decoder
        - pe_input : max positional encoding of the encoder (max len of sequence)
        - pe_target : max positional encoding of the decoder (max len of sequence)
        - rate : dropout rate in the feed forward network
        - bidirectional_decoder : if the decoder must be bidirectional or not
        
    Inputs on call :
        - inp : the tokenized inputs of the encoder
        - tar : the tokenized input of the decoder
        - training : whether or not to activate dropout
        - input_token_types_ids : (optional) the token types id for sentence separation of the encoder
        - output_token_types_ids : (optional) the token types id for sentence separation of the decoder
        
    Outputs on call :
        - final_output : the prediction of probability per position and token (use an argmax to get the predicted token)
        - attention_weights :
    """
    
    
    
    def __init__(self, num_layers = 2, d_model = 512, num_heads = 8, dff = 1024, input_vocab_size = 10000, target_vocab_size = 10000, pe_input = 512, pe_target = 512,num_types = 2, rate=0.1, bidirectional_encoder = True, bidirectional_decoder = False):
        super(Transformer, self).__init__()

        self.encoder = Encoder(num_layers, d_model, num_heads, dff, 
                               input_vocab_size, pe_input, num_types, rate, bidirectional_encoder)

        self.decoder = Decoder(num_layers, d_model, num_heads, dff, 
                               target_vocab_size, pe_target,num_types, rate, bidirectional_decoder)

        self.final_layer = tf.keras.layers.Dense(target_vocab_size)
    
    def call(self, inp, tar, training = True, input_token_types_ids = None, output_token_types_ids = None):
        
        enc_output, dec_padding_mask = self.encoder(inp, training, token_types_ids = input_token_types_ids)

#        dec_padding_mask = create_padding_mask(inp)
        
        dec_output, attention_weights = self.decoder(
            tar, enc_output, training, dec_padding_mask, token_types_ids = output_token_types_ids)

        final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)

        return final_output, attention_weights 
    
    
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000, temperature = 10):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)
        self.temperature = temperature
        self.warmup_steps = warmup_steps
    
    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)/self.temperature
    
    
    
    
    
    
    
    
    
    
    

