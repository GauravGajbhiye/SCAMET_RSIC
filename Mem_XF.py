#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 10 11:25:42 2022

@author: gaurav
"""

import tensorflow as tf
import numpy as np


#Defining the Transformer parameters
batch_size =64
num_layers = 3
d_model = 512
buffer_size = 256
dff = 2048
num_heads = 8
target_vocab_size = 203
input_vocab_size = target_vocab_size
dropout_rate = 0.2


#Defining Positional embedding and masking function

def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates

def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model)
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])  
    pos_encoding = angle_rads[np.newaxis, ...] 
    return tf.cast(pos_encoding, dtype=tf.float32)


def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    return seq[:, tf.newaxis, tf.newaxis, :]

def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask

def create_masks(inp, tar):
    enc_padding_mask = create_padding_mask(inp)# image
    dec_padding_mask = create_padding_mask(inp)# image
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])# report
    dec_target_padding_mask = create_padding_mask(tar)#report
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)#report
    return enc_padding_mask, combined_mask, dec_padding_mask


# scaled dot product attention
def scaled_dot_product_attention(q, k, v, mask):
    matmul_qk = tf.matmul(q, k, transpose_b=True)
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)  
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    output = tf.matmul(attention_weights, v)  
    return output, attention_weights


def gelu(x):
    return 0.5 * x * (1 + tf.tanh(tf.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3))))

tf.keras.utils.get_custom_objects().update(
    {'gelu': tf.keras.layers.Activation(gelu)})


# point-wise feed forward neural network
def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(d_model)
    ])



#Multi-Head-Attention
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
        self.dense = tf.keras.layers.Dense(d_model, activation='gelu')
        
    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention,(batch_size, -1, self.d_model))
        output = self.dense(concat_attention)
        return output, attention_weights


class MemoryMHA(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, num_memory):
        super(MemoryMHA, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.num_memory = num_memory
        assert d_model % self.num_heads == 0
        self.depth = d_model // self.num_heads
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
        self.k_memory = tf.Variable(tf.random_normal_initializer(0, 1/self.depth)(shape=(1, self.num_memory, self.d_model)))
        self.v_memory = tf.Variable(tf.random_normal_initializer(0, 1/self.num_memory)(shape=(1, self.num_memory, self.d_model)))
        self.k_w = tf.keras.layers.Dense(d_model)
        self.v_w = tf.keras.layers.Dense(d_model)
        self.dense = tf.keras.layers.Dense(d_model)
   
    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)
        
        k_memory = self.k_w(self.k_memory)
        v_memory = self.v_w(self.v_memory)
        
        k = tf.concat([k,tf.tile(k_memory,[k.shape[0],1,1])],axis=1) # (batch_size,seq_len_k+num_memory,d_model)

        v = tf.concat([v,tf.tile(v_memory,[v.shape[0],1,1])],axis=1) # (batch_size,seq_len_q+num_memory,d_model)
               
        
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        memory_attention = tf.reshape(scaled_attention,(batch_size, -1, self.d_model))
        memory_out = self.dense(memory_attention)
        return memory_out, attention_weights


# Define Transformers Encoder
#Simple Fully Connected Encoder
class FC_Encoder(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(FC_Encoder, self).__init__()
        self.fc = tf.keras.layers.Dense(d_model, activation='relu')
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dr1 = tf.keras.layers.Dropout(rate)
   
    def call(self, x, training):
        x_fc = self.norm1(self.fc(self.dr1(x)))
        return x_fc, x_fc
        
#Muti-Attentive Encoder
class CSA_Encoder(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(CSA_Encoder, self).__init__()
        self.fc = tf.keras.layers.Dense(d_model, activation='relu')
        self.Ws = tf.keras.layers.Dense(d_model, activation='relu')
        self.Wc = tf.keras.layers.Dense(d_model, activation='relu')
        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dr = tf.keras.layers.Dropout(rate)
        self.dr1 = tf.keras.layers.Dropout(rate)
        self.dr2 = tf.keras.layers.Dropout(rate)
   
    def call(self, x, training):
        x_fc = self.norm(self.fc(self.dr(x)))
        #Channel-attention
        c_avg = tf.math.reduce_mean(x_fc, axis=1)[:,tf.newaxis,:]
        c_att = x_fc*tf.nn.sigmoid(self.Wc(self.dr1(c_avg)))
        #Spatial-attention
        s_avg = tf.math.reduce_mean(x_fc, axis=-1)[:,:,tf.newaxis]
        s_att = x_fc*tf.nn.sigmoid(self.Ws(self.dr2(s_avg)))
        
        return self.norm1(tf.nn.relu(c_att)), self.norm2(tf.nn.relu(s_att))
    


#Define Transformers Decoder Layer
class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()
        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)
        self.ffn1 = point_wise_feed_forward_network(d_model, dff)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dr1 = tf.keras.layers.Dropout(rate)
        self.dr2 = tf.keras.layers.Dropout(rate)
        self.dr3 = tf.keras.layers.Dropout(rate) 

    
    def call(self, x, enc_fc, enc_ff, training, look_ahead_mask, padding_mask):
        #SM1:
        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)  
        out1 = self.layernorm1(self.dr1(attn1, training=training) + x)
        #SM2:
        attn2, attn_weights_block2 = self.mha2(enc_ff, enc_fc, out1, padding_mask)
        out2 = self.layernorm2(self.dr2(attn2, training=training) + out1)
        #SM3:
        ff_out2 = self.ffn1(out2)
        ff_out2 = self.layernorm3(self.dr3(ff_out2, training=training) + out2)
    
        return ff_out2, attn_weights_block1, attn_weights_block2
    

class Mem_DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, num_memory, d_model, num_heads, dff, rate=0.1):
        super(Mem_DecoderLayer, self).__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.mmha = MemoryMHA(d_model, num_heads, num_memory)
        self.ffn = point_wise_feed_forward_network(d_model, dff)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dr1 = tf.keras.layers.Dropout(rate)
        self.dr2 = tf.keras.layers.Dropout(rate)
        self.dr3 = tf.keras.layers.Dropout(rate) 

    
    def call(self, x, enc_fc, enc_ff, training, look_ahead_mask, padding_mask):
        #SM1:
        attn1, attn_weights_block1 = self.mha(x, x, x, look_ahead_mask)  
        out1 = self.layernorm1(self.dr1(attn1, training=training) + x)
        #SMM2:
        attn2, attn_weights_block2 = self.mmha(enc_ff, enc_fc, out1, padding_mask)
        out2 = self.layernorm2(self.dr2(attn2, training=training) + out1)
        #SM3:
        ff_out2 = self.ffn(out2)
        ff_out2 = self.layernorm3(self.dr3(ff_out2, training=training) + out2)
    
        return ff_out2, attn_weights_block1, attn_weights_block2
    
   
#Define Transformers Decoder 
class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, EMB_MAT, num_heads, dff, target_vocab_size, maximum_position_encoding, rate=0.1):
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model, embeddings_initializer=EMB_MAT)
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)
        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)
    
    def call(self, x, enc_att, enc_ff, training, look_ahead_mask, padding_mask):
        seq_len = tf.shape(x)[1]
        attention_weights = {}
        x = self.embedding(x) #(batch_size, maxlen-1, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x, training=training)
        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_att, enc_ff, training, look_ahead_mask, padding_mask)
            attention_weights['decoder_layer{}_block1'.format(i+1)] = block1
            attention_weights['decoder_layer{}_block2'.format(i+1)] = block2
        return self.norm(x), attention_weights
    
class Memory_Decoder(tf.keras.layers.Layer):
    def __init__(self, num_memory, num_layers, d_model, EMB_MAT, num_heads, dff, target_vocab_size, maximum_position_encoding, rate=0.1):
        super(Memory_Decoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model, embeddings_initializer=EMB_MAT)
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)
        self.dec_layers = [Mem_DecoderLayer(num_memory, d_model, num_heads, dff, rate) for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_fc, enc_ff, training, look_ahead_mask, padding_mask):
        seq_len = tf.shape(x)[1]
        attention_weights = {}
        x = self.embedding(x) #(batch_size, maxlen-1, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x, training=training)
        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_fc, enc_ff, training, look_ahead_mask, padding_mask)
            attention_weights['decoder_layer{}_block1'.format(i+1)] = block1
            attention_weights['decoder_layer{}_block2'.format(i+1)] = block2
        
        return self.norm(x), attention_weights
        



class Mem_Transformer(tf.keras.Model):
    def __init__(self, num_memory, num_layers, d_model, EMB_MAT, num_heads, dff, input_vocab_size, target_vocab_size, pe_input, pe_target, rate=0.2):
        super(Mem_Transformer, self).__init__()

        #self.encoder = FC_Encoder(d_model, num_heads, dff, rate) #Simple Fully-connected Encoder
        self.encoder = CSA_Encoder(d_model, num_heads, dff, rate) #Muti-Attentive Encoder
        #self.decoder = Decoder(num_layers, d_model, EMB_MAT, num_heads, dff, target_vocab_size, pe_target, rate)# Transformer Decoder
        self.decoder = Memory_Decoder(num_memory, num_layers, d_model, EMB_MAT, num_heads, dff, target_vocab_size, pe_target, rate)  #Memory-Guided Decoder
        self.final_layer = tf.keras.layers.Dense(target_vocab_size) # Final Prediction
    
    def call(self, inp, tar, training, enc_padding_mask, look_ahead_mask, dec_padding_mask): 
        c_att, s_att = self.encoder(inp, training)
        dec_output, attention_weights = self.decoder(tar, c_att, s_att, training, look_ahead_mask, dec_padding_mask) 
        final_output = self.final_layer(dec_output)
        return tf.nn.log_softmax(final_output), c_att, s_att

num_memory = 40
#EMB_MAT = tf.keras.initializers.Constant(np.random.random((input_vocab_size, d_model)))
EMB_MAT=None
img_inp = np.random.randn(batch_size,100,1420)
target = np.random.randint(0, input_vocab_size, (batch_size, 30))

m_transformer = Mem_Transformer(num_memory, num_layers, d_model, EMB_MAT, num_heads, dff, input_vocab_size, target_vocab_size, pe_input=input_vocab_size, pe_target=target_vocab_size, rate=dropout_rate)

out, _, _ = m_transformer(img_inp, target, False, None, None, None)
