#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 14:55:06 2022

@author: gaurav
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from pickle import load
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tensorflow.keras.preprocessing.image import array_to_img, img_to_array
from My_XF import Mem_Transformer, create_masks

data_path = './SCAMET/'
tokenizer = load(open(data_path+'/data_preprocess/sc_tokenizer.p',"rb"))

vocab_size = len(tokenizer.word_index)
max_len = 23

batch_size = 128
num_layers = 3
d_model =  512
buffer_size = 512
dff = d_model*4
num_heads = 8
target_vocab_size = vocab_size
input_vocab_size = target_vocab_size
dropout_rate = 0.2
EMB_MAT=None
num_memory = 30

img_path = data_path+'/Sydney_captions/Images/'

img_feature_path = data_path+'/Sydney_captions/Img_features/'

transformer = Mem_Transformer(num_memory, num_layers, d_model, EMB_MAT, num_heads, dff, input_vocab_size, target_vocab_size, pe_input=max_len, pe_target=target_vocab_size, rate=dropout_rate)

chkpt_path = data_path+'/Chk_Path/EN_3DE'



chkpt = tf.train.Checkpoint(transformer=transformer)
chkpt_manager = tf.train.CheckpointManager(chkpt, chkpt_path, max_to_keep=5)
chkpt.restore(chkpt_manager.latest_checkpoint)


# Retrieving captions, channel attention and spatial attention
def feat_att(test_img_n):
  img_id = test_img_n
  img_feat= np.load(img_feature_path+img_id+'EFB3.npy')
  img_tensor_val=tf.reshape(img_feat, (1, img_feat.shape[0], img_feat.shape[1]))
  dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
  captions = [['<start>', 0.0]]
  for i in range(max_len):
      all_cap=[]
      for cap in captions:
        sentence, score = cap
        if sentence.split()[-1] == '<end>':
          all_cap.append(cap)
          continue
        dec_input = tokenizer.texts_to_sequences([sentence])
        _, combined_mask, _ = create_masks(img_tensor_val, dec_input)
        predictions, c_att, s_att = transformer(img_tensor_val, tf.cast(dec_input, tf.int16), False, None, combined_mask, None)      
        predictions = predictions[: ,-1:, :]
        pred_arg = tf.cast(tf.argsort(predictions, axis=-1), tf.int32)
        pred_args = pred_arg.numpy()[0][0][-4:]
        for pred_id in pred_args:
          word = tokenizer.index_word[pred_id]
          caption = [sentence+ ' '+ word, score + predictions[0][0][pred_id].numpy()]
          all_cap.append(caption)
      ordered = sorted(all_cap, key = lambda tup: tup[1], reverse=True)
      captions = ordered[:4]
  return (captions, c_att, s_att)



def preprocess(img_):
    img_ = tf.io.read_file(img_path+img_)
    img_ = tf.image.decode_jpeg(img_, channels=3)
    img_ = tf.image.convert_image_dtype(img_, dtype=tf.float32)
    img_ = tf.image.resize(img_,(256, 256))
    return img_.numpy()



test_dict = load(open(data_path+'/data_preprocess/sc_test_dict.p',"rb"))
test_imgs = list(test_dict.keys())

test_img_id = np.random.randint(len(test_imgs))

pred_cap, s_att, c_att = feat_att(test_imgs[test_img_id])
print("Real Captions:",test_dict[test_imgs[test_img_id]])
print("Predicted Caption:",pred_cap[0][0])


img_ = preprocess(test_imgs[test_img_id])
c_att = np.resize(c_att, (10,10,512))
s_att = np.resize(s_att, (10,10,512))

c_avg = np.mean(c_att, axis=-1)
s_avg = np.mean(s_att, axis=-1)

c_hm = np.uint8((c_avg/np.max(c_avg))*255)
s_hm = np.uint8((s_avg/np.max(s_avg))*255)

jet = cm.get_cmap("jet")
jet_colors = jet(np.arange(256))[:, :3]

def heatmap(hm, img_):
    jet_hm = jet_colors[hm]
    jet_hm = array_to_img(jet_hm)
    jet_hm = jet_hm.resize((img_.shape[0], img_.shape[1]))
    jet_hm = img_to_array(jet_hm)
    hm_img = 0.003*jet_hm + 0.8*img_
    return hm_img
    
c_heatmap = heatmap(c_hm, img_)
s_heatmap = heatmap(s_hm, img_)


fig = plt.figure(figsize=(10, 10))

fig.add_subplot(1, 3, 1)
plt.imshow(img_)
plt.title('Original Image')
plt.axis('off')

fig.add_subplot(1, 3, 2)
plt.imshow(c_heatmap)
plt.title('Channel Attention')
plt.axis('off')

fig.add_subplot(1, 3, 3)
plt.imshow(s_heatmap)
plt.title('Spatial Attention')
plt.axis('off')