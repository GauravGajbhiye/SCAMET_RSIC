#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 14:41:19 2022

@author: gaurav
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from pickle import load
import tensorflow as tf
import numpy as np
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

transformer = Mem_Transformer(num_memory, num_layers, d_model, EMB_MAT, num_heads, dff, input_vocab_size, target_vocab_size, pe_input=max_len, pe_target=target_vocab_size, rate=dropout_rate)

img_feature_path = data_path+'/Sydney_captions/Img_features/'

chkpt_path = data_path+'/Chk_Path/EN_3DE'

chkpt = tf.train.Checkpoint(transformer=transformer)
chkpt_manager = tf.train.CheckpointManager(chkpt, chkpt_path, max_to_keep=5)
chkpt.restore(chkpt_manager.latest_checkpoint)


def feat_evaluate(test_img_n):
  img_id = test_img_n.split('/')[-1]
  img_feat= np.load(img_feature_path+img_id+'.npy')
  img_tensor_val=tf.reshape(img_feat, (1, img_feat.shape[0], img_feat.shape[1]))
  dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
  result = []
  for i in range(max_len):
      _, combined_mask, _ = create_masks(img_tensor_val, dec_input)
      predictions, attention_weights = transformer(img_tensor_val, dec_input, False, None, combined_mask, None)      
      predictions = predictions[: ,-1:, :]
      predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
      pred_id_int = predicted_id.numpy()[0][0]
      result.append(tokenizer.index_word[pred_id_int])
      if tokenizer.index_word[pred_id_int] == '<end>':
          return result
      dec_input = tf.concat([dec_input, predicted_id], axis=-1)
  return result

#beam search algorithm
def bs_evaluate(test_img_n):
  img_id = test_img_n.split('/')[-1]
  img_feat= np.load(img_feature_path+img_id+'.npy')
  img_tensor_val=tf.reshape(img_feat, (1, img_feat.shape[0], img_feat.shape[1]))
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
        predictions, img_att = transformer(img_tensor_val, tf.cast(dec_input, tf.int16), False, None, combined_mask, None)      
        predictions = predictions[: ,-1:, :]
        pred_arg = tf.cast(tf.argsort(predictions, axis=-1), tf.int32)
        pred_args = pred_arg.numpy()[0][0][-3:]
        for pred_id in pred_args:
          word = tokenizer.index_word[pred_id]
          caption = [sentence+ ' '+ word, score + predictions[0][0][pred_id].numpy()]
          all_cap.append(caption)
      ordered = sorted(all_cap, key = lambda tup: tup[1], reverse=True)
      captions = ordered[:3]
  return captions


test_dict = load(open(data_path+'/data_preprocess/sc_test_dict.p',"rb"))
test_imgs = list(test_dict.keys())


import json
# storing the json file for predicted ana actual report.
actual_= []
pred = []
pred_1 = []
pred_2 = []
pred_3 = []

img_id = 0
for img in test_imgs:
    pred_bs_caption = bs_evaluate(img)
    pred_capt = '<start> '+' '.join(feat_evaluate(img))
    actual_.append({"image_id":img_id, "caption":test_dict[img]})
    pred.append({"image_id":img_id,"caption":pred_capt})
    pred_1.append({"image_id":img_id,"caption":pred_bs_caption[0][0]})
    pred_2.append({"image_id":img_id,"caption":pred_bs_caption[1][0]})
    pred_3.append({"image_id":img_id,"caption":pred_bs_caption[2][0]})

    img_id+=1
    #print('-caption completed- :', img_id)


coco_actual_cap={'info': {
            'description': None,
            'url': None,
            'version': None,
            'year': None,
            'contributor': None,
            'date_created': None,
        },
        'images':[
            {
                'license':None,
                
                'file_name':None,
                'id':image_id,
                'width':None,
                'date_captured':None,
                'height':None
            }
            for image_id in range(len(actual_))
        ],
        'licenses':[
        ],
        'type':'captions',
        'annotations':[
            {
                'image_id':actual_[i]["image_id"],
                'id':j,
                'caption':actual_[i]["caption"][j]
            }
            for i in range(len(actual_)) for j in range(len(actual_[i]["caption"]))
        ]
    }



with open("/home/gaurav/TF2/F30K/coco-caption/annotations/SC_actual_test_cap.json","w", encoding='utf-8') as AC:
     print(str(json.dump(coco_actual_cap, AC)))
    
with open("/home/gaurav/TF2/F30K/coco-caption/predictions/SC_pred_test_cap.json","w") as PC:
    json.dump(pred, PC)   
    
with open("/home/gaurav/TF2/F30K/coco-caption/predictions/SC_bs1_test_cap.json","w") as PC:
    json.dump(pred_1, PC)    
    
with open("/home/gaurav/TF2/F30K/coco-caption/predictions/SC_bs2_test_cap.json","w") as PC:
    json.dump(pred_2, PC) 

with open("/home/gaurav/TF2/F30K/coco-caption/predictions/SC_bs3_test_cap.json","w") as PC:
    json.dump(pred_3, PC) 