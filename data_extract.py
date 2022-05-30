#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 14 13:47:22 2022

@author: gaurav
"""

from zipfile import ZipFile
import tensorflow as tf
import json
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np


data_path = './SCAMET/'
dataset = data_path+'RSIC_datasets.zip'

all_data = ZipFile(dataset,'r')

sc_data = json.loads(all_data.open('modified-master/dataset_sydney_modified.json','r').read())
ucm_data = json.loads(all_data.open('modified-master/dataset_ucm_modified.json','r').read())
rsicd_data = json.loads(all_data.open('modified-master/dataset_rsicd_modified.json','r').read())


# data extacting and spliting 
def data_extract(data_dict):
    test_dict = {}
    train_imgs, train_capts, val_imgs, val_capts = [],[],[],[]
    for info in data_dict['images']:
        if info['split'] == 'train':
            for i in info['sentences']:
                capt = '<start> ' + i['raw'].lower().replace(',',' ').replace('-',' ').replace(';',' ').replace('. .',' .') + ' <end>'
                img_id = info['filename']
                train_capts.append(capt)
                train_imgs.append(img_id)
        
        elif info['split'] == 'val':
            for i in info['sentences']:
                capt = '<start> ' + i['raw'].lower().replace(',',' ').replace('-',' ').replace(';',' ').replace('. .',' .') + ' <end>'
                img_id = info['filename']
                val_capts.append(capt)
                val_imgs.append(img_id)
                
        elif info['split'] == 'test':
            for i in info['sentences']:
                capt = '<start> ' + i['raw'].lower().replace(',',' ').replace('-',' ').replace(';',' ').replace('. .',' .') + ' <end>'
                img_id = info['filename']
                if img_id in test_dict:
                    test_dict[img_id].append(capt)
                else:
                    test_dict[img_id]=[capt]
                    
    
    return (train_imgs, train_capts, val_imgs, val_capts, test_dict)
 
                
sc_train_imgs, sc_train_capts, sc_val_imgs, sc_val_capts, sc_test_dict = data_extract(sc_data)                
ucm_train_imgs, ucm_train_capts, ucm_val_imgs, ucm_val_capts, ucm_test_dict = data_extract(ucm_data)   
rsicd_train_imgs, rsicd_train_capts, rsicd_val_imgs, rsicd_val_capts, rsicd_test_dict = data_extract(rsicd_data)     
              
# removing less significant words from vocabulary
def vocab_(capt):
    word_freq={}
    for sent in capt:
        for word in sent.split():  
            if word not in word_freq:
                word_freq[word]=1
            else:
                word_freq[word]+=1
    vocab_words = [ w for w, f in word_freq.items() if word_freq[w]>1]
    print('original vocabulary: ', len(word_freq))
    print('truncated vocabulary: ', len(vocab_words))
    return vocab_words

sc_voc_words = vocab_(sc_train_capts)
ucm_voc_words = vocab_(ucm_train_capts)
rsicd_voc_words = vocab_(rsicd_train_capts)


# creating new caption lists
def new_capts(capts, vocab_words):
    capt_=[]
    for capt in capts:
        capt_.append(' '.join([word for word in capt.split() if word in vocab_words]))
    return capt_

#updated sydney-captions
sc_train_capts_ = new_capts(sc_train_capts, sc_voc_words)
sc_val_capts_ = new_capts(sc_val_capts, sc_voc_words)

#updated UCM-captions
ucm_train_capts_ = new_capts(ucm_train_capts, ucm_voc_words)
ucm_val_capts_ = new_capts(ucm_val_capts, ucm_voc_words)

#updated RSICD
rsicd_train_capts_ = new_capts(rsicd_train_capts, rsicd_voc_words)
rsicd_val_capts_ = new_capts(rsicd_val_capts, rsicd_voc_words)



# tokenizing training and validation captions
def tokenize(train_capt, val_capt):
    tokenizer = Tokenizer(oov_token='<unk>', filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
    tokenizer.fit_on_texts(train_capt)
    tokenizer.word_index['<pad>'] = 0
    tokenizer.index_word[0] = '<pad>'
    vocab_size=len(tokenizer.word_index)+1
    train_sequences = tokenizer.texts_to_sequences(train_capt)
    max_len = max(len(seq) for seq in train_sequences)
    train_pad_capts = pad_sequences(train_sequences, padding='post', maxlen=max_len)
    
    val_sequences = tokenizer.texts_to_sequences(val_capt)
    val_pad_capts = pad_sequences(val_sequences, padding='post', maxlen=max_len)
    
    print('Maximum caption len :', max_len)
    print('Vocabulary Size: ', vocab_size)
    return (tokenizer, train_pad_capts, val_pad_capts)
    
sc_tokenizer, sc_train_pad_capts, sc_val_pad_capts = tokenize(sc_train_capts_, sc_val_capts_)


pickle.dump(sc_train_imgs, open(data_path+'/data_preprocess/sc_train_imgs.p',"wb"))
pickle.dump(sc_train_pad_capts, open(data_path+'/data_preprocess/sc_train_capts.p',"wb"))
pickle.dump(sc_val_imgs, open(data_path+'/data_preprocess/sc_val_imgs.p',"wb"))
pickle.dump(sc_val_pad_capts, open(data_path+'/data_preprocess/sc_val_capts.p',"wb"))
pickle.dump(sc_tokenizer, open(data_path+'/data_preprocess/sc_tokenizer.p',"wb"))
pickle.dump(sc_test_dict, open(data_path+'/data_preprocess/sc_test_dict.p',"wb"))




def preprocess(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, dtype=tf.float32)
    img = tf.image.resize(img, (300, 300))
    #img = tf.image.resize(img, (224, 224))
    return img, image_path

# defining pre-trained Efficient B3 model
import efficientnet.tfkeras as efn
efnet = efn.EfficientNetB3(include_top=False, weights='imagenet', input_tensor=None,input_shape=(300,300,3))


sc_img_path = data_path+'/Sydney_captions/Images/'

'''
img_names = [img_path + info['filename'] for info in sc_data['images']]
for img_name in img_names:
    jpg_img = img_name[:-3]+"jpeg"
    tif_img = Image.open(img_name)
    out_img = tif_img.convert("RGB")
    out_img.save(jpg_img, "JPEG", quality=90)
'''

img_names_jpg = [sc_img_path + info['filename'][:-3]+"jpeg" for info in sc_data['images']]
image_dataset = tf.data.Dataset.from_tensor_slices(img_names_jpg)
image_dataset = image_dataset.map(preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(64)

#Save the features
sc_img_feature_path = data_path+'/Sydney_captions/Img_features/' 

for img, path in image_dataset:
    batch_features = efnet(img) # shape = (9,9,1408)
    batch_features = tf.reshape(batch_features, (batch_features.shape[0], -1, batch_features.shape[3])) #shape=(batch,81,1048)
    for bf, p in zip(batch_features, path):
        img_id = p.numpy().decode('utf-8').split('/')[-1]
        path_of_feature = sc_img_feature_path+img_id+'EFB3'
        np.save(path_of_feature, bf.numpy())



