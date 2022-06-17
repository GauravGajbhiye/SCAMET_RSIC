#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 14:52:41 2022

@author: gaurav
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from pickle import load
import tensorflow as tf
import time
import matplotlib.pyplot as plt
import numpy as np
#from My_XF import Mem_Transformer, create_masks
from Mem_XF import Mem_Transformer, create_masks

data_path = './SCAMET/'

#for Sydney Captions Dataset 

train_imgs = load(open(data_path+'/data_preprocess/sc_train_imgs.p',"rb"))
train_capts = load(open(data_path+'/data_preprocess/sc_train_capts.p',"rb"))
val_imgs = load(open(data_path+'/data_preprocess/sc_val_imgs.p',"rb"))
val_capts = load(open(data_path+'/data_preprocess/sc_val_capts.p',"rb"))
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
EMB_MAT = None
num_memory = 30 # None, 10, 20, 30, 40, 50

transformer = Mem_Transformer(num_memory, num_layers, d_model, EMB_MAT, num_heads, dff, input_vocab_size, target_vocab_size, pe_input=max_len, pe_target=target_vocab_size, rate=dropout_rate)

img_feature_path = data_path+'/Sydney_captions/Img_features/'  


def map_func(img_name, cap):
    img_id = img_name.decode('utf-8')
    #img_id = img_name.decode('utf-8').split('/')[-1]
    img_tensor = np.load(img_feature_path+img_id+'EFB3.npy')
    return img_tensor, cap
    
train_data = tf.data.Dataset.from_tensor_slices((train_imgs, train_capts))
train_data = train_data.map(lambda x1, x2: tf.numpy_function(map_func, [x1, x2], [tf.float32, tf.int32]), num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_data = train_data.shuffle(buffer_size).batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

val_data = tf.data.Dataset.from_tensor_slices((val_imgs, val_capts))
val_data = val_data.map(lambda x1, x2: tf.numpy_function(map_func, [x1, x2], [tf.float32, tf.int32]), num_parallel_calls=tf.data.experimental.AUTOTUNE)
val_data = val_data.shuffle(buffer_size).batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

del train_imgs, train_capts, val_imgs, val_capts


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

  def __init__(self, d_model, warmup_steps=45000):
    super(CustomSchedule, self).__init__()
    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)
    self.warmup_steps = warmup_steps

  def __call__(self, step):
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps**-1.5)

    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

learning_rate = CustomSchedule(d_model)

#Defining the Loss function
# using Adam Optimizer for the network
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

# custom-loss function

def loss_function(real, pred):
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss_ = loss_object(real, pred)

  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask
  #return tf.reduce_mean(loss_)
  return tf.reduce_sum(loss_)/tf.reduce_sum(mask)

def accuracy_function(real, pred):
  accuracies = tf.equal(real, tf.argmax(pred, axis=2, output_type=real.dtype))

  mask = tf.math.logical_not(tf.math.equal(real, 0))
  accuracies = tf.math.logical_and(mask, accuracies)

  accuracies = tf.cast(accuracies, dtype=tf.float32)
  mask = tf.cast(mask, dtype=tf.float32)
  
  return tf.reduce_sum(accuracies)/tf.reduce_sum(mask)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')
val_loss = tf.keras.metrics.Mean(name='val_loss')
val_accuracy = tf.keras.metrics.Mean(name='val_accuracy')



chkpt_path = data_path+'/Chk_Path/EN_3DE'

chkpt = tf.train.Checkpoint(transformer=transformer, optimizer=optimizer)
chkpt_manager = tf.train.CheckpointManager(chkpt, chkpt_path, max_to_keep=5)
if chkpt_manager.latest_checkpoint:
    print("Found a checkpoint")
    #chkpt.restore(chkpt_manager.latest_checkpoint)
    

def train_step(inp, tar, training):
  tar_inp = tar[:, :-1]
  tar_real = tar[:, 1:]
  _, combined_mask, _ = create_masks(inp, tar_inp)

  with tf.GradientTape() as tape:
    predictions, _, _ = transformer(inp, tar_inp, 
                                 training, 
                                 None, 
                                 combined_mask, 
                                 None)
    loss = loss_function(tar_real, predictions)

  gradients = tape.gradient(loss, transformer.trainable_variables)    
  #optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))
  optimizer.apply_gradients((grad, var) for (grad, var) in zip(gradients, transformer.trainable_variables) if grad is not None)
  train_loss(loss)
  train_accuracy(accuracy_function(tar_real, predictions))

def val_step(inp, tar, training):
  tar_inp = tar[:, :-1]
  tar_real = tar[:, 1:]
  _, combined_mask, _ = create_masks(inp, tar_inp)
  predictions, _, _ = transformer(inp, tar_inp, training, None, combined_mask, None)
  val_loss(loss_function(tar_real, predictions))
  val_accuracy(accuracy_function(tar_real, predictions))


#Training of custom Transformer for caption generation 
EPOCHS = 60
Val_Loss=[]
Train_Loss=[]
Val_Acc=[]
Train_Acc=[]
Epochs=[]
for epoch in range(EPOCHS):
  start = time.time()
  train_loss.reset_states()
  train_accuracy.reset_states()
  val_loss.reset_states()
  val_accuracy.reset_states()
  for (batch, (t_inp, t_tar)) in enumerate(train_data):
    train_step(t_inp, t_tar, training=True)
      #t_loss, t_acc = train_step(t_inp, t_tar, training=True)
  
  for (batch, (v_inp, v_tar)) in enumerate(val_data):
    val_step(v_inp, v_tar, training=False)
    #v_loss, v_acc = train_step(v_inp, v_tar, training=False)
    
  if (epoch + 1) % 5 == 0:
    chkpt_save_path = chkpt_manager.save()
    print ('Saving checkpoint for epoch {} at {}'.format(epoch+1, chkpt_save_path))

  print(f'Epoch:{epoch+1}, Train_loss:{train_loss.result():.4f}, Train_acc:{train_accuracy.result():.4f}, Val_loss:{val_loss.result():.4f}, Val_acc:{val_accuracy.result():.4f} ')
  Val_Loss.append(val_loss.result().numpy())
  Train_Loss.append(train_loss.result().numpy())
  Val_Acc.append(val_accuracy.result().numpy())
  Train_Acc.append(train_accuracy.result().numpy())
  Epochs.append(epoch+1)
  print ('Time taken for epoch: {} secs\n'.format(time.time() - start)) 


n_par = np.sum([np.prod(v.get_shape().as_list()) for v in transformer.trainable_variables])
print("Number of trainable parameters : ", n_par)

plt.plot(Epochs, Train_Acc, color='green', label=' Train_Acc ')
plt.plot(Epochs, Val_Acc, color='red',label=' Val_Acc ')
plt.plot(Epochs, Train_Loss, color='orange', label=' Train_loss ')
plt.plot(Epochs, Val_Loss, color='blue',label=' Val_loss ')
plt.legend()
plt.show()


