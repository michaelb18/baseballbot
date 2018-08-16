#1. import libraries
from __future__ import print_function

import glob
import math
import os

from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset
tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

#2. read in dataset
baseball=pd.read_csv("baseballdatabank/core/Batting.csv",sep=',')
baseball=baseball.dropna(axis=0, how="any")

#3. split features and labels
def preproccessDataset(dataset):
	feature_dict={};
	cat=np.asarray(dataset['R'])
	ab=np.asarray(dataset['AB'])
	names=np.asarray(dataset['playerID'])
	years=np.asarray(dataset['yearID'])
	count=0;
	for i in cat:
		if years[count] >= 2000:
			if names[count] in feature_dict:
				feature_dict.get(names[count]).append(cat[count])
				feature_dict.get(names[count]).append(ab[count])
			else:
				feature_dict[names[count]]=[cat[count]]
				feature_dict.get(names[count]).append(ab[count])
		count=count+1

	features=[]
	labels=[]
	for k in feature_dict:
		size=len(feature_dict.get(k))
		if(size>=7):
			l=[]
			labels.append([feature_dict.get(k)[size-2]])
			l.append(feature_dict.get(k)[size-8])
			l.append(feature_dict.get(k)[size-7])
			l.append(feature_dict.get(k)[size-6])			
			l.append(feature_dict.get(k)[size-5])
			l.append(feature_dict.get(k)[size-4])
			l.append(feature_dict.get(k)[size-3])
			features.append(l)
	return features,labels

features,labels=preproccessDataset(baseball)
print(len(labels))

#4. split training and testing (doesn't happen in this program due to a lack of data).
#5. split data into batches (doesn't happen in this program due to a lack of data).
#note: not used, using full gradient decenst due to a lack of data
def input_fn(batch_size,features,labels,numEpochs):
	#features={key:np.array(value) for key,value in dict(features).items()}
	ds = Dataset.from_tensor_slices((features,labels))
	ds = ds.batch(batch_size).repeat(numEpochs)
	features,labels=ds.make_one_shot_iterator().get_next()
	return features,labels

#6. train the model
inputs=tf.placeholder(dtype=tf.float32,shape=(None,6),name='inputs')
ls=tf.placeholder(dtype=tf.float32)
step=tf.Variable(dtype=tf.int32, initial_value=0)
print(inputs.shape)

#make the neural network's structure (using low level tf api)
dense1=tf.layers.dense(inputs=inputs,units=15,activation=tf.nn.relu,name='dense1')
#dense2=tf.layers.dense(inputs=dense1,units=15,activation=tf.nn.relu,name='dense2')
#dense3=tf.layers.dense(inputs=dense2,units=15,activation=tf.nn.relu,name='dense3')
#dense4=tf.layers.dense(inputs=dense3,units=15,activation=tf.nn.relu,name='dense4')
output_layer=tf.layers.dense(inputs=dense1,units=1,activation=tf.nn.relu,name='output_layer')
step=tf.Variable(0,trainable=False,name="step")
num_hidden_per_layer=35
num_out=1
num_in=3
num_epochs=130

#make loss function and gradient descent
loss=tf.losses.absolute_difference(ls,output_layer)
opt=tf.train.GradientDescentOptimizer(learning_rate=.001,name="opt").minimize(loss,global_step=step)

init=tf.global_variables_initializer()
save=tf.train.Saver(max_to_keep=7)
with tf.Session() as sess:
	sess.run(init)
	save.save(sess,"models/RModels/model.ckpt",global_step=step.eval(),write_meta_graph=True)
	#save=tf.train.import_meta_graph('models/ERAModels/model.ckpt-1000.meta')
	#save.restore(sess,tf.train.latest_checkpoint("./models/ERAModels"))
	
	for i in range(1000):
		#feature,label=input_fn(batch_size=150,features=features,labels=labels, numEpochs=1)
		sess.run(opt,feed_dict={inputs:features,ls:labels})
		step=tf.add(step,1)
		if(i%50==0):
			print("Saving model at step:",i,". Loss:",sess.run(loss,feed_dict={inputs:features,ls:labels}))
			#save.save(sess,"models/ERAModels/model.ckpt",global_step=step.eval(),write_meta_graph=False)
		
	save.save(sess,"models/RModels/model.ckpt",global_step=step.eval(),write_meta_graph=True)#,write_meta_graph=False
#7. train the model
