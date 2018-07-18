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

baseball=pd.read_csv("../../baseballdatabank/core/Pitching.csv",sep=',')

def preproccessDataset(dataset):
	feature_dict={};
	cat=dataset['ERA']
	names=dataset['playerID']
	#cat=cat.tail(2000)
	#names=names.tail(2000)
	count=0;
	for i in cat:
		
		if names[count] in feature_dict:
			feature_dict.get(names[count]).append(cat[count])
		else:
			feature_dict[names[count]]=[cat[count]]
		count=count+1

	features=[]
	labels=[]
	for k in feature_dict:
		size=len(feature_dict.get(k))
		if(size>=4):
			l=[]
			labels.append([feature_dict.get(k)[size-1]])
			l.append(feature_dict.get(k)[size-4])
			l.append(feature_dict.get(k)[size-3])
			l.append(feature_dict.get(k)[size-2])
			features.append(l)
	return features,labels

features,labels=preproccessDataset(baseball)
print(len(labels))
def input_fn(batch_size,features,labels,numEpochs):
	#features={key:np.array(value) for key,value in dict(features).items()}
	ds = Dataset.from_tensor_slices((features,labels))
	ds = ds.batch(batch_size).repeat(numEpochs)
	features,labels=ds.make_one_shot_iterator().get_next()
	return features,labels


inputs=tf.placeholder(dtype=tf.float32,shape=(None,3))
ls=tf.placeholder(dtype=tf.float32)
print(inputs.shape)
dense1=tf.layers.dense(inputs=inputs,units=70,activation=tf.nn.relu)
outputs=tf.layers.dense(inputs=dense1,units=70,activation=tf.nn.relu)
loss=tf.losses.absolute_difference(ls,outputs)
opt=tf.train.GradientDescentOptimizer(learning_rate=.001,name="opt").minimize(loss)
#opt=tf.contrib.estimator.clip_gradients_by_norm(opt,5.0)
init=tf.global_variables_initializer()
with tf.Session() as sess:
	sess.run(init)
	for i in range(2000):
		feature,label=input_fn(batch_size=5,features=features,labels=labels, numEpochs=1)
		#print("Feature:",feature.eval())
		sess.run(opt,feed_dict={inputs:feature.eval(),ls:label.eval()})
		print(sess.run(loss,feed_dict={inputs:feature.eval(),ls:label.eval()}))

	
