#1. import libraries
import numpy as numpy
import csv
import tensorflow as tf
#HR Predictor

#2. read in data
input_dict={}
with open('baseballdatabank/core/Batting.csv','r') as csvfile:

	readCSV=csv.reader(csvfile,delimiter=',')
	for row in readCSV:
		if(row[0]!='playerID'):
			if(int(row[1])>2000):
				if(row[0] in input_dict):
					if(float(int(row[11]))!=0):	
						input_dict.get(row[0]).append(row[11])
				else:
					if(float(int(row[11]))!=0):				
						input_dict[row[0]]=[row[11]]
print(input_dict)
#3. split features and labels				
inputList=list()
outputList=list()
for k in input_dict:
	size=len(input_dict.get(k))
	if(size>=4):
		l=list()
		
		outputList.append([input_dict.get(k)[size-1]])
		l.append(input_dict.get(k)[size-4])	
		l.append(input_dict.get(k)[size-3])
		l.append(input_dict.get(k)[size-2])
		inputList.append(l)		
numpy.random.seed(1)

#4. split training and testing (doesn't happen in this program due to a lack of data)
#5. split data into batches (doesn't happen in this program due to a lack of data) 

#make the neural network (using lowest level tensorflow api)
num_hidden_per_layer=20
num_out=1
num_in=3
num_epochs=15000
input_label=tf.placeholder(tf.float32, name='input')
output_label=tf.placeholder(tf.float32)

#make weights 
first_weight=tf.Variable(tf.random_uniform([num_in,num_hidden_per_layer],-1,1))
second_weight=tf.Variable(tf.random_uniform([num_hidden_per_layer,num_hidden_per_layer],-1,1))
third_weight=tf.Variable(tf.random_uniform([num_hidden_per_layer,num_hidden_per_layer],-1,1))
fourth_weight=tf.Variable(tf.random_uniform([num_hidden_per_layer,num_out],-1,1))

#each layer is a mathematical operation. In each line, I multiply weights by inputs and then plug that into an activation function. In math terms, I do y=activation(mx). We'll worry about what an activation function is and why its used later, the important part is that each layer does y=mx+b (m=weights, x=features, b=bias , bias is 0).
first_hidden_layer=tf.nn.softplus(tf.matmul(input_label,first_weight))
second_hidden_layer=tf.nn.softplus(tf.matmul(first_hidden_layer,second_weight))
third_hidden_layer=tf.nn.softplus(tf.matmul(second_hidden_layer,third_weight))
dropout=tf.layers.dropout(inputs=third_hidden_layer)
output_layer=tf.nn.softplus(tf.matmul(second_hidden_layer,fourth_weight),name="output_layer")

#used 4 layers because that's what I found worked best through experimentation

#note - one layers output is the next layers input.

#make the loss function
loss=tf.reduce_mean(abs(output_layer-output_label))
opt=tf.train.GradientDescentOptimizer(.001).minimize(loss)
save=tf.train.Saver(max_to_keep=1)
init=tf.initialize_all_variables()
#6. train the model
with tf.Session() as sess:
	sess.run(init)
	save.save(sess,"models/HRModels/model.ckpt",write_meta_graph=True)
	err=sess.run(loss, feed_dict={input_label:inputList, output_label:outputList})
	i=0
	for k in range(num_epochs):
		sess.run(opt, feed_dict={input_label:inputList, output_label:outputList})
		err=sess.run(loss, feed_dict={input_label:inputList, output_label:outputList})
		i=i+1
		if(i%10==0):
			print("Err:",err)
			print("I:",i) 
	save.save(sess,"models/HRModels/model.ckpt",write_meta_graph=True)
	print("Test HR:",sess.run(output_layer,feed_dict={input_label:[[16,9,15]]}))
	print("Total Loss:",err)
print("Done")
#7. evalutate the model
