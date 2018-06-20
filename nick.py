import numpy as numpy
import csv
import tensorflow as tf
#import codecs
e=2.7182818284
def unbounded(x):
	return e**x
def sigmoid(x):
	return 1/(1+numpy.exp(-x))

def sigmoidDeriv(x):
	return x*(1-x)
#Ian Kinsler
inputs1=numpy.array([[.280,.249,.218],[.169,.289,.273],[.287,.265,.268],[.256,.259,.261],[.261,.227,.277],[.287,.299,.315],[.263,.263,.246],[.263,.246,.251],[.291,.291,.319],[.273,.330,.243],[.288,.276,.302],[.278,.286,.294],[.255,.314,.326],[.286,.278,.292],[.255,.297,.284],[.313,.331,.316],[.209,.279,.267],[.243,.247,.247],[.301,.307,.287],[.270,.297,.266],[.246,.257,.319],[.260,.291,.280],[.272,.272,.271],[.266,.285,.305],
[.259,.261,.271],[.287,.170,.213],[.267,.301,.348],[.263,.246,.251],[.268,.267,.278],
[.259,.255,.285],[.300,.287,.284],[.202,.226,.217],[.277,.279,.254],[.259,.255,.285],
[.289,.264,.263],[.231,.231,.260],[.278,.303,.291],[.263,.263,.246],[.275,.296,.288],
[.248,.224,.239],[.234,.210,.202],[.303,.300,.255],[.272,.276,.272],[.184,.208,.259]])
outputs1=numpy.array([[.303,.273,.276,.264,.268,.306,.276,.263,.264,.319,.307,.259,.320,.273,.270,.249,.293,.247,.300,
.318,.300,.292,.241,.268,.204,.288,.310,.263,.249,.272,.297,.270,.303,.272,.295,.259,.304,.276,.236,
.193,.221,.255,.232,.238]]).T
outputs=numpy.array([[0]]).T
inputs=numpy.array([[0,0,0]])
#numpy.random.seed(1)
input_dict={}
with open('../baseballdatabank/core/Batting.csv','r') as csvfile:

	readCSV=csv.reader(csvfile,delimiter=',')#codecs.open('baseballdatabank/core/Batting.csv','rb','utf-8')
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
inputs=numpy.asarray(inputList)

outputs=numpy.asarray(outputList)

#hyperparams
num_hidden_per_layer=20
num_out=1
num_in=3
num_epochs=15000
input_label=tf.placeholder(tf.float32)
output_label=tf.placeholder(tf.float32)

first_weight=tf.Variable(tf.random_uniform([num_in,num_hidden_per_layer],-1,1))
second_weight=tf.Variable(tf.random_uniform([num_hidden_per_layer,num_hidden_per_layer],-1,1))
third_weight=tf.Variable(tf.random_uniform([num_hidden_per_layer,num_hidden_per_layer],-1,1))
fourth_weight=tf.Variable(tf.random_uniform([num_hidden_per_layer,num_out],-1,1))

first_hidden_layer=tf.nn.softplus(tf.matmul(input_label,first_weight))
second_hidden_layer=tf.nn.softplus(tf.matmul(first_hidden_layer,second_weight))
third_hidden_layer=tf.nn.softplus(tf.matmul(second_hidden_layer,third_weight))
dropout=tf.layers.dropout(inputs=third_hidden_layer)
output_layer=tf.nn.softplus(tf.matmul(second_hidden_layer,fourth_weight))

loss=tf.reduce_mean(abs(output_layer-output_label))
opt=tf.train.GradientDescentOptimizer(.001).minimize(loss)

init=tf.initialize_all_variables()

with tf.Session() as sess:
	sess.run(init)
	err=sess.run(loss, feed_dict={input_label:inputs, output_label:outputs})
	i=0
	for k in range(num_epochs):
		sess.run(opt, feed_dict={input_label:inputs, output_label:outputs})
		err=sess.run(loss, feed_dict={input_label:inputs, output_label:outputs})
		i=i+1
		if(i%10==0):
			print("Err:",err)
			print("I:",i) 
	print("Test HR:",sess.run(output_layer,feed_dict={input_label:[[10,10,10]]}))
	print("Total Loss:",err)
print("Done")



