import numpy as numpy
import csv
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
					if(float(int(row[6]))!=0):	
						input_dict.get(row[0]).append((int(row[8]))/float(int(row[6])))
				else:
					if(float(int(row[6]))!=0):				
						input_dict[row[0]]=[int(row[8])/float(int(row[6]))]			
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
synapse=2*numpy.random.random((3,1))-1
synapse2=2*numpy.random.random((1,3))-1
print("inputs:",inputs.shape)
print("ouputs:",outputs.shape)

print(inputs.shape)
print("-----------------------")
print(outputs.shape)
#for i in range(200):
	#l1=sigmoid(numpy.dot(inputs,synapse))
#	l2=sigmoid(numpy.dot(l1,synapse2))
#	l2Error=outputs-l2
#	l2Delta=l2Error*sigmoidDeriv(l2)
#	l1Error=l2Delta.dot(synapse2.T)	
#	l1Delta= l1Error * sigmoidDeriv(l1)
#	synapse += inputs.T.dot(l1Delta)
#	synapse2 += l1.T.dot(l2Delta)
#	if(i%1000==0):
#		print("L1Error:",l1Error)
#		print("L2Error:",l2Error)
synapse=2*numpy.random.random((3,1))-1
print(inputs.shape)
for i in range(50000):
	l1=sigmoid(numpy.dot(inputs,synapse))
	l1Error=outputs-l1	
	l1Delta= l1Error * sigmoidDeriv(l1)
	synapse += inputs.T.dot(l1Delta)
print ("Error:")
print(l1Error)
print ("Output")
print (synapse)
print("Prediction:",sigmoid(numpy.dot(numpy.array([[.269,.265,.285]]), synapse)))
