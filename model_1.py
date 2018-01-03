import tensorflow as tf 
import numpy as np 
import string
import tflearn

capt_len = 5
epoch = 1000
characters = string.digits + string.ascii_letters
char_len = len(characters)

def conv_layer(size,prev_depth,depth,stride,input,name,pad,pool=True,res=True):
	w = tf.get_variable(shape = [size,size,prev_depth,depth], name = 'w{0}'.format(name), initializer = tf.contrib.layers.xavier_initializer())
	b = tf.get_variable(shape = [depth], name = 'b{0}'.format(name), initializer = tf.contrib.layers.xavier_initializer())
	conv_layer = tf.nn.conv2d(input, w, strides=[1,stride,stride,1], padding = pad)+b
	conv = tflearn.layers.normalization.batch_normalization(conv_layer,trainable=True,restore=True,reuse=False)
	if res:
		conv = tf.nn.relu(conv)
	if pool:
		conv = tf.nn.max_pool(conv, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')
	return conv,w,b

def residual_block(size,prev_depth,depth,i,pad,pool,res,shortcut,j):
	L1 = ['first_conv_{0}'.format(j),'first_{0}'.format(j),'first__{0}'.format(j)]
	L2 = ['second_conv_{0}'.format(j),'second_{0}'.format(j),'second__{0}'.format(j)]
	L1 = conv_layer(size[0],prev_depth,depth,1,i,'first_{0}'.format(j),pad[0],pool[0],res[0])
	L2 = conv_layer(size[1],depth,depth,1,L1[0],'second_{0}'.format(j),pad[1],pool[1],res[1])
	# shortcut residual
	if shortcut:
		L = ['w_s_{0}'.format(j),'b_s_{0}'.format(j)] 
		L[0] = tf.get_variable(shape = [1,1,prev_depth,depth],name = 'w_s_{0}'.format(j), initializer = tf.contrib.layers.xavier_initializer())
		L[1] = tf.get_variable(shape = [depth], name = 'b_s_{0}'.format(j), initializer = tf.contrib.layers.xavier_initializer())
		Input = tf.nn.conv2d(i, L[0], strides=[1,1,1,1], padding = 'SAME')+L[1]
		Input = tflearn.layers.normalization.batch_normalization(Input,trainable=True,restore=True,reuse=False)
	else:
		Input = i
		Input = tflearn.layers.normalization.batch_normalization(Input,trainable=True,restore=True,reuse=False)
	res = Input + L2[0]
	conv = tf.nn.relu(res)
	conv = tf.nn.max_pool(conv, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')
	return conv

def fc(flatten,i,input_size,output_size,activate,index,initializer):
	if flatten:
		flatten_1 = tf.contrib.layers.flatten(i)
		Input = flatten_1
	else:
		Input = i
	wc ,bc = 'w_{0}'.format(index), 'b_{0}'.format(index)
	wc = tf.get_variable(shape = [input_size,output_size], name = 'w_{0}'.format(index), initializer = initializer)
	bc = tf.get_variable(shape = [output_size], name = 'b_{0}'.format(index), initializer = initializer)
	fc = tf.matmul(Input,wc) + bc
	if activate:
		fc = tf.nn.relu(fc)
	return fc,wc,bc

def nn_architecture(X):
	# create first convolutional layer
	conv_1, _1, __1 = conv_layer(5,1,32,1,X,'conv1','SAME',True,True)

	# create second convolutional layer
	conv_2, _2, __2 = conv_layer(3,32,64,1,conv_1,'conv2','SAME',False,True)

	# create 1st residual layer
	conv_3 = residual_block([3,3],64,64,conv_2,['SAME','SAME'],[False,False],[True,False],False,1)

	# create 2nd residual layer
	conv_4 = residual_block([3,3],64,96,conv_3,['SAME','SAME'],[False,False],[True,False],True,2)

	# add one convolutional layer
	conv_5, _5, __5 = conv_layer(3,96,128,1,conv_4,'conv_5','SAME',False,True)

	# drop out with keep prob 0.5
	conv_5 = tf.nn.dropout(conv_5,0.5)

	# first fully-connected layer
	fc1, _fc1, __fc1 = fc(True,conv_5,20480,1024,True,1,tf.contrib.layers.xavier_initializer())

	# lastly, classicfication layer
	classifier, _, __ = fc(True, fc1, 1024, capt_len*char_len, False, 2, tf.contrib.layers.xavier_initializer())
	output = tf.reshape(classifier,[-1,capt_len,char_len])

	return output
