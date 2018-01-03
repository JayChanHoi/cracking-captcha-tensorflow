import tensorflow as tf 
import numpy as np
import gen_captcha
import model_1
import string
import tflearn
import importlib
from termcolor import colored
import time
from sys import platform

if platform =="win32":
	print('the operatiing system is window , import colorama')
	from colorama import init
	init()
elif paltform == "darwin":
	print('the operating system us Mac OS , no colorama is need')

modle = model_1
batch_size = 45
capt_len = 5
iteration = 1000
characters = string.digits + string.ascii_letters
char_len = len(characters)
if __name__ == '__main__':
	start = time.time()
	# create placeholder for input
	X = tf.placeholder(dtype=tf.float32, shape = [None,60,160,1])
	Y = tf.placeholder(dtype=tf.float32, shape = [None,capt_len,char_len])
	l = tf.placeholder(dtype=tf.float32, shape = [1])
	# lr = tf.placeholder(dtype=tf.float32, shape = [None,1])
	# implement the model architecture
	output = modle.nn_architecture(X)
	# lr = l*e^(-(i)/100)
	# define the loss 
	cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=Y)
	cross_entropy = tf.reduce_mean(cross_entropy)
	cross_entropy_sum = tf.summary.scalar("cross_entropy",cross_entropy)
	# define optimizer
	optimizer = tf.train.AdamOptimizer(0.0001).minimize(cross_entropy)
	# correct prediction
	pred = output
	# define captcha text
	capt_text = Y
	# boolean return
	corr_pred = tf.equal(tf.argmax(pred,2),tf.argmax(capt_text,2))
	# transform boolean to number
	corr_pred = tf.cast(corr_pred,tf.float32)
	# define accuracy 
	accuracy = tf.reduce_mean(corr_pred)
	accuracy_sum = tf.summary.scalar("accuracy",accuracy)
	# define initializer
	init = tf.global_variables_initializer()
	# define saver
	saver = tf.train.Saver()
	# start trainning
	with tf.Session() as sess:
		# sess.run(init)
		writer = tf.summary.FileWriter("./captcha_graph_1", sess.graph)
		saver.restore(sess,'./my_captcha_model.ckpt')
		x,y = next(gen_captcha.capt_generation(batch_size,capt_len))
		_, loss, accu, Pred, loss_plot, accu_plot = sess.run([optimizer,cross_entropy,accuracy,pred,cross_entropy_sum,accuracy_sum],feed_dict={X:x,Y:y})
		writer.add_summary(loss_plot)
		writer.add_summary(accu_plot)
		init_pred_list = []
		init_real_list = []
		init_pred_vs_real = []
		for j in range(batch_size):
			init_pred_list.append(gen_captcha.decode(Pred[j]))
			init_real_list.append(gen_captcha.decode(y[j]))
			init_pred_item = list(init_pred_list[j])
			init_real_item = list(init_real_list[j])
			for k in range(len(init_pred_list[j])):
				if init_pred_item[k] == init_real_item[k]:
					init_real_item[k] = colored(init_real_item[k],'red')
					init_pred_item[k] = colored(init_pred_item[k],'red')
				init_pred_list[j] = "".join(init_pred_item)
				init_real_list[j] = "".join(init_real_item)
			init_pred_vs_real.append(init_pred_list[j] + '   ,  ' + init_real_list[j])
		init_Pred_vs_Real = "\n   ".join(init_pred_vs_real)
		print('initial loss:{0} , initial accuracy: {1}\nPrediction VS Real:\n   {2}'.format(loss,accu,init_Pred_vs_Real))

		# generate captcha for trainning
		for i in range(iteration):
			x,y = next(gen_captcha.capt_generation(batch_size,capt_len))
			_, loss, accu, Pred, loss_plot, accu_plot = sess.run([optimizer,cross_entropy,accuracy,pred,cross_entropy_sum,accuracy_sum],feed_dict={X:x,Y:y})
			writer.add_summary(loss_plot,i)
			writer.add_summary(accu_plot,i)
			if (i+1) % 10 == 0:
				print('step:{0} , loss:{1} , accuracy:{2}'.format(i+1,loss,accu))
			if (i+1) % 100 == 0:
				pred_list = []
				real_list = []
				pred_vs_real = []
				for j in range(batch_size):
					pred_list.append(gen_captcha.decode(Pred[j]))
					real_list.append(gen_captcha.decode(y[j]))
					pred_item = list(pred_list[j])
					real_item = list(real_list[j])
					for k in range(len(pred_list[j])):
						if pred_item[k] == real_item[k]:
							real_item[k] = colored(real_item[k],'red')
							pred_item[k] = colored(pred_item[k],'red')
						pred_list[j] = "".join(pred_item)
						real_list[j] = "".join(real_item)
					pred_vs_real.append(pred_list[j] + '   ,  ' + real_list[j])
				Pred_vs_Real = "\n   ".join(pred_vs_real)
				print('step:{0} checkpoint saved\nPrediction VS Real:\n   {1}'.format(i+1,Pred_vs_Real))
				saver.save(sess,'./captcha_model_step{0}.ckpt'.format(i+1))
			
		saver.save(sess,'./my_captcha_model.ckpt')
		print('Total Running Time : {0} hrs'.format((time.time() - start)/3600))




