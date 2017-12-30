import tensorflow as tf 
import numpy as np
import model
import gen_captcha
import string
import tflearn
from PIL import Image
import glob
import os
import operator
import matplotlib.pyplot as plt

batch_size = 10
capt_len = 5
characters = string.digits + string.ascii_letters
char_len = len(characters)
dir_path = "the directory path of your own capthca testing set"
if __name__ == '__main__':
	# create placeholder for input
	X = tf.placeholder(dtype=tf.float32, shape = [None,60,160,1])
	# implement the model architecture
	output = model.nn_architecture(X)
	# correct prediction
	pred = output
	# boolean return
	corr_pred = tf.equal(tf.argmax(pred,2),tf.argmax(capt_text,2))
	# transform boolean to number
	corr_pred = tf.cast(corr_pred,tf.float32)
	# define accuracy
	accuracy = tf.reduce_mean(corr_pred)
	# define initializer
	init = tf.global_variables_initializer()
	# define saver
	saver = tf.train.Saver()
	# start trainning
	with tf.Session() as sess:
		# sess.run(init)
		saver.restore(sess,'./my_captcha_model.ckpt')
		image_list = {}
		real_list = {}
		im_list = {}
		for filename in glob.glob(dir_path + '/*.png'):
			image = Image.open(filename)
			index = os.path.splitext(os.path.basename(filename))[0].split('capt')[0]
			basename = os.path.splitext(os.path.basename(filename))[0].split('capt')[1]
			I = image.convert('L')
			I = np.array(I.getdata())/255.0
			I = np.reshape(I,[60,160,1])
			im_list[str(index)] = image
			image_list[str(index)] = I
			real_list[str(index)] = basename
		the_list = [image_list.get(str(i)) for i in range(len(image_list))]
		the_real_list = [real_list.get(str(i)) for i in range(len(real_list))]
		the_im_list = [im_list.get(str(i)) for i in range(len(image_list))]
		x = the_list
		Pred , accu = sess.run([pred,accuracy],feed_dict={X:x})
		print(accu)
		pred_list = []
		for i in range(30):
			pred_list.append(gen_captcha.decode(Pred[i]))
		print('prediction: {0}'.format(pred_list))
		print('real: {0}'.format(the_real_list))
		for i in range(30):
			plt.imshow(the_im_list[i])
			plt.title('real: {0} \n prediction:{1}'.format(the_real_list[i],pred_list[i]))
			plt.savefig(f'captcha_{i}_"{the_real_list[i]}".png')
			
		# saver.save(sess,'./my_captcha_model.ckpt')