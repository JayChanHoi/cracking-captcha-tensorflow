from captcha.image import ImageCaptcha
import string as string
import matplotlib.pyplot as plt
import random
import numpy as np

Image = ImageCaptcha()
# classes of captcha character choice
characters = string.digits + string.ascii_letters
# generate captcha for trainning 
def capt_generation(batch_size,capt_len):
	# capt_len define the length of our captcha
	# batch_size define how many sample we want to train in each optimzaiton
	char_len = len(characters)
	# define captcha height,width,depth
	capt_height = 60
	capt_width = 160
	capt_depth = 1
	# define input x and output y
	z = []
	x = np.zeros((batch_size,capt_height,capt_width,capt_depth))
	y = np.zeros((batch_size , capt_len , char_len))
	# put array format captcha into x as input
	for i in range(batch_size):
		random_capt = ''.join([characters[random.randrange(char_len)] for char in range(capt_len)])
		capt = Image.generate_image(random_capt).convert('L')
		img = np.array(capt.getdata())/255.0
		img = np.reshape(img,[60,160,capt_depth])
		z.append(capt)
		x[i] = img
		for j,ch in enumerate(random_capt):
			y[i,j,characters.find(ch)] = 1
	yield x,y

# decode output to text
def decode(y):
	y_list = np.argmax(y,1)
	return ''.join([characters[i] for i in y_list])

if __name__ == '__main__':
	x,y= next(capt_generation(64,4))
	i = random.randrange(len(x))
	print(x[i])
	print(len(x))
	print(y[i])
	print(decode(y[i]))
	# plt.imshow(z[i])
	plt.title(decode(y[i]))
	plt.show()

