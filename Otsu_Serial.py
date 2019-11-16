import numpy as np
from numpy import array
from scipy import misc
from PIL import Image
import datetime
from matplotlib.pyplot import imread

MAX_BRIGHTNESS = 255
GRAYLEVEL = 256

def doOtsuSerial(imageName):
	face = imread(imageName)
	image1 = face

	print(face.shape)
	y_size1 = face.shape[0]
	x_size1 = face.shape[1]

	image2 = np.empty((y_size1, x_size1))

	hist = [0] * GRAYLEVEL
	prob = [0.0] * GRAYLEVEL
	myu = [0.0] * GRAYLEVEL
	omega = [0.0] * GRAYLEVEL
	sigma = [0.0] * GRAYLEVEL


	def otsu_th():
		# /* Histogram generation */
		for y in range(0, y_size1):
			for x in range(0, x_size1):
				hist[image1[y][x]] += 1

		# /* calculation of probability density */
		for i in range(0, GRAYLEVEL):
			prob[i] = float(hist[i]) / (x_size1 * y_size1)
		
		# /* omega & myu generation */
		omega[0] = prob[0]
		myu[0] = 0.0  # /* 0.0 times prob[0] equals zero */
		for i in range(1, GRAYLEVEL):
			omega[i] = omega[i-1] + prob[i]
			myu[i] = myu[i-1] + i*prob[i]

		# Sigma Maximization. Sigma stands for inter-class variance and determines optimal threshold value.s
		threshold = 0
		max_sigma = 0.0
		for i in range(0, GRAYLEVEL-1):
			if (omega[i] != 0.0 and omega[i] != 1.0):
				sigma[i] = ((myu[GRAYLEVEL-1]*omega[i] - myu[i])
							** 2) / (omega[i]*(1.0 - omega[i]))
			else:
				sigma[i] = 0.0
			if (sigma[i] > max_sigma):
				max_sigma = sigma[i]
				threshold = i

		print("Threshold Value = " + str(threshold))

		# /* binarization output into image2 */
		x_size2 = x_size1
		y_size2 = y_size1
		for y in range(0, y_size2):
			for x in range(0, x_size2):
				if (image1[y][x] > threshold):
					image2[y][x] = MAX_BRIGHTNESS
				else:
					image2[y][x] = 0


	a = datetime.datetime.now()
	otsu_th()
	b = datetime.datetime.now()
	print("Time: " + str(b-a))
	img = Image.fromarray(image2)
	img = img.convert('L')
	img.save('OtsuThresholding-Serial-' + str(face.shape) + '-' + imageName)
	print()


images = ['image14.pgm', 'image15.pgm', 'image25.pgm', 'img0001.pgm']
for i in images:
	doOtsuSerial(i)
