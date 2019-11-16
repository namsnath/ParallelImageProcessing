import numpy as np
from numpy import array
from scipy import misc
from PIL import Image
import pymp
import datetime
from matplotlib.pyplot import imread


def doGaussianBlurParallel(imageName, threads=2):
	print('Threads: ', threads)
	a = datetime.datetime.now()
	pymp.config.nested = True
	inputImg = imread(imageName)

	print('Image shape: ', inputImg.shape)
	kernel = array([
					[1/16, 2/16, 1/16],
					[2/16, 4/16, 2/16],
					[1/16, 2/16, 1/16]
				]).flatten()
	l = inputImg.shape[0]
	b = inputImg.shape[1]

	paddedImg = pymp.shared.array((l+2, b+2), dtype='uint8')
	i = None
	j = None
	with pymp.Parallel(threads) as p1:
		with pymp.Parallel(threads) as p2:
			for i in p1.range(0, l):
				for j in p2.range(0, b):
					paddedImg[i+1][j+1] = inputImg[i][j]


	resultImg = pymp.shared.array((l, b), dtype='uint8')
	i = None
	j = None
	with pymp.Parallel(threads) as p1:
		with pymp.Parallel(threads) as p2:
			for i in p1.range(1, l+1):
				for j in p2.range(1, b+1):
					reqdSubMatrix = paddedImg[i-1:i+2, j-1:j+2].flatten()
					resultImg[i-1][j-1] = kernel.dot(reqdSubMatrix)


	img = Image.fromarray(resultImg)
	img.save('GaussianBlur-Parallel-' + str(inputImg.shape) + '-' + imageName)
	b = datetime.datetime.now()
	print('Time taken: ', b-a)
	print()


images = ['image14.pgm', 'image15.pgm', 'image25.pgm', 'img0001.pgm']
for i in images:
	doGaussianBlurParallel(i, 2)
	doGaussianBlurParallel(i, 4)
	doGaussianBlurParallel(i, 8)
