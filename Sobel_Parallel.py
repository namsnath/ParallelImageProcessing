import numpy as np
from numpy import array
from scipy import misc
from PIL import Image
import pymp
import datetime
import math
from matplotlib.pyplot import imread
import warnings
warnings.filterwarnings("ignore")

def doSobelParallel(imageName, threads=2):
	print('Threads: ', threads)
	a = datetime.datetime.now()
	pymp.config.nested = True
	face = imread(imageName)
	print(face.shape)
	convx = array([
		[-1, 0, 1],
		[-2, 0, 2],
		[-1, 0, 1]]).flatten()
	l = face.shape[0]
	b = face.shape[1]
	#padded = np.zeros((l+2,b+2))
	padded = pymp.shared.array((l+2, b+2), dtype='uint8')
	i = None
	j = None
	with pymp.Parallel(threads) as p1:
		with pymp.Parallel(threads) as p2:
			for i in p1.range(0, l):
				for j in p2.range(0, b):
					padded[i+1][j+1] = face[i][j]


	resx = pymp.shared.array((l, b), dtype='uint8')
	i = None
	j = None
	with pymp.Parallel(threads) as p1:
		with pymp.Parallel(threads) as p2:
			for i in p1.range(1, l+1):
				for j in p2.range(1, b+1):
					reqdSubMatrix = padded[i-1:i+2, j-1:j+2].flatten()
					resx[i-1][j-1] = (convx.dot(reqdSubMatrix)) ** 2


	resy = pymp.shared.array((l, b), dtype='uint8')
	i = None
	j = None
	convy = array([
			[1, 2, 1],
			[0, 0, 0],
			[-1, -2, -1]
	]).flatten()
	with pymp.Parallel(threads) as p1:
		with pymp.Parallel(threads) as p2:
			for i in p1.range(1, l+1):
				for j in p2.range(1, b+1):
					reqdSubMatrix = padded[i-1:i+2, j-1:j+2].flatten()
					resy[i-1][j-1] = (convy.dot(reqdSubMatrix)) ** 2


	res2 = pymp.shared.array((l, b), dtype='uint8')
	with pymp.Parallel(threads) as p1:
		with pymp.Parallel(threads) as p2:
			for i in p1.range(0, l):
				for j in p2.range(0, b):
					res2[i][j] = int((resx[i-1][j-1]+resy[i-1][j-1]))
					if res2[i][j] > 15:
						res2[i][j] = 255

	img = Image.fromarray(res2)
	img.save('SobelEdge-Parallel-' + imageName)
	b = datetime.datetime.now()
	print("Time: "+str(b-a))
	print()


images = [
	'OtsuThresholding-Parallel-(256, 256)-image14.pgm', 
	'OtsuThresholding-Parallel-(512, 512)-image15.pgm', 
	'OtsuThresholding-Parallel-(1024, 1024)-image25.pgm',
	'OtsuThresholding-Parallel-(2048, 3072)-img0001.pgm'
]
for i in images:
	doSobelParallel(i, 2)
	doSobelParallel(i, 4)
	doSobelParallel(i, 8)
