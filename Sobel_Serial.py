import numpy as np
from numpy import array
from matplotlib.pyplot import imread
from PIL import Image
import pymp
import datetime
import math
from matplotlib.pyplot import imread
import warnings
warnings.filterwarnings("ignore")

def doSobelSerial(imageName):
	a = datetime.datetime.now()
	pymp.config.nested = True
	face = imread(imageName)
	print(face.shape)
	convx = array([[-1, 0, 1],
				[-2, 0, 2],
				[-1, 0, 1]])
	l = face.shape[0]
	b = face.shape[1]
	#padded = np.zeros((l+2,b+2))
	padded = pymp.shared.array((l+2, b+2), dtype='uint8')
	i = None
	j = None

	for i in range(0, l):
		for j in range(0, b):
			padded[i+1][j+1] = face[i][j]


	resx = pymp.shared.array((l, b), dtype='uint8')
	i = None
	j = None

	for i in range(1, l+1):
		for j in range(1, b+1):
			resx[i-1][j-1] = (convx[0][0]*padded[i-1][j-1] + convx[0][1]*padded[i-1][j]+convx[0][2]*padded[i-1][j+1] +
							convx[1][0]*padded[i][j-1]+convx[1][1]*padded[i][j] + convx[1][2]*padded[i][j+1] +
							convx[2][0]*padded[i+1][j-1] + convx[2][1]*padded[i+1][j] + convx[2][2]*padded[i+1][j+1])

			resx[i-1][j-1] = (resx[i-1][j-1]**2)


	resy = pymp.shared.array((l, b), dtype='uint8')
	i = None
	j = None
	convy = [[1, 2, 1],
			[0, 0, 0],
			[-1, -2, -1]
			]

	for i in range(1, l+1):
		for j in range(1, b+1):
			resy[i-1][j-1] = (convy[0][0]*padded[i-1][j-1] + convy[0][1]*padded[i-1][j]+convy[0][2]*padded[i-1][j+1] +
							convy[1][0]*padded[i][j-1]+convy[1][1]*padded[i][j] + convy[1][2]*padded[i][j+1] +
							convy[2][0]*padded[i+1][j-1] + convy[2][1]*padded[i+1][j] + convy[2][2]*padded[i+1][j+1])

			resy[i-1][j-1] = (resy[i-1][j-1]**2)


	res2 = pymp.shared.array((l, b), dtype='uint8')

	for i in range(0, l):
		for j in range(0, b):
			res2[i][j] = int((resx[i-1][j-1]+resy[i-1][j-1]))
			if res2[i][j] > 15:
				res2[i][j] = 255


	img = Image.fromarray(res2)
	img.save('SobelEdge-Serial-' + imageName)
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
	doSobelSerial(i)
