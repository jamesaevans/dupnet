import numpy as np
from PIL import Image
import glob, sys
from datetime import datetime

# Convert im_arr of RGB values to matrix of luminance values
def luma(im_arr):
	xmax, ymax, cols = im_arr.shape
	toRet = np.zeros( (xmax, ymax, 1) )
	for x in range(xmax):
		for y in range(ymax):
			# Source for formula: https://en.wikipedia.org/wiki/Luma_(video)
			pixel = im_arr[x][y]
			r = pixel[0]
			g = pixel[1]
			b = pixel[2]
			toRet[x][y] = 0.2126*r + 0.7152*g + 0.0722*b

	return toRet

# Formula source: https://en.wikipedia.org/wiki/Structural_similarity 
def SSIM(im_arr1, im_arr2):
	# Convert to luminance matrices
	lum_arr1 = luma(im_arr1)
	lum_arr2 = luma(im_arr2)

	# Compute summary statistics
	mu1 = np.average(lum_arr1)
	mu2 = np.average(lum_arr2)

	cov = np.cov(lum_arr1.flatten(), lum_arr2.flatten())[0][1]

	sigma1 = np.var(lum_arr1)
	sigma2 = np.var(lum_arr2)

	# Set constants
	L = 2**8 - 1
	k1 = 0.01
	k2 = 0.03
	c1 = (k1*L)**2.0
	c2 = (k2*L)**2.0

	# SSIM calc
	num = (2*mu1*mu2 + c1)*(2*cov + c2)
	denom = (mu1**2 + mu2**2 + c1)*(sigma1 + sigma2 + c2)

	return num/denom

# Compute EMD using Hungarian Algorithm
def emd(bins1, bins2):
	emd = []
	emd.append(0)
	for i in range(1,len(bins1)):
		if i in bins1:
			tmp1 = bins1[i]
		else:
			tmp1 = 0
		if i in bins2:
			tmp2 = bins2[i]
		else:
			tmp2 = 0
		emd.append(tmp1 + emd[-1] - tmp2)

	toRet = 0
	for term in emd:
		toRet += abs(term)

	return toRet


# Source: http://www.pyimagesearch.com/2014/07/14/3-ways-compare-histograms-using-opencv-python/
def colHist(im_arr1, im_arr2):
	xmax, ymax = im_arr1.shape
	histogram1 = dict()
	histogram2 = dict()

	for x in range(xmax):
		for y in range(ymax):
			tmp1 = im_arr1[x][y]
			tmp2 = im_arr2[x][y]
			if tmp1 in histogram1:
				histogram1[tmp1] += 1
			else:
				histogram1[tmp1] = 1
			if tmp2 in histogram2:
				histogram2[tmp2] += 1
			else:
				histogram2[tmp2] = 1

	return emd(histogram1, histogram2)

# Convert RGB to grayscale
def rgb2gray(rgb):
	return np.around(np.dot(rgb[...,:3], [0.299, 0.587, 0.114])).astype('int')

if __name__ == '__main__':
	train_dir =  sys.argv[1]
	baseline_dir = sys.argv[2]

	# hard-coded original images
	baselines_filenames = ['0 - altar_boy_s_001435.png',
                '1 - beaker_s_000604.png',
                '2 - bos_taurus_s_000507.png',
                '3 - car_train_s_000043.png',
                '4 - cichlid_s_000031.png',
                '5 - fog_s_000397.png',
                '6 - mcintosh_s_000643.png',
                '7 - phone_s_002161.png',
                '8 - stegosaurus_s_000125.png',
                '9 - willow_tree_s_000645.png'
                ]

	PHI = 0.78

	start = datetime.now()
	if len(sys.argv) != 2:
		print 'Usage: python comparisons.py (ssim | histogram)'
	elif sys.argv[1] == 'ssim':
		baselines = []
		for b in baselines_filenames:
			baselines.append(np.array(Image.open(baseline_dir + b)))

		num_samples = 0
		num_correct_1 = 0
		num_correct_2 = 0
		for i in range(10):
			for file in glob.glob(train_dir + str(i) + "/*.png"):
				probs = []
				for j in range(len(baselines)):
					tmpIm = Image.open(file)
					tmpArr = np.array(tmpIm)
					probs.append(SSIM(tmpArr, baselines[j]))
				guess = probs.index(max(probs))
				num_samples += 1
				if guess == i:
					num_correct_1 += 1
					num_correct_2 += 1
				else:
					sortProb = sorted(probs, reverse=True)
					if probs.index(sortProb[1]) == i:
						num_correct_2 += 1

			print '*'

		print float(num_correct_1)/(num_samples)
		print float(num_correct_2)/num_samples  
	elif sys.argv[1] == 'histogram':
		baselines = []
		for b in baselines_filenames:
			baselines.append(rgb2gray(np.array((Image.open(baseline_dir + b)))))

		num_samples = 0
		num_correct_1 = 0
		num_correct_2 = 0
		for i in range(10):
			for file in glob.glob(train_dir + str(i) + "/*.png"):
				probs = []
				for j in range(len(baselines)):
					tmpIm = Image.open(file)
					tmpArr = rgb2gray(np.array(tmpIm))
					probs.append(colHist(tmpArr, baselines[j]))
				guess = probs.index(min(probs))
				num_samples += 1
				if guess == i:
					num_correct_1 += 1
					num_correct_2 += 1
				else:
					sortProb = sorted(probs)
					if probs.index(sortProb[1]) == i:
						num_correct_2 += 1

			print '*'

		print float(num_correct_1)/(num_samples)
		print float(num_correct_2)/num_samples 
	else:
		print 'Usage: python comparisons.py (ssim | histogram)'

	end = datetime.now()
	print (end-start)





