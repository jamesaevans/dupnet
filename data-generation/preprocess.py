import numpy as np
from PIL import Image
from sets import Set
from PIL import ImageEnhance, ImageFilter
import random

def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

# Return a random number between minScale and maxScale
def getScaledFactor(minScale, maxScale):
    factor = random.random()*2
    factor = min(maxScale, factor)
    factor = max(minScale, factor)
    return factor


# To-do: download automatically if data-sets not present
def extract_from_CIFAR(numClasses):
  """ Read and save examples from CIFAR100 data files.
      Will save numClasses images, each from a different superclass.

      Note: data sets must be pre-downloaded and stored in the
      same directory as this script.
  """
  images = unpickle('train')

  # these are 'coarse' classes
  seenClasses = Set()

  k = 0
  while(len(seenClasses) < numClasses):
    thisClass = images["coarse_labels"][k]
    if  thisClass not in seenClasses:
      red = images["data"][k][0:1024]
      blue = images["data"][k][2048:3072]
      green = images["data"][k][1024:2048]

      # pixels flattened in row-major order
      pixels = np.zeros((32, 32, 3), dtype=np.uint8)
      
      for i in range(0, 32):
        for j in range(0, 32):
          idx = i*32+j
          pixels[i,j] = [red[idx], green[idx], blue[idx]]

      img = Image.fromarray(pixels, 'RGB')
      img.save(images["filenames"][k])
      seenClasses.add(thisClass)

    k = k + 1


# Partition the image represented by im_arr into around k continuous subspaces.
# Transition probability is prob.
def partitionImage(im_arr, k, prob):
	dims = im_arr.shape
	partitions = np.zeros( (dims[0],dims[1]) )

	numPixels = dims[0]*dims[1]

	# Start partition 1 at a random point
	curx = random.randint(0, dims[0]-1)
	cury = random.randint(0, dims[1]-1)
	frontier = set() # nodes to be visited next
	frontier.add((curx, cury))
	partitions[curx][cury] = 1

	# Expore all pixels in the image/matrix
	curPartition = 1
	numVisited = 1
	numThisCluster = 1
	while numVisited < numPixels:
		if len(frontier) > 0 and numThisCluster < numPixels/float(k) :
			curx, cury = frontier.pop()
			if curx+1 < dims[0]:
				if random.random() < prob and partitions[curx+1][cury] == 0:
					partitions[curx+1][cury] = partitions[curx][cury]
					numVisited += 1
					frontier.add((curx+1, cury))
					numThisCluster += 1

			if cury+1 < dims[1] and partitions[curx][cury+1] == 0:
				if random.random() < prob:
					partitions[curx][cury+1] = partitions[curx][cury]
					numVisited += 1
					frontier.add((curx, cury+1))
					numThisCluster += 1

			if curx-1 >= 0 and partitions[curx-1][cury] == 0:
				if random.random() < prob:
					partitions[curx-1][cury] = partitions[curx][cury]
					numVisited += 1
					frontier.add((curx-1, cury))
					numThisCluster += 1

			if cury-1 >= 0 and partitions[curx][cury-1] == 0:
				if random.random() < prob:
					partitions[curx][cury-1] = partitions[curx][cury]
					numVisited += 1
					frontier.add((curx, cury-1))
					numThisCluster += 1

		else:
			# randomly select a new vertex and add it to frontier
			curPartition += 1
			while partitions[curx][cury] != 0:
				curx = random.randint(0, dims[0]-1)
				cury = random.randint(0, dims[1]-1)
			partitions[curx][cury] = curPartition
			numVisited += 1
			numThisCluster = 1
			frontier.add((curx, cury))

	return partitions


# Returns a list of tuples (x, y) whose values in arr is k
def which(arr, k):
	toRet = []
	xmax, ymax = arr.shape
	for x in range(xmax):
		for y in range(ymax):
			if arr[x][y] == k:
				if (x,y) not in toRet:
					toRet.append((x,y))
	return toRet

# Model each partition as an independent multivariate Gaussian distribution
def l2transform(im_arr, partitions, k):
	xmax, ymax, cols = im_arr.shape
	toRet = np.zeros( (xmax, ymax, cols) )

	# Iterate over each cluster (first cluster is 1)
	for i in range(1,k+1):
		indices = which(partitions, i)
		if len(indices) < 5:
			continue

		# Sort pairs by x coord then y coord
		sorted(indices, key=lambda el: (el[0], el[1])) 
		numEls = len(indices)

		# set covariance to be inversely proportional to Manhattan distance
		# between pixels
		cov = a = np.zeros((numEls, numEls))
		np.fill_diagonal(cov,40) # variance of each pixel is 1
		for (x1,y1) in indices:
			for (x2,y2) in indices:
				if (x1,y1) != (x2,y2):
					i1 = indices.index((x1,y1))
					i2 = indices.index((x2,y2))
					cov[i1][i2] = float(10)/(2*(abs(x1-x2) + abs(y1-y2)))

		# A little hacky: create separate distribution for each color channel
		# Covariance matrix will not change
		for j in range(0,cols):
			mu = [im_arr[x][y][j] for (x,y) in indices]
			transformed = np.random.multivariate_normal(mu, cov, 1)
			for (x,y) in indices:
				itemp = indices.index((x,y))
				toRet[x][y][j] = transformed[0][itemp]



	return Image.fromarray(toRet.astype('uint8'))

def manipulate_image(image):
  """Mutate input image by randomly applying the following transformation
     with randomly selected arguments. 

     Transformations:
        - Brightness transform
        - Contrast transform
        - Sharpness transform
        - Color transform
        - Rotation (in 90 degree increments)

    Returns:
      Manipulated image
  """

  MIN_SCALE_FACTOR = 0.25
  MAX_SCALE_FACTOR = 1.75
  PROB_THRESHOLD = 3

  # Build up manipulated image
  activeImage = image.copy()

  # im_arr = np.array(activeImage)
  # if (im_arr.shape[0] != 32):
  #   print image.shape

  # Apply brightness transformation
  if random.randint(1, 10) <= PROB_THRESHOLD:
    factor = getScaledFactor(MIN_SCALE_FACTOR, MAX_SCALE_FACTOR)
    enhancer = ImageEnhance.Brightness(image)
    activeImage = enhancer.enhance(factor)

  # Apply Contrast transformation
  if random.randint(1, 10) <= PROB_THRESHOLD:
    factor = getScaledFactor(MIN_SCALE_FACTOR, MAX_SCALE_FACTOR)
    enhancer = ImageEnhance.Contrast(image)
    activeImage = enhancer.enhance(factor)

  # Apply Sharpness transformation
  if random.randint(1, 10) <= PROB_THRESHOLD:
    factor = getScaledFactor(MIN_SCALE_FACTOR, MAX_SCALE_FACTOR)
    enhancer = ImageEnhance.Sharpness(image)
    activeImage = enhancer.enhance(factor)

  # Apply Color transformation
  if random.randint(1, 10) <= PROB_THRESHOLD:
    factor = getScaledFactor(MIN_SCALE_FACTOR, MAX_SCALE_FACTOR)
    enhancer = ImageEnhance.Color(image)
    activeImage = enhancer.enhance(factor)

  # Rotates image either 90, 180, or 270 degrees
  if random.randint(1, 10) <= PROB_THRESHOLD:
    randVal = random.random()
    angle = 0
    if randVal < 0.33:
      angle = 90
    elif randVal < 0.66:
      angle = 180
    else:
      angle = 270
    activeImage = activeImage.rotate(angle)

  # Apply Gaussian blur transformation
  if random.randint(1, 10) <= PROB_THRESHOLD:
  	window = 5
  	if random.randint(1, 10) <= 5:
  		window = 3
  	activeImage = activeImage.filter(ImageFilter.MinFilter(window))

  # Apply Gaussian noise
  if random.randint(1, 10) <= PROB_THRESHOLD:
  	im_arr = np.array(activeImage)
  	noisy = im_arr + 10*np.random.randn(*im_arr.shape)
  	activeImage = Image.fromarray(noisy.astype('uint8'))

  # Compress and embed in larger matrix
  if random.randint(1, 10) <= PROB_THRESHOLD:
    im_arr = np.array(activeImage)
    wall = np.zeros((32,32,3))
    activeImage.thumbnail((24,24,3), Image.ANTIALIAS)
    im_arr = np.array(activeImage)
    wall[0:im_arr.shape[0], 0:im_arr.shape[1]] = im_arr
    activeImage = Image.fromarray(wall.astype('uint8'))

  return activeImage

def extract_all_from_CIFAR(courseLabels, fineLabels):
  images = unpickle('train')
  base_dir = './img/semantic/'
  NUM_IMAGES = 60000

  for k in range(0, NUM_IMAGES):
      course = images["coarse_labels"][k]
      fine = images["fine_labels"][k]
      
      # If in one of our classes, extract image
      if course in courseLabels or fine in fineLabels:
        red = images["data"][k][0:1024]
        blue = images["data"][k][2048:3072]
        green = images["data"][k][1024:2048]

        # pixels flattened in row-major order
        pixels = np.zeros((32, 32, 3), dtype=np.uint8)
        
        for i in range(0, 32):
          for j in range(0, 32):
            idx = i*32+j
            pixels[i,j] = [red[idx], green[idx], blue[idx]]

        img = Image.fromarray(pixels, 'RGB')

        if course in courseLabels:
          img.save(base_dir + 'course/' + str(course) + '/' + str(k) + '.png')

        if fine in fineLabels:
          img.save(base_dir + 'fine/' + str(fine) + '/' + str(k) + '.png')


def save_from_CIFAR(marker, num_images):
  images = unpickle('train')
  base_dir = './img/negatives/originals/'

  for k in range(marker, num_images+marker):
    red = images["data"][k][0:1024]
    blue = images["data"][k][2048:3072]
    green = images["data"][k][1024:2048]

    # pixels flattened in row-major order
    pixels = np.zeros((32, 32, 3), dtype=np.uint8)
    
    for i in range(0, 32):
      for j in range(0, 32):
        idx = i*32+j
        pixels[i,j] = [red[idx], green[idx], blue[idx]]

    img = Image.fromarray(pixels, 'RGB')

    img.save(base_dir + str(k) + '.png')


if __name__ == '__main__':
  filenames = ['0 - altar_boy_s_001435.png',
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
  




