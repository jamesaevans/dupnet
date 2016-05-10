import numpy as np
from PIL import Image
from sets import Set
from PIL import ImageEnhance
import random

def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

def getScaledFactor(minScale, maxScale):
  	factor = random.random()*2
  	factor = min(maxScale, factor)
  	factor = max(minScale, factor)
  	return factor

def extract_from_CIFAR(numClasses):
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

  # Build up manipulated image
  activeImage = image

  # Apply brightness transformation
  if random.randint(1, 10) <= 4:
  	factor = getScaledFactor(MIN_SCALE_FACTOR, MAX_SCALE_FACTOR)
	enhancer = ImageEnhance.Brightness(image)
	activeImage = enhancer.enhance(factor)

  # Apply Contrast transformation
  if random.randint(1, 10) <= 4:
  	factor = getScaledFactor(MIN_SCALE_FACTOR, MAX_SCALE_FACTOR)
	enhancer = ImageEnhance.Contrast(image)
	activeImage = enhancer.enhance(factor)

  # Apply Color transformation
  if random.randint(1, 10) <= 4:
  	factor = getScaledFactor(MIN_SCALE_FACTOR, MAX_SCALE_FACTOR)
	enhancer = ImageEnhance.Sharpness(image)
	activeImage = enhancer.enhance(factor)

  # Apply Color transformation
  if random.randint(1, 10) <= 4:
  	factor = getScaledFactor(MIN_SCALE_FACTOR, MAX_SCALE_FACTOR)
  	enhancer = ImageEnhance.Color(image)
	activeImage = enhancer.enhance(factor)

  # Rotates image either 90, 180, or 270 degrees
  if random.randint(1, 10) <= 4:
  	randVal = random.random()
  	angle = 0
  	if randVal < 0.33:
  		angle = 90
  	elif randVal < 0.66:
  		angle = 180
  	else:
  		angle = 270
	activeImage = activeImage.rotate(angle)

  return activeImage

if __name__ == '__main__':
	f = Image.open('img/originals/9 - willow_tree_s_000645.png')
	for i in range(0, 1000):
		manipulate_image(f).save('img/transformed/9/' + str(i) + '.png')
	




