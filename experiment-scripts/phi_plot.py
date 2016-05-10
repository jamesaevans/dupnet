import matplotlib.pyplot as plt
import numpy as np

def testPhi(phi):
	POS_FILE = 'pos_outputs.txt'
	NEG_FILE = 'neg_outputs.txt'
	posLines = open(POS_FILE).read().split('\n')
	negLines = open(NEG_FILE).read().split('\n')

	# Total number of training examples
	totalNum = len(posLines) + len(negLines)

	posCorrect = 0
	for pos in posLines:
		vals = pos.split(',')[1:-1]
		probs = []
		for v in vals:
			probs.append(float(v))
		correct = int(pos.split(',')[0].split('-')[-1][0]) # correct class
		guess = probs.index(max(probs))
		if guess == correct and probs[correct] > phi:
			posCorrect += 1

	negCorrect = 0
	for neg in negLines:
		vals = neg.split(',')[1:]
		probs = []
		for v in vals:
			probs.append(float(v))
		if max(probs) <= phi:
			negCorrect += 1

	print(negCorrect)
	return (posCorrect+negCorrect)/float(totalNum)

def getMaxPositiveDist():
	guessDist = []
	correctDist = []
	lines = open('pos_outputs.txt').read().split('\n')
	for pos in lines:
		vals = pos.split(',')[1:]
		probs = []
		for v in vals:
			probs.append(float(v))
		correct = int(pos.split(',')[0].split('-')[-1][0])
		print probs
		guessDist.append(max(probs))
		correctDist.append(probs[correct])

	return guessDist, correctDist

def getMaxNegativeDist():
	toRet = []
	lines = open('pos_outputs.txt').read().split('\n')
	for l in lines:
		vals = l.split(',')[1:-1]
		probs = []
		for v in vals:
			probs.append(float(v))
		toRet.append(max(probs))

	return toRet






