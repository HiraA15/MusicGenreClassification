import scipy.io
import numpy

data = scipy.io.loadmat("gtzan.mat")
print data['fold9_features'][0]

#will calculate the euclidian distance between two freq's
def euclidianDistance(list1, list2):
	distance = 0
	for i in range(len(list2)):
		distance = distance + (list1[i] - list2[i])**2
	return distance

print euclidianDistance(data['fold1_features'][0], data['fold2_classes'][0])

def kNN(k, song, matrix):
	dic = {}
	for i in range(len(matrix)):
		index = i
		dic[euclidianDistance(song, matrix[i])] = (index)

	distances = dic.keys()
	distances.sort()

	kShortestDistances = []
	for x in range(k):
		kShortestDistances.append((distances[x], dic[distances[x]]))
		del dic[distances[x]]

	return kShortestDistances

print kNN(1, data['fold1_features'][0], data['fold8_features'])

#These are to record the names of all fetures and classes just for method defining purposes.
folds = ['fold1_features', 'fold2_features', 'fold3_features', 'fold4_features', 'fold5_features', 'fold6_features', 'fold7_features', 'fold8_features', 'fold9_features', 'fold10_features']
classes = ['fold1_classes', 'fold2_classes', 'fold3_classes', 'fold4_classes', 'fold5_classes', 'fold6_classes', 'fold7_classes', 'fold8_classes', 'fold9_classes', 'fold10_classes']

#this categorizes a song given the k nearest closest song and it's ID. It picks the one that occurs frequently.
def configure_ID(dictionary):
	genres = []
	for i in range(1, 11):
		genre_counter = 0
		for value in dictionary.values():
			if (value == i):
				genre_counter += 1
		genres.append(genre_counter)
	#for a tie between a genre, it will delete the farthest point.
	maxNumber = max(genres)
	for genre in genres:
		if (maxNumber == genre):
			if (genres.index(maxNumber) != genres.index(genre)):
				del dictionary[dictionary.keys()[-1]]
				configure_ID(dictionary)

	return genres.index(maxNumber) + 1

#takes a song, it's 'K', and the name of its feature_fold, then returns the ID of genre that the song was classified as a number from 1 to 10.
# 1 = blues
# 2 = classical
# 3 = country
# 4 = disco
# 5 = hiphop
# 6 = jazz
# 7 = metal
# 8 = pop
# 9 = reggae
# 10 = rock
def categorize(k, song, foldName):
	closestsSongs = {}
	for fold in folds:
		if (fold != foldName):
			points = kNN(k, song, data[fold])
			for p in points:
				closestsSongs[p[0]] = [p[1], fold]

	closestDistances = closestsSongs.keys()
	closestDistances.sort()
     #will pick the k closest song out of mix of songs
	KClosestSongs = {}
	for x in range(k):
		KClosestSongs[closestDistances[x]] = closestsSongs[closestDistances[x]]
		del closestsSongs[closestDistances[x]]
      #picks an ID of the K point chosen and stores in a dict with its distance.
	KClosestSongsIDs = {}
	for d in KClosestSongs.keys():
		classNumber = folds.index(KClosestSongs[d][1])
		KClosestSongsIDs[d] = data['fold' + str(classNumber + 1) + '_classes'][KClosestSongs[d][0]][0]
	return configure_ID(KClosestSongsIDs)

print categorize(1, data['fold1_features'][0], 'fold1_features')
print categorize(3, data['fold1_features'][0], 'fold1_features')
print categorize(5, data['fold1_features'][0], 'fold1_features')

#implements a classification for a complete feature fold and returns ID in a list.
def totalClassification(k, matrixSong):
	id_classes = []
	for i in data[matrixSong]:
		id_classes.append(categorize(k, i, matrixSong))
	return id_classes

print totalClassification(1, 'fold1_features')

def accuracy(ids1, ids2):
	actualIDs = [item for sublist in ids2 for item in sublist]
	print actualIDs

	correctClassificationNumber = 0
	for i in range(len(actualIDs)):
		if (ids1[i] == actualIDs[i]):
			correctClassificationNumber += 1
	return (correctClassificationNumber/float(len(ids1)))

test1 = totalClassification(1, 'fold9_features')
print accuracy(test1, data['fold9_classes'])

def totalAccuracy(k):
	sum_average = 0.0
	for	i in range(10):
		print accuracy(totalClassification(k, folds[i]), data[classes[i]])
		sum_average += accuracy(totalClassification(k, folds[i]), data[classes[i]])/10.0
		print sum_average
	return sum_average

print totalAccuracy(1)
print totalAccuracy(3)
print totalAccuracy(5)

def confusion(ids1, ids2):
	confusionMatrix = numpy.zeros((10,10), dtype=int)
	actualIDs = [item for sublist in ids2 for item in sublist]
	for count1, count2 in zip(ids1, actualIDs):
		confusionMatrix[count1-1][count2-1] += 1
	return confusionMatrix

print confusion(test1, data['fold9_classes'])

def matrixConfusion(k):
	fullMatrix = numpy.zeros((10,10), dtype=int)
	for fold in folds:
		ids = totalClassification(k, fold)
		fullMatrix += confusion(ids, data['fold' + str(folds.index(fold) + 1) + '_classes'])
	print fullMatrix

	accuracy = 0
	for i in range(10):
		print fullMatrix[i][i]
		print float(sum(fullMatrix[i]))
		accuracy += (fullMatrix[i][i])/float(sum(fullMatrix[i]))
	return accuracy/10

print matrixConfusion(1)
print matrixConfusion(5)
