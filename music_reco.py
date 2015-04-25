import sys
import os
import glob
from pyechonest import config
from pyechonest import catalog
from pyechonest import song
from scipy.sparse import hstack, diags, csr_matrix
from sklearn.preprocessing import normalize
import numpy as np
from numpy import linalg as LA
import csv
import math
from random import randint
from scipy.spatial import distance
from scipy import stats

config.ECHO_NEST_API_KEY = "IEDW5SHUALLUEWCG3"

msd_subset_path = './MillionSongSubset'
msd_subset_data_path=os.path.join(msd_subset_path,'data')
msd_subset_addf_path=os.path.join(msd_subset_path,'AdditionalFiles')
msd_code_path='./MSongsDB'
sys.path.append( os.path.join(msd_code_path,'PythonSrc') )

# imports specific to the MSD
import hdf5_getters as GETTERS

def getSongProperties(songCount = 3000, splitData = True):
    songDict = {}
    songIdDict = {}
    songIdCount = 0
    for root, dirs, files in os.walk(msd_subset_data_path):
        files = glob.glob(os.path.join(root,'*.h5'))
        for f in files:
            h5 = GETTERS.open_h5_file_read(f)
            tempo = GETTERS.get_tempo(h5)
            danceability = GETTERS.get_danceability(h5)
            energy = GETTERS.get_energy(h5)
            loudness = GETTERS.get_loudness(h5)
            #print GETTERS.get_artist_terms(h5)
            timbre = GETTERS.get_segments_timbre(h5)
            artist_hotness = GETTERS.get_artist_hotttnesss(h5)
            song_key = GETTERS.get_key(h5)
            songIdDict[GETTERS.get_song_id(h5)] = songIdCount
            songDict[songIdCount] = [tempo,danceability,energy,loudness,artist_hotness,song_key]
            songIdCount += 1
            h5.close()
            #if len(songDict) >2:
             #   break
        #if len(songDict) >2:
         #   break
        if songIdCount > songCount and splitData:
            break
    return songIdDict,songDict


def loadFile(filename, songSet = {}, trimData = False):
    trainUsers = []
    trainSongs = []
    trainRatings = []
    testData = {}
    userIdDict = {}
    userIdCount = 0
    songIdCount = max(songSet.values())+1 if len(songSet) > 0   else 0
    testInstanceCount = 400000
    lineCount = 0
    maxRating = 0
    minRating = 0
    with open(filename,'rb') as infile:
        for line in infile:
            row = line.rstrip().split("\t")
            song = row[1]
            user = row[0]
            rating = 1 + 5*math.log(1 + int(row[2]))
            
            if rating > maxRating:
                maxRating = rating
            if rating < minRating:
                minRating = rating
            if user not in userIdDict:
                userIdDict[user] = userIdCount
                userIdCount += 1
                
            user = userIdDict[user]
                
            if song not in songSet:
                songSet[song] = songIdCount
                songIdCount += 1
                
            song = songSet[song]
            
            if lineCount < testInstanceCount and randint(0,20) % 3 ==0:
                testData[user,song] = rating
                    
            else:
                trainUsers.append(user)
                trainSongs.append(song)
                trainRatings.append(rating)
            
            lineCount += 1 
            if trimData and lineCount > 100000:
                break
    
    #print "Maximum  = ",maxRating," and minimum = ",minRating
    return userIdDict,songSet,csr_matrix((trainRatings,(trainUsers,trainSongs))),testData

def trimTestData(testData, trainData, songProperties):
    fiveCount = 0
    actualTestData = {}
    for user,song in testData.keys():
        if trainData[user].nnz == 0 or song not in songProperties:
            continue
        count = 0
        for arr in trainData[user].nonzero():
            for songIndex in arr:
                if songIndex in songProperties and songIndex != 0:
                    count += 1
        if count >= 2:
            fiveCount += 1 
            actualTestData[user,song] = testData[user,song]
    return actualTestData  

def computeCosineSimilarity(songProperties, song1, song2):
    if song1 not in songProperties or song2 not in songProperties:
        return 0
    song1Mod = np.dot(songProperties[song1],songProperties[song1])
    song2Mod = np.dot(songProperties[song2],songProperties[song2])
    if song1Mod != 0 and song2Mod != 0:
        return np.dot(songProperties[song1],songProperties[song2])/(song1Mod * song2Mod)
    return 0

def computeEuclideanSimilarity(songProperties, song1, song2):
    if song1 not in songProperties or song2 not in songProperties:
        return 0
    return distance.euclidean(songProperties[song1],songProperties[song2])

def computePearsonSimilarity(songProperties, song1, song2):
    if song1 not in songProperties or song2 not in songProperties:
        return 0
    return stats.pearsonr(songProperties[song1],songProperties[song2])[0]

def matrix_factorization(R, r=15, mew=0.001, reg=0.1, numIterations = 20):
    N = R.shape[0]#len(R)
    M = R.shape[1]#len(R[0])
    P = np.random.rand(N,r)
    Q = np.random.rand(M,r)
    Q = Q.T
    count = 0
    for iteration in xrange(numIterations):
        nnzArray = R.nonzero()
        for index in range(0,len(nnzArray[0])):
            i = nnzArray[0][index]
            j = nnzArray[1][index]
            if R[i,j] > 0:
                eij = R[i,j] - np.dot(P[i],Q.T[j])
                for k in xrange(r):
                    P[i,k] = P[i,k] + mew * (2 * eij * Q[k,j] -  2*reg* P[i,k])
                    Q[k,j] = Q[k,j] + mew * (2 * eij * P[i,k] -  2*reg* Q[k,j])
            count += 1
            if count % 100000 == 0:
                print "Count = ",count
        
        count = 0
        error = 0
        n = 1
        nnzArray = R.nonzero()
        for index in range(0,len(nnzArray[0])):
            i = nnzArray[0][index]
            j = nnzArray[1][index]
            if R[i,j] > 0:
                error += pow(R[i,j] - np.dot(P[i],Q.T[j]), 2)
                for k in xrange(r):
                    error += reg * ( pow(P[i,k],2) + pow(Q[k,j],2) )
                n += 1
            count += 1
            if count % 100000 == 0:
                print "Count = ",count

        if math.sqrt(error/n) < 0.01:
            break
  
    return P, Q.T

def computeMFRMSE(testData,trainData, r = 5, numIterations = 5):
    nP,nQ = matrix_factorization(trainData,r = r,numIterations=numIterations)
    predictedMatrix = np.dot(nP,nQ.T)
    instances = 0
    error = 0
    for pair,actualRating in testData.iteritems():
        user = pair[0]
        songId = pair[1]
        #print "Predicted = ",predictedMatrix[user,songId]," actual = ",actualRating
        if user < predictedMatrix.shape[0] and songId < predictedMatrix.shape[1]:
            error += (predictedMatrix[user,songId] - actualRating)**2
            instances += 1

    rmse = (error/instances)**0.5
    return rmse


def runMF():
	userIdDict,songSet, trainData, testData =  loadFile('train_triplets.txt', songIdDict, trimData=True)
	for i in range(1,21):
		print "r=",i," and RMSE = ",computeMFRMSE(testData,trainData,r=i,numIterations = 10)

def runHybrid(simFunc):
    for i in range(1,21):
        userIdDict,songSet, trainData, testData =  loadFile('train_triplets.txt', songIdDict, trimData=True)
        actualTestData = trimTestData(testData,trainData,songProperties)
        print i,",",computeRMSE(actualTestData,trainData,songProperties, simFunc)
    
