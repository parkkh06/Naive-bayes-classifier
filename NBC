import itertools
import json
import numpy as np
import pandas as pd
import re
import random
import operator
import math
import time
from collections import Counter
from matplotlib import pyplot as plt

################################################
def load(path):
    review = []
    for line in open(path, 'r'):
        row = json.loads(line)
        review.append(row)
    return np.array(review)

##################################################

def preprocessing(sentence):
    letters = re.sub('[^a-zA-Z]', ' ', sentence)
    words_list = letters.lower().split()
    return words_list

#################################################

def from_iterable(iterables):        # Flatten list : 2D -> 1D
    for it in iterables:
        for element in it:
            yield element

##################################################

def DefaultClassifier(Train, Test, string):          # Baseline 비교용
    if string == 'isFunny':
        index = 5000
    elif string == 'isUseful':
        index = 5001
    elif string == 'isCool':
        index = 5002
    elif string == 'isPositive':
        index = 5003
        
            
    Y_1Count = 0
    for i in range(len(Train)):
        if Train[i][index] == 1:
            Y_1Count += 1                                # Y = 1의 갯수
    
    Y_1Pr = Y_1Count / len(Train)
    
    if Y_1Pr > 0.5:
        return [1 for i in range(len(Test))]
    elif Y_1Pr < 0.5:
        return [0 for i in range(len(Test))]

#####################################################

def Accuracy(predict, label):
    equal = 0
    for a, b in zip(predict, label):
            if a == b:
                equal += 1
    accuracy = equal / len(predict)
    return accuracy

#####################################################

def Error(list):
    error = []
    for i in list:
        error.append(1-i)
    return error

#####################################################
########## 1 - (a)

def Filter(data):
    FilteredData = []                      # list type
    for i in range(len(data)):
        if 2 < data[i]['votes']['funny'] + data[i]['votes']['useful'] + data[i]['votes']['cool'] < 11:
            FilteredData.append(data[i])
    return FilteredData                 #return list type

##########################################################

path = ''   # Location of file
RawData = load(path+"yelp_academic_dataset_review.json")

with open(path+"filtered.json", 'w') as file:
    file.write(json.dumps(Filter(RawData)))       


FilteredData = load(path+"filtered.json")
tmp = np.array(FilteredData[0])              # need to extract data, Filtered Data is the list of length 1

review = []
for i in range(len(tmp)):
    review.append(tmp[i]['text'])           # extract review part, which is key 'text' of the dictionary

Words = []                                #2D list
for i in review:
    Words.append(preprocessing(i))          # store each words in the nested list

FlattenWords = list(itertools.chain.from_iterable(Words))     # flattening 2d list 

result = Counter(FlattenWords)              # counting appearance of each words with dict type : {'words' : number}

Top5000 = result.most_common(5000)      # tuples in list, 1st : the / 5000th : malt 

############################################################################


def GenerateDataPoints(dictionary):           # each data points are like this : [1, 1, 0, 1, ... 1, 0, 0]  - list type
    global Top5000                            # input : dictionary
    
    X_variable = []
    for i in range(len(Top5000)):
        X_variable.append(Top5000[i][0])
    Y_variable = ['isFunny', 'isUeful', ' isCool', 'isPositive']
        
    review = set(preprocessing(dictionary['text']))     # list type
    
    X_counts = [0 for i in range(5000)]             # X features list
    for i in range(len(X_variable)):
        if X_variable[i] in review:
            X_counts[i] = 1
    
    Y_counts = [0 for i in range(4)]            # Y features list - binary label
    if dictionary['votes']['funny'] > 0:        # Y1 - funny / Y2 - useful / Y3 - cool / Y4 - positive
        Y_counts[0] = 1
    if dictionary['votes']['useful'] > 0:
        Y_counts[1] = 1
    if dictionary['votes']['cool'] > 0:
        Y_counts[2] = 1
    if dictionary['stars'] > 3.5:
        Y_counts[3] = 1
    
    counts = X_counts + Y_counts
    
    return counts                           # length of list is 5004, 5000 X features, 4 Y class
    
#########################################################3

def GenerateConditionalProb(Train, string):             # string : indicating class
    
    if string == 'isFunny':
        index = 5000
    elif string == 'isUseful':
        index = 5001
    elif string == 'isCool':
        index = 5002
    elif string == 'isPositive':
        index = 5003
        
    oneVector = [1 for i in range(5000)]


    isSomething = []                                       # data with Y label = 1       
    for i in range(len(Train)):
        if Train[i][index] == 1:
            isSomething.append(Train[i])                 

    isSomethingCount = []
    for i in range(5000):
        cnt = 0
        for j in range(len(isSomething)):
            cnt += isSomething[j][i]
        isSomethingCount.append(cnt)                         # number of appearance, vector with length 5000

    isSomethingDen = [len(isSomething)+2 for i in range(5000)]  # Laplace Smoothing Denominator

    isSomethingNum = [a+b for a, b in zip(isSomethingCount, oneVector)]  # Laplace Smoothing numeratir

    isSomethingPr = [a/b for a, b in zip(isSomethingNum, isSomethingDen)]    # list of conditional prob Pr[Xi = 1 | Y = 1],
                                                                              #        Pr[Xi = 0 | Y = 1] = 1 - Pr[Xi = 1 | Y = 1]
    
    notSomething = []               # data with Y label = 0
    for i in range(len(Train)):
        if Train[i][index] == 0:
            notSomething.append(Train[i])

    notSomethingCount = []
    for i in range(5000):
        cnt = 0
        for j in range(len(notSomething)):
            cnt += notSomething[j][i]
        notSomethingCount.append(cnt)
    
    notSomethingDen = [len(notSomething)+2 for i in range(5000)]

    notSomethingNum = [a+b for a, b in zip(notSomethingCount, oneVector)]

    notSomethingPr = [a/b for a, b in zip(notSomethingNum, notSomethingDen)]   # list of conditional prob Pr[Xi = 1 | Y = 0] 

    return isSomethingPr, notSomethingPr, isSomething, notSomething


###########################################################################################################


def NBC(Train, Test, string):                    # Train set / Test set : 2d list / string : indicates class
    
    isSomethingPr, notSomethingPr, isSomething, notSomething = GenerateConditionalProb(Train, string)
    
    Y_1ConditionPr = []
    for i in range(len(Test)):
        Testi = []
        for j in range(5000):
            if Test[i][j] == 1:
                Testi.append(isSomethingPr[j])
            elif Test[i][j] == 0:
                Testi.append(1-isSomethingPr[j])
        Y_1ConditionPr.append(Testi)
        
    for testi in Y_1ConditionPr:
        testi.append(len(isSomething) / len(Train))      # probability of Y = 1
        
    Y_0ConditionPr = []
    for i in range(len(Test)):
        Testi = []
        for j in range(5000):
            if Test[i][j] == 1:
                Testi.append(notSomethingPr[j])
            elif Test[i][j] == 0:
                Testi.append(1-notSomethingPr[j])
        Y_0ConditionPr.append(Testi)
        
    for testi in Y_0ConditionPr:
        testi.append(len(notSomething) / len(Train))      # probability of  Y = 0
    
    Y_1LogValue = []
    for testi in Y_1ConditionPr:
        Y_1LogValue.append(sum(np.log(np.array(testi))))

    Y_0LogValue = []
    for testi in Y_0ConditionPr:
        Y_0LogValue.append(sum(np.log(np.array(testi))))

    PredictedY = []
    for a, b in zip(Y_1LogValue, Y_0LogValue):
        if a > b:
            PredictedY.append(1)
        elif a < b:
            PredictedY.append(0)
            
    return PredictedY                 ## return list of predicted class


###########################################################################

###
## Main body
###

TrainSize = [50, 100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600]
start = time.time()

NBCAccuracyisFunny = []
DefaultAccuracyisFunny = []

NBCAccuracyisUseful = []
DefaultAccuracyisUseful = []

NBCAccuracyisCool = []
DefaultAccuracyisCool = []

NBCAccuracyisPositive = []
DefaultAccuracyisPositive = []

    
for size in TrainSize:
    
    SampleTrain = random.sample(list(tmp), size)       # dict in the list / [50, 100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600]
    TrainSet = [] 
    for i in SampleTrain:
        TrainSet.append(GenerateDataPoints(i))            # randomly select train set
        
    
    SampleTest = random.sample(list(tmp), 8000)           # fix the test set
    TestSet = []                                           # 2d list
    for i in SampleTest:
        TestSet.append(GenerateDataPoints(i))
    
    
    #############################################################
    ################ Class : isFunny
    
    PredictedYisFunny = NBC(TrainSet, TestSet, 'isFunny')
    
    DefaultYisFunny = DefaultClassifier(TrainSet, TestSet, 'isFunny')
    
    LabelYisFunny = []
    for i in range(len(TestSet)):
        LabelYisFunny.append(TestSet[i][5000])
    
    NBCScoreisFunny = Accuracy(PredictedYisFunny, LabelYisFunny)
    DefaultScoreisFunny = Accuracy(DefaultYisFunny, LabelYisFunny)
    
    NBCAccuracyisFunny.append(NBCScoreisFunny)
    DefaultAccuracyisFunny.append(DefaultScoreisFunny)
    
    ############################################################
    ################ Class : isUseful
    
    PredictedYisUseful = NBC(TrainSet, TestSet, 'isUseful')
    
    DefaultYisUseful = DefaultClassifier(TrainSet, TestSet, 'isUseful')
    
    LabelYisUseful = []
    for i in range(len(TestSet)):
        LabelYisUseful.append(TestSet[i][5001])
    
    NBCScoreisUseful = Accuracy(PredictedYisUseful, LabelYisUseful)
    DefaultScoreisUseful = Accuracy(DefaultYisUseful, LabelYisUseful)
    
    NBCAccuracyisUseful.append(NBCScoreisUseful)
    DefaultAccuracyisUseful.append(DefaultScoreisUseful)
    
    ###########################################################
    ################# Class : isCool

    PredictedYisCool = NBC(TrainSet, TestSet, 'isCool')
    
    DefaultYisCool = DefaultClassifier(TrainSet, TestSet, 'isCool')
    
    LabelYisCool = []
    for i in range(len(TestSet)):
        LabelYisCool.append(TestSet[i][5002])

    
    NBCScoreisCool = Accuracy(PredictedYisCool, LabelYisCool)
    DefaultScoreisCool = Accuracy(DefaultYisCool, LabelYisCool)
    
    
    NBCAccuracyisCool.append(NBCScoreisCool)
    DefaultAccuracyisCool.append(DefaultScoreisCool)
    
    #############################################################
    ################ Class : isPositive
    
    PredictedYisPositive = NBC(TrainSet, TestSet, 'isPositive')
    
    DefaultYisPositive = DefaultClassifier(TrainSet, TestSet, 'isPositive')
    
    LabelYisPositive = []
    for i in range(len(TestSet)):
        LabelYisPositive.append(TestSet[i][5003])
    
    NBCScoreisPositive = Accuracy(PredictedYisPositive, LabelYisPositive)
    DefaultScoreisPositive = Accuracy(DefaultYisPositive, LabelYisPositive)
    
    NBCAccuracyisPositive.append(NBCScoreisPositive)
    DefaultAccuracyisPositive.append(DefaultScoreisPositive)
    
    ###############################################################
    
print("time :", time.time() - start)



plt.plot(TrainSize, Error(NBCAccuracyisFunny), label = 'NBC')
plt.plot(TrainSize, Error(DefaultAccuracyisFunny), label = 'Baseline')
plt.xlabel('Train Size')
plt.ylabel('Error rate')
plt.title('Class : isFunny')
plt.legend()
plt.show
print(Error(NBCAccuracyisFunny))
print(Error(DefaultAccuracyisFunny))

plt.plot(TrainSize, NBCAccuracyisFunny, label = 'NBC')
plt.plot(TrainSize, DefaultAccuracyisFunny, label = 'Baseline')
plt.xlabel('Train Size')
plt.ylabel('Accuracy')
plt.title('Class : isFunny')
plt.legend()
plt.show
print(NBCAccuracyisFunny)
print(DefaultAccuracyisFunny)


plt.plot(TrainSize, Error(NBCAccuracyisUseful), label = 'NBC')
plt.plot(TrainSize, Error(DefaultAccuracyisUseful), label = 'Baseline')
plt.xlabel('Train Size')
plt.ylabel('Error rate')
plt.title('Class : isUseful')
plt.legend()
plt.show
print(Error(NBCAccuracyisUseful))
print(Error(DefaultAccuracyisUseful))

plt.plot(TrainSize, NBCAccuracyisUseful, label = 'NBC')
plt.plot(TrainSize, DefaultAccuracyisUseful, label = 'Baseline')
plt.xlabel('Train Size')
plt.ylabel('Accuracy')
plt.title('Class : isUseful')
plt.legend()
plt.show
print(NBCAccuracyisUseful)
print(DefaultAccuracyisUseful)


plt.plot(TrainSize, Error(NBCAccuracyisCool), label = 'NBC')
plt.plot(TrainSize, Error(DefaultAccuracyisCool), label = 'Baseline')
plt.xlabel('Train Size')
plt.ylabel('Error rate')
plt.title('Class : isCool')
plt.legend()
plt.show
print(Error(NBCAccuracyisCool))
print(Error(DefaultAccuracyisCool))

plt.plot(TrainSize, NBCAccuracyisCool, label = 'NBC')
plt.plot(TrainSize, DefaultAccuracyisCool, label = 'Baseline')
plt.xlabel('Train Size')
plt.ylabel('Accuracy')
plt.title('Class : isCool')
plt.legend()
plt.show
print(NBCAccuracyisCool)
print(DefaultAccuracyisCool)

plt.plot(TrainSize, Error(NBCAccuracyisPositive), label = 'NBC')
plt.plot(TrainSize, Error(DefaultAccuracyisPositive), label = 'Baseline')
plt.xlabel('Train Size')
plt.ylabel('Error rate')
plt.title('Class : isPositive')
plt.legend()
plt.show
print(Error(NBCAccuracyisPositive))
print(Error(DefaultAccuracyisPositive))

plt.plot(TrainSize, NBCAccuracyisPositive, label = 'NBC')
plt.plot(TrainSize, DefaultAccuracyisPositive, label = 'Baseline')
plt.xlabel('Train Size')
plt.ylabel('Accuracy')
plt.title('Class : isPositive')
plt.legend()
plt.show
print(NBCAccuracyisPositive)
print(DefaultAccuracyisPositive)
