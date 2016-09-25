
# coding: utf-8

# In[ ]:

import numpy as np
import csv
import requests
from os.path import abspath
path = abspath('iris.data')

# open data
with open(path, 'r') as file:
    data = []
    for entry in csv.reader(file):
        entry[4] = entry[4].replace("Iris-setosa", '1').replace('Iris-versicolor', '2').replace('Iris-virginica', '3')
        data.append(entry)
        
data = np.array(data,dtype='float') # change format from string to float
data = np.roll(data,1,axis=1) # rotate the label to front
label, feature = data[:,0], data[:,1:] # split data into label/feature arrays


# In[ ]:

from pyspark import SparkContext, SparkConf

# initialize spark config
conf = SparkConf().setMaster('local[*]').setAppName('naiveBayes_iris')
# initialize spark context
sc = SparkContext(conf = conf)


# In[ ]:

from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.feature import Normalizer
from pyspark.mllib.classification import NaiveBayes, NaiveBayesModel
from pyspark.mllib.util import MLUtils

# spark read data
LabelRDD = sc.parallelize(label)
FeatureRDD = sc.parallelize(feature)
# combine data into format: [[label, [feature1,..., feature n]],...]
datasetRDD = LabelRDD.zip(FeatureRDD)
# transform data format into labeledpoint
datasetRDD = datasetRDD.map(lambda entry: LabeledPoint(entry[0], entry[1]))

#-----Single training-----
# random candidate select 67% for training
r = 2/3
trainingRDD, testRDD = datasetRDD.randomSplit([r,1-r])
del r
print('training dataset:', trainingRDD.count())
print('test dataset:', testRDD.count())

model = NaiveBayes.train(trainingRDD) # training model with naiveBayes

# make prediction with model then compare with label to form RDD
predictionAndLabel = testRDD.map(lambda p: (model.predict(p.features), p.label)) 
accurancy = predictionAndLabel.filter(lambda r: r[0]==r[1])
for i in range(1,4):
    class_i = predictionAndLabel.filter(lambda r: r[1] == i)
    correctAns_i = predictionAndLabel.filter(lambda r: r[0] == r[1] and r[0] == i)
    print('Accurancy of class', i , ':', correctAns_i.count()/class_i.count())
print('Average accurancy :', accurancy.count()/testRDD.count())
#-------------------------

#-----Testing for 1000 times-----
# result = []
# for i in range(1000):
#     # random candidate select 67% for training
#     r = 2/3
#     trainingRDD, testRDD = datasetRDD.randomSplit([r, 1-r])
#     del r
    
#     model = NaiveBayes.train(trainingRDD)

#     predictionAndLabel = testRDD.map(lambda p: (model.predict(p.features), p.label))
#     accurancy = predictionAndLabel.filter(lambda r: r[0]==r[1])
#     acc = accurancy.count()/testRDD.count()
#     result.append(acc)
# #     print('Accurancy:', accurancy.count()/testRDD.count())
# print('Overall accurancy:', np.mean(result))
#--------------------------------

# -----save model-----
# import shutil
# output_dir = '/home/jovyan/work/sparkml/model'
# shutil.rmtree(output_dir, ignore_errors=True)
# model.save(sc, output_dir)

# -----test load model-----
# sameModel = NaiveBayesModel.load(sc, output_dir)
# predictionAndLabel2 = testRDD.map(lambda p: (sameModel.predict(p.features), p.label))
# accuracy = predictionAndLabel2.filter(lambda r: r[0] == r[1]).count() / testRDD.count()
# print('Same model accuracy :{}'.format(accuracy))

sc.stop()

