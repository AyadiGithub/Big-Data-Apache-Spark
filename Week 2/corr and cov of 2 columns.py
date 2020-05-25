
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession

sc = SparkContext.getOrCreate(SparkConf().setMaster("local[*]"))

spark = SparkSession \
    .builder \
    .getOrCreate()

import random

#Defining rddX and rddY and parallelizing them in spark
rddX = sc.parallelize(range(100))
rddY = sc.parallelize(range(100))

#Calculating the mean for both Rdds
meanX = rddX.sum()/float(rddX.count())
meanY = rddY.sum()/float(rddY.count())

print(meanX)
print(meanY)


#Combining both rdds into one rddXY
rddXY = rddX.zip(rddY)

#Calculating covariance of both columns
covXY = rddXY.map(lambda xy : (float(xy[0])-meanX)*(float(xy[1])-meanY)).sum()/rddXY.count()
covXY

#importing sqrt function
from math import sqrt  
#Calculating Std Deviation of each rdd
n = float(rddXY.count())
sdX = sqrt(rddX.map(lambda x : pow(x-meanX,2)).sum()/n)
sdX

sdY = sqrt(rddY.map(lambda y : pow(y-meanY,2)).sum()/n)
sdY

#Calculating Skewness
skewnessX = (1/n)*rddX.map(lambda x : pow(x-meanX,3)/pow(sdX,3)).sum()
print(skewnessX)

skewnessY = (1/n)*rddY.map(lambda y : pow(y-meanY,3)/pow(sdY,3)).sum()
print(skewnessY)


#Calculating Kurtosis (a measurement indicating the number of outliers and length of tail in the distribution)
KurtosisX = rddX.map(lambda x : pow(x-meanX,4)/pow(sdX,4)).sum()/n
print(KurtosisX)

kurtosisY = rddY.map(lambda y : pow(y-meanY,4)/pow(sdY,4)).sum()/n
print(kurtosisY)

#Calculating Correlation of the two rdds
#A value of 1.0 means total correlation, -1.0 means inverse correlation
#A value of 0 means no correlation

corrXY = covXY / (sdX*sdY)
corrXY


