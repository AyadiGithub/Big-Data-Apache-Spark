
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession


sc = SparkContext.getOrCreate(SparkConf().setMaster("local[*]"))

spark = SparkSession \
    .builder \
    .getOrCreate()
    

#creating the columns
from pyspark.mllib.stat import Statistics
import random

column1 = sc.parallelize(range(100))
column2 = sc.parallelize(range(100,200))
column3 = sc.parallelize(list(reversed(range(100))))
column4 = sc.parallelize(random.sample(range(100),100))

#Creating a covariance matrix as data
data = column1.zip(column2).zip(column3).zip(column4).map(lambda nested : (nested[0][0][0],nested[0][0][1],nested[0][1],nested[1]))
print(Statistics.corr(data))

data.take(10)
