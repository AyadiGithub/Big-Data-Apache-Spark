from IPython.display import Markdown, display
def printmd(string):
    display(Markdown('# <span style="color:red">'+string+'</span>'))


if ('sc' in locals() or 'sc' in globals()):
    printmd('<<<<<!!!!! It seems that you are running in a IBM Watson Studio Apache Spark Notebook. Please run it in an IBM Watson Studio Default Runtime (without Apache Spark) !!!!!>>>>>')

#!pip install pyspark==2.4.5

try:
    from pyspark import SparkContext, SparkConf
    from pyspark.sql import SparkSession
except ImportError as e:
    printmd('<<<<<!!!!! Please restart your kernel after installing Apache Spark !!!!!>>>>>')
    

sc = SparkContext.getOrCreate(SparkConf().setMaster("local[*]"))

spark = SparkSession \
    .builder \
    .getOrCreate()
    
# Python function that decides if a value is >50 (true) or not (false)

def gt50(i):
    if i>50:
        return True
    else:
        return False

print(gt50(8))
print(gt50(52))

# Simplifying the function

def gt50(i):
    return i>50

print(gt50(8))
print(gt50(52))

#Using Lambda notation

gt50 = lambda i: i>50
gt50

print(gt50(8))
print(gt50(52))

# Shuffling list

from random import shuffle
l = list(range(100))
shuffle(l)
l
rdd = sc.parallelize(l)

#filtering values in the list that are equal or less than 50
#Using the collect function should not be done on Big Data. instead take(n)

rdd.filter(gt50).collect()

#Lambda function can be used directly as well
rdd.filter(lambda i: i>50).collect()


#Filter elements > 50 and < 75 and compute the sum of all elements
rdd.filter(lambda x: x < 75).filter(lambda x: x > 50).sum()


