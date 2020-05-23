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

sc = SparkContext.getOrCreate(SparkConf() .setMaster("local[*]"))

spark = SparkSession \
    .builder \
    .getOrCreate()

rdd = sc.parallelize(range(100))

rdd.count()

Count = rdd.count()
print('The count is: ', Count)

rdd.sum()

Sum = rdd.sum()
print('The sum is: ', Sum)

