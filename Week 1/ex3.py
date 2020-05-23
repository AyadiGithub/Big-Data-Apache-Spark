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


#create a DataFrame, register a temporary query table and issue SQL commands against it.

from pyspark.sql import Row

#Creating a DataFrame with columns id and value
df = spark.createDataFrame([Row(id=1, value='value1'),Row(id=2, value='value2')])

#checking the created DataFrame

df.show()

#Printing DataFrame schema

df.printSchema()

#Registering DataFrame as query table
df.createOrReplaceTempView('df_view')

#ExecutingL query
df_result = spark.sql('select value from df_view where id=2')

#Examining contents of result
df_result.show()

#Getting result as a string
df_result.first().value

#Counting number of rows in DF
df.count()
