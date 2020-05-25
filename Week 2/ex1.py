from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession

sc = SparkContext.getOrCreate(SparkConf().setMaster("local[*]"))

spark = SparkSession \
    .builder \
    .getOrCreate()

#Creating Dataframe
df = spark.read.parquet(r'C:\****\Week 2\hmp.parquet')
df.show(10)

#creating a corresponding query tbale
df.createOrReplaceTempView('df')
df.printSchema()

#Finding out if classes are balanced using aggregation in SQL.
#spark.sql('select class,count(*) from df group by class').show()

#Using Dataframe API instead of SQL to count the classes
df.groupBy('class').count().show()

#Importing Matplotlib
counts = df.groupBy('class').count().orderBy('count')
counts.show()

# SQL CODE BELOW
# =============================================================================
# spark.sql('''
#     select 
#         *,
#         max/min as minmaxratio -- compute minmaxratio based on previously computed values
#         from (
#             select 
#                 min(ct) as min, -- compute minimum value of all classes
#                 max(ct) as max, -- compute maximum value of all classes
#                 mean(ct) as mean, -- compute mean between all classes
#                 stddev(ct) as stddev -- compute standard deviation between all classes
#                 from (
#                     select
#                         count(*) as ct -- count the number of rows per class and rename it to ct
#                         from df -- access the temporary query table called df backed by DataFrame df
#                         group by class -- aggrecate over class
#                 )
#         )   
# ''').show()
# =============================================================================

# DataFrame API Code below:

from pyspark.sql.functions import col, min, max, mean, stddev
df \
    .groupBy('class') \
    .count() \
    .select ([
        min(col("count")).alias('min'),
        max(col("count")).alias('max'),
        mean(col("count")).alias('mean'),
        stddev(col("count")).alias('stddev')
    ]) \
    .select([
        col('*'),
        (col("max") / col("min")).alias('minmaxratio')
    ]) \
    .show()

#Showing class count in ascending order
counting = df.groupBy('class').count()
counting.sort("count", ascending=True).show()  

import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="whitegrid")

# Initialize the matplotlib figure
f, ax = plt.subplots(figsize=(15, 15))

# Ploting the Count x Class table

counting = counts.toPandas()


sns.set_color_codes("pastel")
sns.barplot(x="count", y="class", data=counting,
            label="Count", color="b")


#Rebalancing classes
from pyspark.sql.functions import min

# create a lot of distinct classes from the dataset
classes = [row[0] for row in df.select('class').distinct().collect()]
classes

# compute the number of elements of the smallest class in order to limit the number of samples per calss
min = df.groupBy('class').count().select(min('count')).first()[0]
min

# define the result dataframe variable
df_balanced = None

# iterate over distinct classes
for cls in classes:
    
    # only select examples for the specific class within this iteration
    # shuffle the order of the elements (by setting fraction to 1.0 sample works like shuffle)
    # return only the first n samples
    df_temp = df \
        .filter("class = '"+cls+"'") \
        .sample(False, 1.0) \
        .limit(min)
    
    # on first iteration, assing df_temp to empty df_balanced
    if df_balanced == None:    
        df_balanced = df_temp
    # afterwards, append vertically
    else:
        df_balanced=df_balanced.union(df_temp)

df_balanced.groupBy('class').count().show()

#Equal count per class achieved