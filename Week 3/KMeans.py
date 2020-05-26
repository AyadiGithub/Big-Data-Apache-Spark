from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession


sc = SparkContext.getOrCreate(SparkConf().setMaster("local[*]"))

spark = SparkSession \
    .builder \
    .getOrCreate()

df = spark.read.parquet('C:/*******/Week 3/hmp.parquet')
df.show()


from pyspark.ml.feature import VectorAssembler
VectorAssembler = VectorAssembler(inputCols=["x","y","z"], 
                                  outputCol="features")

#Importing KMeans from Clustering package in pyspark
from pyspark.ml.clustering import KMeans

KMeans = KMeans().setK(13).setSeed(1)

#Importing pipeline from pyspark.ml 
from pyspark.ml import Pipeline

#Calling the pipeline stages
Pipeline = Pipeline(stages=[VectorAssembler,KMeans])

#Setting the model
model = Pipeline.fit(df)

#To measure the performance of our model, we use WSSSE (Within Set Sum of Squared Errors) measure.
WSSSE = model.stages[1].computeCost(VectorAssembler.transform(df))
print("The WSSSE for the model = ",WSSSE)

df.createOrReplaceTempView('df')
df = spark.sql("select * from df where class in ('Climb_stairs','Brush_teeth')")
