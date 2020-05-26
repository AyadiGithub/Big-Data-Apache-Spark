from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession


sc = SparkContext.getOrCreate(SparkConf().setMaster("local[*]"))

spark = SparkSession \
    .builder \
    .getOrCreate()

df = spark.read.parquet('C:/*******/Week 3/hmp.parquet')
df.show()
df.createOrReplaceTempView('df')

from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder, Normalizer
from pyspark.ml.linalg import Vectors
from pyspark.ml import Pipeline

#Data pre-processing
indexer = StringIndexer(inputCol="class", outputCol="classIndex")
encoder = OneHotEncoder(inputCol="classIndex", outputCol="categoryVec")
vectorAssembler = VectorAssembler(inputCols=["x","y","z"],
                                  outputCol="features")
normalizer = Normalizer(inputCol="features", outputCol="features_norm", p=1.0)

#Setting up the pipeline
pipeline = Pipeline(stages=[indexer,encoder,vectorAssembler,normalizer])
model = pipeline.fit(df)
prediction = model.transform(df)
prediction.show()

#Creating new pipeline for KMeans
#Importing KMeans and ClusteringEvaluator to measure performance of the model
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

#Setting up Kmeans
kmeans = KMeans(featuresCol="features").setK(14).setSeed(1)
pipeline = Pipeline(stages=[vectorAssembler, kmeans])
model = pipeline.fit(df)
predictions = model.transform(df)

#Evaluating using Squared Euclidean Distance
evaluator = ClusteringEvaluator()
silhouette = evaluator.evaluate(predictions)
print("Silhouette with squared euclidean distance = ",silhouette)

import numpy as np

Ks = 15
mean_acc = np.zeros((Ks-1))
ConfusionMx = [];
for n in range(7,Ks):
    
    #Train Model and Predict  
    kmeans = KMeans(featuresCol="features", k=n, seed=1)
    pipeline = Pipeline(stages=[vectorAssembler, kmeans])
    model = pipeline.fit(df)
    predictions = model.transform(df)
    evaluator = ClusteringEvaluator()
    mean_acc[n-1] = evaluator.evaluate(predictions)

    
mean_acc
print( "The best accuracy was with", mean_acc.max(), "with k=", mean_acc.argmax()+1) 

#Lets inflate dataset by x10 to see if it helps with the accuracy
from pyspark.sql.functions import col
#Denormalizing the dataset
df_denormalized = df.select([col('*'),(col('x')*10)]).drop('x').withColumnRenamed('(x * 10)','x')

#Lets test again with the best k = 7 on the denormalized dataset

kmeans = KMeans(featuresCol="features").setK(7).setSeed(1)
pipeline = Pipeline(stages=[vectorAssembler, kmeans])
model = pipeline.fit(df_denormalized)
predictions = model.transform(df_denormalized)

evaluator = ClusteringEvaluator()

silhouette = evaluator.evaluate(predictions)
print("Silhouette with squared euclidean distance = " + str(silhouette))

#Lets try GaussianMixture instead of KMeans
from pyspark.ml.clustering import GaussianMixture
gmm = GaussianMixture(featuresCol="features").setK(14).setSeed(1)
pipeline = Pipeline(stages=[vectorAssembler,gmm])
model = pipeline.fit(df)
predictions = model.transform(df)
evaluator = ClusteringEvaluator()

silhouette = evaluator.evaluate(predictions)
print("Silhouette with squared euclidean distance = " + str(silhouette))

#Lets try finding the best K using an iterative method
Ks = 15
mean_acc = np.zeros((Ks-1))
ConfusionMx = [];
for n in range(7,Ks):
    
    #Train Model and Predict  
    gmm = GaussianMixture(featuresCol="features", k=n, seed=1)
    pipeline = Pipeline(stages=[vectorAssembler,gmm])
    model = pipeline.fit(df)
    predictions = model.transform(df)
    evaluator = ClusteringEvaluator()
    mean_acc[n-1] = evaluator.evaluate(predictions)

    
mean_acc
print( "The best accuracy was with", mean_acc.max(), "with k=", mean_acc.argmax()+1) 



