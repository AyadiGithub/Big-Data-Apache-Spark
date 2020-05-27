from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession


sc = SparkContext.getOrCreate(SparkConf().setMaster("local"))

spark = SparkSession \
    .builder \
    .getOrCreate()

#Creating a Dataframe from a local file
df = spark.read.option("header", "true").option("inferSchema", "true").csv(r'C:\******\Week 4\jfk_weather.csv')

#Creating a query table
df.createOrReplaceTempView('df')

#Lets import Translate and col from sql functions
from pyspark.sql.functions import translate, col

#Time to clean the dataset
df_cleaned = df \
    .withColumn("HOURLYWindSpeed", df.HOURLYWindSpeed.cast('double')) \
    .withColumn("HOURLYWindDirection", df.HOURLYWindDirection.cast('double')) \
    .withColumn("HOURLYStationPressure", translate(col("HOURLYStationPressure"), "s,", "")) \
    .withColumn("HOURLYPrecip", translate(col("HOURLYPrecip"), "s,", "")) \
    .withColumn("HOURLYRelativeHumidity", translate(col("HOURLYRelativeHumidity"), "*", "")) \
    .withColumn("HOURLYDRYBULBTEMPC", translate(col("HOURLYDRYBULBTEMPC"), "*", "")) \
        
df_cleaned =   df_cleaned \
     .withColumn("HOURLYStationPressure", df_cleaned.HOURLYStationPressure.cast('double')) \
     .withColumn("HOURLYPrecip", df_cleaned.HOURLYPrecip.cast('double')) \
     .withColumn("HOURLYRelativeHumidity", df_cleaned.HOURLYRelativeHumidity.cast('double')) \
     .withColumn("HOURLYDRYBULBTEMPC", df_cleaned.HOURLYDRYBULBTEMPC.cast('double')) \
    
df_filtered = df_cleaned.filter("""
    HOURLYWindSpeed <> 0
    and HOURLYWindSpeed IS NOT NULL
    and HOURLYWindDirection IS NOT NULL
    and HOURLYStationPressure IS NOT NULL
    and HOURLYPressureTendency IS NOT NULL
    and HOURLYPrecip IS NOT NULL
    and HOURLYRelativeHumidity IS NOT NULL
    and HOURLYDRYBULBTEMPC IS NOT NULL
""")   

#We want to predict one value based on others. For this, its helpful to print
#A correlation matrix

#Importing VectorAssembler, StringIndexer, Vectors, Normalizer and Pipeline
from pyspark.ml.feature import StringIndexer
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import Normalizer
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler

#Transforming HOURLYWindSpeed, HOURLYWindDirection, HOURLYStationPressure columns into a "features" column
vectorAssembler = VectorAssembler(inputCols=["HOURLYWindSpeed","HOURLYWindDirection","HOURLYStationPressure"], 
                                  outputCol="features")
df_pipeline = vectorAssembler.transform(df_filtered)

#Importing Correlation from ml.stat and checking the correlation for our features
from pyspark.ml.stat import Correlation
Correlation.corr(df_pipeline,"features").head()[0].toArray()

import random
random.seed(42)
#Spliting the dataset randomly into 80/20 train/test split
splits = df_filtered.randomSplit([0.8, 0.2])
df_train = splits[0]
df_test = splits[1]

#Transforming HOURLYWindSpeed, Elevation, HOURLYStationPressure columns into a "features" column
vectorAssembler = VectorAssembler(inputCols=[
                                    "HOURLYWindDirection",
                                    "ELEVATION",
                                    "HOURLYStationPressure"],
                                  outputCol="features")

#Normalizing the features colomn
normalizer = Normalizer(inputCol="features", outputCol="features_norm", p=1.0)


#Create a function to evaluate the regression prediction performance
#We use RMSE (Root Mean Squared Error)
def regression_metrics(prediction):
    from pyspark.ml.evaluation import RegressionEvaluator
    evaluator = RegressionEvaluator(labelCol="HOURLYWindSpeed", predictionCol="prediction", metricName="rmse")
    rmse = evaluator.evaluate(prediction)
    print("RMSE on test data = %g" % rmse)    
    

#Creating a Linear Regression model
#Importing Linear Regression 
from pyspark.ml.regression import LinearRegression
LR = LinearRegression(labelCol="HOURLYWindSpeed", featuresCol='features_norm', maxIter=100, regParam=0.0, elasticNetParam=0.0)
pipeline = Pipeline(stages=[vectorAssembler, normalizer, LR])
model = pipeline.fit(df_train)
prediction = model.transform(df_test)
regression_metrics(prediction)

#Lets try Gradient Boosted Tree Regressor
#Importing GBTRegressor
from pyspark.ml.regression import GBTRegressor
GBT = GBTRegressor(labelCol="HOURLYWindSpeed", maxIter=100)
pipeline = Pipeline(stages=[vectorAssembler, normalizer, GBT])
model = pipeline.fit(df_train)
prediction = model.transform(df_test)
regression_metrics(prediction)

#Lets turn this into CLASSIFICATION and predict HOURLYWindDirection
#We need to turn HOURLYWindDirection into a discrete value. We use Bucketizer

#We import Bucketizer
from pyspark.ml.feature import Bucketizer
#Lets setup the Bucketizer parameters
bucketizer = Bucketizer(splits=[0, 180, float('inf')], inputCol="HOURLYWindDirection", outputCol="HOURLYWindDirectionBucketized")


#We need to define a function to evaluate the model performance. 0 is bad, 1 is good 
def classification_metrics(prediction):
    from pyspark.ml.evaluation import MulticlassClassificationEvaluator
    MCEvaluator = MulticlassClassificationEvaluator().setMetricName("accuracy").setPredictionCol("prediction").setLabelCol("HOURLYWindDirectionBucketized")
    Accuracy = MCEvaluator.evaluate(prediction)
    print("Accuracy on test Data = %g" % Accuracy)

#Lets setup Logistic Regression as a baseline Classifier
from pyspark.ml.classification import LogisticRegression
LogisticRegression = LogisticRegression(labelCol="HOURLYWindDirectionBucketized", maxIter=10)

#Transforming HOURLYWindSpeed, HOURLYDRYBULBTEMPC columns into a "features" column
vectorAssembler = VectorAssembler(inputCols=["HOURLYWindSpeed","HOURLYDRYBULBTEMPC"],
                                  outputCol="features")

#Setting up pipeline and model
pipeline = Pipeline(stages=[bucketizer, vectorAssembler, normalizer, LogisticRegression])
model = pipeline.fit(df_train)
prediction = model.transform(df_test)
classification_metrics(prediction)

#Lets try RandomForestClassifier
from pyspark.ml.classification import RandomForestClassifier
RF = RandomForestClassifier(labelCol="HOURLYWindDirectionBucketized", numTrees=10)
vectorAssembler = VectorAssembler(inputCols=["HOURLYWindSpeed","HOURLYDRYBULBTEMPC","ELEVATION","HOURLYStationPressure","HOURLYPressureTendency","HOURLYPrecip"],
                                  outputCol="features")

pipeline = Pipeline(stages=[bucketizer, vectorAssembler, normalizer, RF])
model = pipeline.fit(df_train)
prediction = model.transform(df_test)
classification_metrics(prediction)

#Lets try GBT Classifier
from pyspark.ml.classification import GBTClassifier
GBT = GBTClassifier(labelCol="HOURLYWindDirectionBucketized", maxIter=100)
vectorAssembler = VectorAssembler(inputCols=["HOURLYWindSpeed","HOURLYDRYBULBTEMPC","ELEVATION","HOURLYStationPressure","HOURLYPressureTendency","HOURLYPrecip"],
                                  outputCol="features")

pipeline = Pipeline(stages=[bucketizer, vectorAssembler, normalizer, GBT])
model = pipeline.fit(df_train)
prediction = model.transform(df_test)
classification_metrics(prediction)






