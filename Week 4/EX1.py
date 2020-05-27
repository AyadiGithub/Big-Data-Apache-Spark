from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession


sc = SparkContext.getOrCreate(SparkConf().setMaster("local"))

spark = SparkSession \
    .builder \
    .getOrCreate()

df_data_1 = spark.read.parquet(r'C:\*****\Week 4\hmp.parquet')
df = df_data_1
df.show()
df.createOrReplaceTempView('df')

#Creating a random split
splits = df.randomSplit([0.8,0.2])
df_train = splits[0]
df_test = splits[1]

#Importing StringIndexer, OneHotEncoder, VectorAssembler, Normalizer from ml features
#Importing Vectors from Vectors
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, Normalizer
from pyspark.ml.linalg import Vectors

index = StringIndexer(inputCol="class", outputCol="label")
vectorAssembler = VectorAssembler(inputCols=["x", "y", "z"],
                                  outputCol="features")

normalizer = Normalizer(inputCol="features", outputCol="features_norm", p=1.0)


#Lets Try Random Forest Classifier
#Importing RandomForestClassifier
from pyspark.ml.classification import RandomForestClassifier

RF = RandomForestClassifier(featuresCol="features", labelCol="label", numTrees=20)

#Importing Pipeline and setting it up
from pyspark.ml import Pipeline
pipeline = Pipeline(stages=[index, vectorAssembler, normalizer, RF])

#fitting training data as model
model = pipeline.fit(df_train)
#Importing a model evaluator (MulticlassClassificationEvaluator) and evaluator our model
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
Evaluator = MulticlassClassificationEvaluator().setMetricName('accuracy').setLabelCol('label').setPredictionCol('prediction')

#Setting f1 score measure
Evaluator1 = MulticlassClassificationEvaluator().setMetricName('f1').setLabelCol('label').setPredictionCol('prediction')

#Lets run the test data
predictions = model.transform(df_test)

#Evaluating with accuracy measure
Evaluator.evaluate(predictions)

#Evaluating with f1 score measure
Evaluator1.evaluate(predictions)
