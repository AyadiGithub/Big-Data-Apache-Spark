from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession


sc = SparkContext.getOrCreate(SparkConf().setMaster("local"))

spark = SparkSession \
    .builder \
    .getOrCreate()

df_data_1 = spark.read.parquet(r'C:\******\Week 4\hmp.parquet')
df = df_data_1
df.show()
df.createOrReplaceTempView('df')

#Our hmp.parquet data is more suitable for classification so lets engineer a new column for LinearR
#This column will be the Eneregy
df_energy = spark.sql("""

select sqrt(sum(x*x)+sum(y*y)+sum(z*z)) as label, class from df group by class
          
          
          
          
          
          
          
          
""")

df_energy.show()

df_energy.createOrReplaceTempView('df_energy')

#Joining df and df_energy together

from pyspark.sql.functions import *
df = df.alias("df")
df_energy = df_energy.alias("df_energy")

df_join = df.join(df_energy, col("df.class") == col("df_energy.class"), 'inner')

df_join.show()


#Importing VectorAssembler and Normalizer 
from pyspark.ml.feature import VectorAssembler, Normalizer
#Assembling columns x,y,z into a features column
vectorAssembler = VectorAssembler(inputCols=["x","y","z"],
                                  outputCol="features")

normalizer = Normalizer(inputCol="features", outputCol="features_norm", p=1.0)

#Importing Linear Regression models and setting the parameters for it
from pyspark.ml.regression import LinearRegression
LR = LinearRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)

#Importing and creating a pipeline
from pyspark.ml import Pipeline
pipeline = Pipeline(stages=[vectorAssembler, normalizer, LR])

#Creating and running the LR model
model = pipeline.fit(df_join)
predictions = model.transform(df_join)

#Using r2 measure to see the performance of the model
model.stages[2].summary.r2




















