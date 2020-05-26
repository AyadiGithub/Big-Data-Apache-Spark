from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType


sc = SparkContext.getOrCreate(SparkConf().setMaster("local[*]"))

spark = SparkSession \
    .builder \
    .getOrCreate()


#Creating a schema for the dataset
schema = StructType([
    StructField("x", IntegerType(), True),
    StructField("y", IntegerType(), True),
    StructField("z", IntegerType(), True)])

import os
#specifying dataset directory and filtering out unwanted files
file_list = os.listdir(r'C:/*******/HMP_Dataset')
file_list_filtered = [s for s in file_list if '_' in s]
file_list_filtered

df = None

from pyspark.sql.functions import lit

#Creating a dataframe out of every file in HMP_Dataset with the schema created
for category in file_list_filtered:
    #Creating a category for each folder name
    data_files = os.listdir(r'C:/******/HMP_Dataset/'+category)
    for data_file in data_files:
        print(data_file)
        
        #Creating a spark df for each datafile
        temp_df = spark.read.option("header", "false").option("delimiter", " ").csv('C:/*****/HMP_Dataset/'+category+'/'+data_file,schema=schema)
        
        #Creating column 'Class' with folder name and column 'source' with datafile name.
        temp_df = temp_df.withColumn('class', lit(category))
        temp_df = temp_df.withColumn('source', lit(data_file))
        if df is None:
            df = temp_df
        else:
            df = df.union(temp_df)

df.show()
    

#   Now we transform class to integer (String indexer)
from pyspark.ml.feature import StringIndexer
indexer = StringIndexer(inputCol='class', outputCol='ClassIndex')
#Using indexer to fit the df and transform it into integers
indexed = indexer.fit(df).transform(df)

indexed.show()


# Now we use OneHotEnconding on the column
from pyspark.ml.feature import OneHotEncoder

encoder = OneHotEncoder(inputCol="ClassIndex", outputCol="categoryVec")
encoded = encoder.transform(indexed)
#Lets see the sparsevector 
encoded.show()


#Sparkml only works on vector objects so columns x,y,z must be transformed

from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler

VectorAssembler = VectorAssembler(inputCols=['x','y','z'],
                                  outputCol='features')

features_vectorized = VectorAssembler.transform(encoded)


features_vectorized.show()


#Normalizing is not necesary for the transformed data due to having similar range of integers
#We will Normalize anyway

from pyspark.ml.feature import Normalizer

normalizer = Normalizer(inputCol="features", outputCol="features_norm", p=1.0)
#Values will be between 0 and 1
normalized_data = normalizer.transform(features_vectorized)
normalized_data.show()



#Creating a Pipeline Object

from pyspark.ml import Pipeline
#Now we retrace our steps and slot them into the pipeline
pipeline = Pipeline(stages=[indexer,encoder,VectorAssembler,normalizer])

model = pipeline.fit(df)
prediction = model.transform(df)
prediction.show()

#Removing undeeded columns

df_train = prediction.drop('x').drop('y').drop('z').drop('class').drop('source').drop('ClassIndex').drop('features')
df_train.show()



