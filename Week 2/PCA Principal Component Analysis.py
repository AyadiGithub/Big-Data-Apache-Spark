#Using PCA (Principal Component Analysis) we can TRANSFORM/reduce a dataset from n multi
#dimensions into a desired number of dimensions k - this is also called Projection
#PCA preserves the ratio of distances between points
#This means:
#dist(Pa,Pb)/dist(Pc,Pd) In R^n Dataset = dist(Pa,Pb)/dist(Pc,Pd) In reduced R^k Dataset
#Components in R^k have the least correlation with each other and called Principal Components
#When applying PCA, we lose information BUT PCA is intelligent in deciding the information that is
#less relevant which minimizes the loss of informaiton. Example: JPEG Compression
#The smaller K chosen, the higher is the information loss. 
#The amount of information loss can be quantified using Sum Squared Error as an example method. 

from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession

sc = SparkContext.getOrCreate(SparkConf().setMaster("local[*]"))

spark = SparkSession \
    .builder \
    .getOrCreate()

#Creating Dataframe
df = spark.read.parquet(r'C:\*****\Week 2\washing.parquet')
df.show(10)

#Number of rows
df.count()

#creating a corresponding query tbale
df.createOrReplaceTempView('washing')
df.printSchema()
spark.sql("SELECT * FROM washing").show()

result = spark.sql("""
SELECT * from (
    SELECT
    min(temperature) over w as min_temperature,
    max(temperature) over w as max_temperature, 
    min(voltage) over w as min_voltage,
    max(voltage) over w as max_voltage,
    min(flowrate) over w as min_flowrate,
    max(flowrate) over w as max_flowrate,
    min(frequency) over w as min_frequency,
    max(frequency) over w as max_frequency,
    min(hardness) over w as min_hardness,
    max(hardness) over w as max_hardness,
    min(speed) over w as min_speed,
    max(speed) over w as max_speed
    FROM washing 
    WINDOW w AS (ORDER BY ts ROWS BETWEEN CURRENT ROW AND 10 FOLLOWING) 
)
WHERE min_temperature is not null 
AND max_temperature is not null
AND min_voltage is not null
AND max_voltage is not null
AND min_flowrate is not null
AND max_flowrate is not null
AND min_frequency is not null
AND max_frequency is not null
AND min_hardness is not null
AND min_speed is not null
AND max_speed is not null
""")

result.count()


#Importing PCA, Vectors, VectorAssembler from pyspark.ml
from pyspark.ml.feature import PCA
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler

#Spark PCA runs parallel on a cluster and is faster than scikitlearn PCA
#Spark does not deal with python arrays
#Spark needs its ml vector object 
#We can use VectorAssembler to convert the arrays to vector objects

#Vector assembler will take all 12 columns and convert them into features column
assembler = VectorAssembler(inputCols=result.columns, outputCol="features")

#Now we need to transform input data
#The result is a DF with the orginal columns and an addtional features column
features = assembler.transform(result)
features.show(10)

features.rdd.map(lambda r: r.features).take(10)
#The above code will give us a subtype of vector object a 'DenseVector'
#There exists another subtype called sparsevector that contains a lot of zeros to save space
#A list of sparsevectors is a sparsematrix

#We define our input and output and the k dimensions
pca = PCA(k=3, inputCol="features", outputCol="pcaFeatures")
#model will calculate the transformaiton rules to perform transformation
model = pca.fit(features)


#Performing the transformation
result_pca = model.transform(features).select("pcaFeatures")
result_pca.show(truncate=False)

#result_pca is the reduced dataset, it must containthe same number of rows
result_pca.count()


#Creating rdd from a sample of data 10%
rdd = result_pca.rdd.sample(False,0.1)

#Extracting features as python arrays

x = result_pca.rdd.map(lambda a: a.pcaFeatures).map(lambda a: a[0]).collect()
y = result_pca.rdd.map(lambda a: a.pcaFeatures).map(lambda a: a[1]).collect()
z = result_pca.rdd.map(lambda a: a.pcaFeatures).map(lambda a: a[2]).collect()

#Now we created 3 arrays and can use them in 3D matplotlib

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(x,y,z, c='r', marker='o')

#x,y,z new dimension arrays arent off specific features so we give them a generic name

ax.set_xlabel('dimension1')
ax.set_ylabel('dimension2')
ax.set_zlabel('dimension3')

plt.show()



