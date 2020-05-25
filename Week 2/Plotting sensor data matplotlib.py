from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession

sc = SparkContext.getOrCreate(SparkConf().setMaster("local[*]"))

spark = SparkSession \
    .builder \
    .getOrCreate()

#Creating Dataframe
df = spark.read.parquet(r'C:\******\Week 2\washing.parquet')
df.show(10)

#Number of rows
df.count()

#creating a corresponding query tbale
df.createOrReplaceTempView('washing')
df.printSchema()
spark.sql("SELECT * FROM washing").show()

#Obtain insights on how voltage behaves
result = spark.sql("select voltage from washing where voltage is not null")
result.collect()
#values are wrapped in a row object. Need to get rid of them
result.rdd.map(lambda row : row.voltage).take(10)
#Using sample functions to extract some data/ fraction 0.1 or 10% for plotting
result_array = result.rdd.map(lambda row : row.voltage).sample(False,0.1).collect()
result_array

#Above is a meaningful array containing integers values for voltage 
#from CouchDB NoSQL database

# %matplotlib inline (This code must be used in jupyter to display plots under code cell)

#Importing pyplot from matplotlib
import matplotlib.pyplot as plt

#Creating a box plot with result_array as parameter
plt.boxplot(result_array)
plt.show()
#Median is the orange line on the plot
#Outliers are individual points on the plot

##############################################################################

#Run charts (like stock market charts) are perfect for time series data
#Run charts x-axis is always 'Time'
#Run charts y-axis is obseved value(s) over time
#To create a Run Chart we need the 'Time' dimension from the dataset. 
#Timestamp ts needs to be added

result = spark.sql("select voltage,ts from washing where voltage is not null order by ts asc")
#We sample first to reduce processing time of subsequent steps
result_rdd = result.rdd.sample(False,0.1).map(lambda row : (row.ts,row.voltage))
result_array_ts = result_rdd.map(lambda ts_voltage: ts_voltage[0]).collect()
result_array_voltage = result_rdd.map(lambda ts_voltage: ts_voltage[1]).collect()
print(result_array_ts[:15])
print(result_array_voltage[:15])


#plotting a Run Chart
plt.plot(result_array_ts,result_array_voltage)
plt.xlabel("time")
plt.ylabel("voltage")
plt.show()



spark.sql("select min(ts),max(ts) from washing").show()

#Showing timestamp data for an hour.
#Timestamp are the number of millisecons passed since the 1st of Jan. 1970
#Interval of 60 minutes (10006060)=3600000

result_timestamp =spark.sql(
"""
select voltage,ts from washing 
    where voltage is not null and 
    ts > 1547808720911 and
    ts <= 1547810064867+3600000
    order by ts asc
""")

result_timestamp_rdd = result_timestamp.rdd.map(lambda row : (row.ts,row.voltage))
result_array_ts1 = result_timestamp_rdd.map(lambda ts_voltage: ts_voltage[0]).collect()
result_array_voltage1 = result_timestamp_rdd.map(lambda ts_voltage: ts_voltage[1]).collect()

from matlplotlib.pyplot import plot as plt
plt.plot(result_array_ts1,result_array_voltage1)
plt.xlabel("time")
plt.ylabel("voltage")
plt.show()



#Scatter plots plot individual data points in 2 or 3 dimensional space
#Scatter plots can be use for classificaiton boundaries, clusters and detecting anomalies
#Plotting scatter plots


result_df = spark.sql("""
select hardness,temperature,flowrate from washing
    where hardness is not null and 
    temperature is not null and 
    flowrate is not null
""")
#We sample first to reduce processing time of subsequent steps
#We need to unwrap the row objects
result_rdd1 = result_df.rdd.sample(False,0.1).map(lambda row : (row.hardness,row.temperature,row.flowrate))
result_array_hardness = result_rdd1.map(lambda hardness_temperature_flowrate: hardness_temperature_flowrate[0]).collect()
result_array_temperature = result_rdd1.map(lambda hardness_temperature_flowrate: hardness_temperature_flowrate[1]).collect()
result_array_flowrate = result_rdd1.map(lambda hardness_temperature_flowrate: hardness_temperature_flowrate[2]).collect()

print(result_array_hardness[:15])
print(result_array_temperature[:15])
print(result_array_flowrate[:15])

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(result_array_hardness,result_array_temperature,result_array_flowrate, c='r', marker='o')

ax.set_xlabel('hardness')
ax.set_ylabel('temperature')
ax.set_zlabel('flowrate')

plt.show()


#Histograms
plt.hist(result_array_hardness)
plt.show()
plt.hist(result_array_temperature)
plt.show()
plt.hist(result_array_flowrate)
plt.show()











