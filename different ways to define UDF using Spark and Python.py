# Databricks notebook source
# MAGIC %md # User-Defined Functions
# MAGIC 
# MAGIC ##### In this notebook we will cover the following steps
# MAGIC 1. Define a function
# MAGIC 1. Create and apply a UDF
# MAGIC 1. Register the UDF to use in SQL
# MAGIC 1. Create and register a UDF with Python decorator syntax
# MAGIC 1. Create and apply a Pandas (vectorized) UDF

# COMMAND ----------

# MAGIC %md ### Define a function
# MAGIC 
# MAGIC Define a function that takes as input a number and returns the square of it.

# COMMAND ----------

# A UDF that takes as input a number and squares it
def squared(s):
  return s * s

# COMMAND ----------

# MAGIC %md ### Create and apply UDF
# MAGIC Register the function as a UDF. This serializes the function and sends it to executors to be able to transform DataFrame records.

# COMMAND ----------

square_udf = udf(squared)

# COMMAND ----------

# MAGIC %md ### Define an example Dataframe to work with.

# COMMAND ----------

#defining a sample dataframe that has 19 rows and values range from 1-19
test_df = spark.range(1, 20)

# COMMAND ----------

display(test_df)

# COMMAND ----------

# MAGIC %md Apply the UDF on the `id` column.

# COMMAND ----------

from pyspark.sql.functions import col

display(test_df.select(square_udf(col("id"))))

# COMMAND ----------

# MAGIC %md ### Register UDF to use in SQL
# MAGIC Register the UDF using `spark.udf.register` to also make it available for use in the SQL namespace.

# COMMAND ----------

test_df.createOrReplaceTempView("test")

square_udf = spark.udf.register("square", squared)

# COMMAND ----------

# MAGIC %sql
# MAGIC -- You can now also apply the UDF from SQL
# MAGIC SELECT square(id) AS squared_value FROM test

# COMMAND ----------

# You can still apply the UDF from Python
display(test_df.select(square_udf(col("id"))))

# COMMAND ----------

# MAGIC %md ### Use Decorator Syntax (Python Only)
# MAGIC 
# MAGIC A Python UDF can be defined using <a href="https://realpython.com/primer-on-python-decorators/" target="_blank">Python decorator syntax</a>. The `@udf` decorator parameter is the Column datatype the function returns.
# MAGIC 
# MAGIC If we [clear](https://docs.databricks.com/notebooks/notebooks-use.html#clear-notebooks-state-and-results) the state of the notebook and execute only the cell `cmd 16` we will no longer be able to call the local Python function (i.e., `display(test_df.select(square_udf(col("id"))))")` will not work).

# COMMAND ----------

from pyspark.sql.functions import udf
@udf("long")
def squared(s):
  return s * s

test_df = spark.range(1, 20)

#read the table in memory as a spark dataframe
display(test_df.select("id", squared("id").alias("id_squared")))

# COMMAND ----------

display(test_df.select(square_udf(col("id"))))

# COMMAND ----------

# MAGIC %md
# MAGIC # Different types of Pandas UDFs

# COMMAND ----------

# MAGIC %md
# MAGIC ## Series to Series UDF
# MAGIC 1. These UDF can be used with `select` and `withColumn`
# MAGIC 2. Takes pandas Series as inputs and returns pandas Series of same length.
# MAGIC 3. We need to define return Python type hint.

# COMMAND ----------

import pandas as pd
from pyspark.sql.functions import col, pandas_udf
from pyspark.sql.types import LongType

# Declare the function and create the UDF
def square(a: pd.Series) -> pd.Series:
    return a * a

square_pandas_udf = pandas_udf(square, returnType=LongType())

test_df = spark.range(1, 20)

# Execute function as a Spark vectorized UDF
test_df.select(square_pandas_udf(col("id"))).show()
# +----------+
# |square(id)|
# +----------+
# |         1|
# |         4|
# |         9|
# |        16|
# |        25|
# |        36|
# |        49|
# |        64|
# |        81|
# |       100|
# |       121|
# |       144|
# |       169|
# |       196|
# |       225|
# |       256|
# |       289|
# |       324|
# |       361|
# +----------+



# COMMAND ----------

#multiple series as input example
test_df_multiple_columns =  test_df.withColumn("squared_id", square_pandas_udf(col("id")))

# Declare the function and create the UDF
def sum_multiple_series(a: pd.Series, b: pd.Series) -> pd.Series:
    return a + b

sum_multiple_series_pandas_udf = pandas_udf(sum_multiple_series, returnType=LongType())

# Execute function as a Spark vectorized UDF
test_df_multiple_columns.select("id", "squared_id", sum_multiple_series_pandas_udf(col("id"), col("squared_id"))).show()

# +---+----------+-----------------------------------+
# | id|squared_id|sum_multiple_series(id, squared_id)|
# +---+----------+-----------------------------------+
# |  1|         1|                                  2|
# |  2|         4|                                  6|
# |  3|         9|                                 12|
# |  4|        16|                                 20|
# |  5|        25|                                 30|
# |  6|        36|                                 42|
# |  7|        49|                                 56|
# |  8|        64|                                 72|
# |  9|        81|                                 90|
# | 10|       100|                                110|
# | 11|       121|                                132|
# | 12|       144|                                156|
# | 13|       169|                                182|
# | 14|       196|                                210|
# | 15|       225|                                240|
# | 16|       256|                                272|
# | 17|       289|                                306|
# | 18|       324|                                342|
# | 19|       361|                                380|
# +---+----------+-----------------------------------+


# COMMAND ----------

# MAGIC %md
# MAGIC ## Iterator of Series to Iterator of Series UDF
# MAGIC - This UDF takes an iterator of batches as input and returns an iterator or batches as output. 
# MAGIC - This UDF only takes as input a single Spark Dataframe column.
# MAGIC - The length of the entire input iterator and the output iterator should be same.
# MAGIC - We need to specify `Iterator[pandas.Series] -> Iterator[pandas.Series]` as type hint.
# MAGIC 
# MAGIC This type of UDF is specially useful if you have a usecase that requires initializing some state for example, loading a trained machine learning model file to apply inference to every input batch.
# MAGIC 
# MAGIC The following example shows how to create a pandas UDF with iterator support.

# COMMAND ----------

import pandas as pd
from typing import Iterator
from pyspark.sql.functions import col, pandas_udf, struct

test_df = spark.range(1, 20)

# When the UDF is called with the column,
# the input to the underlying function is an iterator of pd.Series.
@pandas_udf("long")
def square_plus_one(batch_iter: Iterator[pd.Series]) -> Iterator[pd.Series]:
    for x in batch_iter:
        yield x * x + 1

test_df.select("id", square_plus_one(col("id"))).show()
# +---+-------------------+
# | id|square_plus_one(id)|
# +---+-------------------+
# |  1|                  2|
# |  2|                  5|
# |  3|                 10|
# |  4|                 17|
# |  5|                 26|
# |  6|                 37|
# |  7|                 50|
# |  8|                 65|
# |  9|                 82|
# | 10|                101|
# | 11|                122|
# | 12|                145|
# | 13|                170|
# | 14|                197|
# | 15|                226|
# | 16|                257|
# | 17|                290|
# | 18|                325|
# | 19|                362|
# +---+-------------------+


# COMMAND ----------

# In the UDF, you can initialize some state before processing batches.
# Wrap your code with try/finally or use context managers to ensure
# the release of resources at the end.
y_bc = spark.sparkContext.broadcast(50)

#if you want to broadcast a ml model that is less than 2GB size and is serializable you can broadcast it as well as shown below
# model = spark.context.broadcast(<trained_model>)

@pandas_udf("long")
def square_plus_y(batch_iter: Iterator[pd.Series]) -> Iterator[pd.Series]:
    y = y_bc.value  # initialize states
    try:
        for x in batch_iter:
            yield x * x + y
    finally:
        pass  # release resources here, if any

test_df.select(square_plus_y(col("id"))).show()
# +-----------------+
# |square_plus_y(id)|
# +-----------------+
# |               51|
# |               54|
# |               59|
# |               66|
# |               75|
# |               86|
# |               99|
# |              114|
# |              131|
# |              150|
# |              171|
# |              194|
# |              219|
# |              246|
# |              275|
# |              306|
# |              339|
# |              374|
# |              411|
# +-----------------+

# COMMAND ----------

# MAGIC %md
# MAGIC ## Iterator of multiple Series to Iterator of Series UDF
# MAGIC 
# MAGIC - An Iterator of multiple Series to Iterator of Series UDF is similar to the Iterator of Series to Iterator of Series UDF; however, it takes multiple Spark Dataframe columns as input.
# MAGIC - The underlying Python function expects an iterator of tuple of pandas Series to be passed as input and returns an iterator of pandas Series of similar size as input as output.
# MAGIC - We need to define type hints as `Iterator[Tuple[pandas.Series, …]] -> Iterator[pandas.Series]`

# COMMAND ----------

from typing import Iterator, Tuple
import pandas as pd

#lets create a dummy dataframe with three columns wine_type, wine_quality, alcohol
pdf = pd.DataFrame([("red",2,9), ("red",4,9), ("red", 6, 11), ("white", 5, 12), ("white", 3, 11), ("white", 1, 11)], columns=["wine_type", "wine_quality", "alcohol"])
df = spark.createDataFrame(pdf)

@pandas_udf("long")
def sum_winequality_alcohol(
        iterator: Iterator[Tuple[pd.Series, pd.Series]]) -> Iterator[pd.Series]:
    for wine_quality, alocohol in iterator:
        yield wine_quality + alocohol
        
df.select("wine_type", "wine_quality", "alcohol", sum_winequality_alcohol("wine_quality", "alcohol")).show()     

# +---------+------------+-------+----------------------------------------------+
# |wine_type|wine_quality|alcohol|sum_winequality_alcohol(wine_quality, alcohol)|
# +---------+------------+-------+----------------------------------------------+
# |      red|           2|      9|                                            11|
# |      red|           4|      9|                                            13|
# |      red|           6|     11|                                            17|
# |    white|           5|     12|                                            17|
# |    white|           3|     11|                                            14|
# |    white|           1|     11|                                            12|
# +---------+------------+-------+----------------------------------------------+

# COMMAND ----------

# MAGIC %md
# MAGIC ## Series to scalar UDF
# MAGIC - As the name suggests, series to scalar Pandas UDFs help define aggregation from one or more pandas Series to a scalar value. They are similar to Spark aggregate functions.
# MAGIC - These UDFs are utilized with groupby.agg and pyspark.sql.Window and perform an aggregation over all the data for a given group.
# MAGIC - All the data for a group is loaded in memory.
# MAGIC - We need to provide the python hint as follows:
# MAGIC `pandas.Series, ... -> Any`
# MAGIC - Any is a Python primitive datatype or a numpy scalar datatype.

# COMMAND ----------

import pandas as pd
from pyspark.sql.functions import pandas_udf
from pyspark.sql import Window


pdf = pd.DataFrame([("red",2,9), ("red",4,9), ("red", 6, 11), ("white", 5, 12), ("white", 3, 11), ("white", 1, 11)], columns=["wine_type", "wine_quality", "alcohol"])
df = spark.createDataFrame(pdf)


# Declare a function to calculate mean
@pandas_udf("double")
def mean_value(column: pd.Series) -> float:
    return column.mean()

df.select(mean_value(df['wine_quality'])).show()

# +------------------------+
# |mean_value(wine_quality)|
# +------------------------+
# |                     3.5|
# +------------------------+

df.groupby("wine_type").agg(mean_value(df['wine_quality'])).show()
# +---------+------------------------+
# |wine_type|mean_value(wine_quality)|
# +---------+------------------------+
# |      red|                     4.0|
# |    white|                     3.0|
# +---------+------------------------+

w = Window \
    .partitionBy('wine_type') \
    .rowsBetween(Window.unboundedPreceding, Window.unboundedFollowing)
df.withColumn('mean_value_alcohol', mean_value(df['alcohol']).over(w)).show()
# +---------+------------+-------+------------------+
# |wine_type|wine_quality|alcohol|mean_value_alcohol|
# +---------+------------+-------+------------------+
# |      red|           2|      9| 9.666666666666666|
# |      red|           4|      9| 9.666666666666666|
# |      red|           6|     11| 9.666666666666666|
# |    white|           5|     12|11.333333333333334|
# |    white|           3|     11|11.333333333333334|
# |    white|           1|     11|11.333333333333334|
# +---------+------------+-------+------------------+

# COMMAND ----------


