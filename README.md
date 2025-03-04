# Understanding Pandas UDF in Apache Spark 3.0

## Overview
This repository provides an in-depth guide to **User-Defined Functions (UDFs)** in Apache Spark 3.0 using Databricks notebooks. The notebook demonstrates:

1. Defining a function.
2. Creating and applying a UDF.
3. Registering the UDF for SQL usage.
4. Using Python decorator syntax for UDF creation.
5. Implementing Pandas (vectorized) UDFs.

## Databricks Notebook Source Code

### Define a Function
```python
# A UDF that takes a number and returns its square
def squared(s):
    return s * s
```

### Create and Apply UDF
```python
from pyspark.sql.functions import udf
square_udf = udf(squared)
```

### Define an Example DataFrame
```python
test_df = spark.range(1, 20)
display(test_df)
```

### Apply the UDF
```python
from pyspark.sql.functions import col
display(test_df.select(square_udf(col("id"))))
```

### Register UDF for SQL Usage
```python
test_df.createOrReplaceTempView("test")
square_udf = spark.udf.register("square", squared)
```
```sql
SELECT square(id) AS squared_value FROM test;
```
```python
display(test_df.select(square_udf(col("id"))))
```

### Use Decorator Syntax (Python Only)
```python
from pyspark.sql.functions import udf
@udf("long")
def squared(s):
    return s * s

test_df = spark.range(1, 20)
display(test_df.select("id", squared("id").alias("id_squared")))
```

## Different Types of Pandas UDFs

### Series to Series UDF
```python
import pandas as pd
from pyspark.sql.functions import col, pandas_udf
from pyspark.sql.types import LongType

def square(a: pd.Series) -> pd.Series:
    return a * a

square_pandas_udf = pandas_udf(square, returnType=LongType())
test_df.select(square_pandas_udf(col("id"))).show()
```

### Iterator of Series to Iterator of Series UDF
```python
from typing import Iterator
@pandas_udf("long")
def square_plus_one(batch_iter: Iterator[pd.Series]) -> Iterator[pd.Series]:
    for x in batch_iter:
        yield x * x + 1

test_df.select("id", square_plus_one(col("id"))).show()
```

### Iterator of Multiple Series to Iterator of Series UDF
```python
from typing import Iterator, Tuple
pdf = pd.DataFrame([("red",2,9), ("red",4,9), ("red", 6, 11), ("white", 5, 12), ("white", 3, 11), ("white", 1, 11)], columns=["wine_type", "wine_quality", "alcohol"])
df = spark.createDataFrame(pdf)

@pandas_udf("long")
def sum_winequality_alcohol(iterator: Iterator[Tuple[pd.Series, pd.Series]]) -> Iterator[pd.Series]:
    for wine_quality, alcohol in iterator:
        yield wine_quality + alcohol

df.select("wine_type", "wine_quality", "alcohol", sum_winequality_alcohol("wine_quality", "alcohol")).show()
```

### Series to Scalar UDF
```python
from pyspark.sql.functions import pandas_udf
from pyspark.sql import Window

@pandas_udf("double")
def mean_value(column: pd.Series) -> float:
    return column.mean()

df.select(mean_value(df['wine_quality'])).show()
```

## License
This repository is open for educational purposes. Feel free to modify and improve the content.
