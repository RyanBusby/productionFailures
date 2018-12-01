# __Binary Classification with Apache Spark / HDFS__

[<img src="img/logo.png" style="width: 5px;"/>](https://www.kaggle.com/c/bosch-production-line-performance/data) __↖data source__

#### The goal of the competition is to predict which parts will fail quality control.
#### My goal is to utilize the hadoop ecosystem to handle a large dataset and perform machine learning.

The data are munged into a lower dimension utilizing Apache-Spark's DataFrame class. Each row is summarized by counts of non-null entries, the average of the row, that average squared, and the natural log of that average. Those are examples of aggregating along the columns.

A column is added that indicates outliers, which is an example of aggregation along the rows. All of that is done with [`src/myMunge.py`](src/myMunge.py).

The munged data is modeled with Spark Machine Learning package, predictions are made on the test data, then loaded into hdfs. Those steps execute in [`src/fitLR.py`](src/fitLR.py).

To easily run a toy example, change the first three lines under `if __name__ == '__main__':` in both [`src/myMunge.py`](src/myMunge.py) and [`src/fitLR.py`](src/fitLR.py) to:

```python
if __name__ == '__main__':
  sparkContext = ps.SparkContext('local[2]')
  spark = ps.sql.SparkSession(sparkContext)
  root = '../data/%s'
```

[Hadoop Docs](https://hadoop.apache.org/docs/r3.1.1/hadoop-project-dist/hadoop-common/SingleCluster.html#Configuration) // [Spark docs](https://spark.apache.org/docs/2.4.0/spark-standalone.html#starting-a-cluster-manually)
