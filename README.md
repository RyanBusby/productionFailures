# __Binary Classification with Apache Spark / HDFS__

[<img src="img/logo.png" style="width: 5px;"/>](https://www.kaggle.com/c/bosch-production-line-performance/data) ↖  __the data__

#### The goal of the competition is to predict which parts will fail quality control, my goal is to utilize the hadoop ecosystem to handle a large dataset and perform machine learning.

The data are munged into a lower dimension utilizing Apache-Spark's DataFrame class. Each row is summarized by counts of non-null entries, the average of the row, that average squared, and the natural log of that average. Those are examples of aggregating along the columns.

Then there is a column added in that indicates outliers, which is an example of aggregation along the rows. All of that is done with [src/myMunge.py](src/myMunge.py).

The munged data is modeled with Spark Machine Learning package, and predictions are made on the non labeled data and loaded into hdfs. Those steps execute in [src/fitLR.py](src/fitLR.py).
