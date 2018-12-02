# __Binary Classification with Apache Spark / HDFS__

[<img src="img/logo.png" style="width: 5px;"/>](https://www.kaggle.com/c/bosch-production-line-performance/data) __↖data source__

#### The goal of the competition is to predict which parts will fail quality control
#### My goal is to utilize the hadoop ecosystem to handle a large dataset and establish a pipeline for machine learning

The data are munged into a lower dimension using Apache-Spark's DataFrame class. Each row is summarized by the count of non-null entries, the average of the non-null entries, that average squared, and the natural log of that average.

A column is added that indicates outliers. All of that is done with [`src/myMunge.py`](src/myMunge.py).

The munged data is modeled with Spark Machine Learning package, predictions are made on the test data, then loaded into hdfs. Those steps execute in [`src/fitLR.py`](src/fitLR.py).

##### To easily run a toy example:
1. run myMunge.py, enter 1 when prompted to launch locally.

2. copy the name of the newly created folder (not the path) within the data directory to the train_fname and test_fname variables in src/fitLR.py, accordingly.

3. run fitLR.py, enter 1 when prompted to launch locally.

4. predictions will populate in the data directory of this repo
