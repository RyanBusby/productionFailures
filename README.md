# __Binary Classification with Apache Spark / HDFS__

[<img src="img/logo.png" style="width: 5px;"/>](https://www.kaggle.com/c/bosch-production-line-performance/data) __↖data source__

#### The goal of the competition is to predict which parts will fail quality control
#### My goal is to utilize the hadoop ecosystem to handle a large dataset and establish a pipeline for machine learning

The data are munged into a lower dimension using Apache-Spark's DataFrame class. Each row is summarized by the count of non-null entries, the average of the non-null entries, that average squared, and the natural log of that average.

A column is added that indicates outliers. All of that is done with [`src/munge.py`](src/munge.py).

The munged data is modeled with Spark Machine Learning package, predictions are made on the test data, then loaded into hdfs. Those steps execute in [`src/fit_predict.py`](src/fit_predict.py).

##### To easily run a toy example:
run [```munge_fit_predict.py```](src/munge_fit_predict.py)
Enter 1 when prompted to launch a local spark cluster. Predictions will populate in the data directory of this repo.
