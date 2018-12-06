# __Binary Classification with Apache Spark / HDFS__

[<img src="img/logo.png" style="width: 5px;"/>](https://www.kaggle.com/c/bosch-production-line-performance/data) __↖data source__

#### The goal of the competition is to predict which parts will fail quality control
#### My goal is to utilize the hadoop ecosystem to handle a large dataset and establish a pipeline for machine learning

#### [`munge`](src/munge.py) :
- Aggregate columns using RDD transformations
- Create a column that indicates which of those column aggregations are outliers.

#### [`fit_predict`](src/fit_predict.py) :
- Model data with Spark Machine Learning package
- Predict on test data

#### [`munge_fit_predict`](src/munge_fit_predict.py) :
- Run this as is to use the toy data set example
