# __Binary Classification with Apache HDFS/Spark__

[<img src="img/logo.png" style="width: 5px;"/>](https://www.kaggle.com/c/bosch-production-line-performance/data) ↖  __the data__

#### The goal of the competition is to predict which parts will fail quality control, my goal is to utilize the hadoop ecosystem to handle a large datasets and perform machine learning.

[The data are munged into a lower dimension utilizing Apache-Spark's DataFrame class. Each record is summarized by counts of non null entries, the average of the row measures, that average squared, and the natural log of that average. All of that is done with myMunge.py](src/myMunge.py)

[The munged data is modeled with Spark Machine Learning package, and predictions are made on the non labeled data and loaded into hdfs.](src/fitLR.py)
