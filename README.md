# __Binary Classification__

[<img src="img/logo.png" style="width: 5px;"/>](https://www.kaggle.com/c/bosch-production-line-performance/data) ↖  __the data__

#### The goal is to predict which parts will fail quality control
- The data are measurements related to part as they move through production lines.

- 81% of the dataset is empty - of the roughly one million rows, on average each row has 
- The classes are highly imbalanced: *0.58% failed.*

[The data are munged into a lower dimension utilizing Apache-Spark's DataFrame class. Each record is summarized by counts of outliers and the count of non null values. All of that is done with myMunge.py](myMunge.py)


The munged data is then modeled with Spark Machine Learning package.
