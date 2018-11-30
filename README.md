# __Binary Classification__

[<img src="img/logo.png" style="width: 5px;"/>](https://www.kaggle.com/c/bosch-production-line-performance/data) ↖  __the data__

#### The goal is to predict which parts will fail quality control to reduce manufacturing failures.

- The data represents measurements of parts as they move through production lines.

- They are very sparse: *81% empty.* The classes are highly imbalanced: *0.58% failed.*

The data are munged into a lower dimensions utilizing Apache-Spark's DataFrame class. Each row in summarized by counts of outliers and the count of non null values.
