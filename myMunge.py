import os
import pyspark as ps
from tqdm import tqdm
from pyspark.sql.functions import when
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from datetime import datetime

sparkContext = ps.SparkContext(master='spark://ryans-macbook:7077')
spark = ps.sql.SparkSession(sparkContext)

def myMunge(f):
    df = spark.read.csv(f, header=True, inferSchema=True)
    df = drop_non_numeric_columns(df)
    print('counting observations'.upper())
    num_obs = df.rdd.map(lambda x:(x[0], counts(x), x[-1]))\
    .toDF(['ii','Counts', 'Response'])

    print('labeling outliers'.upper())
    df = multi_nameOuts(df,1.5)(df)

    print('summing outliers'.upper())
    df = df.rdd.map(lambda x:(x[0],sum(x[1:-1]))).toDF(['Id','Outliers'])

    print('joining'.upper())
    df = df.join(num_obs, df.Id == num_obs.ii, 'left')\
    .select('Outliers', 'Counts', 'Response')

    cols = ['Outliers', 'Obs', 'Outs', 'Response']
    df = df.rdd.map(lambda x:(x[0],x[1][0],x[1][1],x[2])).toDF(cols)

    print('balancing classes'.upper())
    df = balance_classes(df)
    return df

def counts(x):
    obs, outs = 0, 0
    for xx in x[1:-1]:
        if xx:
            obs += 1
            if xx > .25 or xx < -.25:
                outs += 1
    return obs, outs

def drop_non_numeric_columns(df):
    types = df.dtypes
    to_drop = [tup[0] for tup in types if tup[1] == 'string']
    df = df.drop(*to_drop)
    return df

def nameOuts(df, col_name, iqrx):
    quants = df.approxQuantile([col_name],[.25,.75],.5)
    q1, q3 = quants[0][0], quants[0][1]
    iqr = q3 - q1
    lb = q1 - iqrx * iqr
    ub = q3 + iqrx * iqr
    return when((df[col_name]<lb) | (df[col_name]>ub),1).otherwise(0)

def multi_nameOuts(df, iqrx):
    def inner(dataframe):
        for col_name in tqdm(df.columns[1:-1]):
            dataframe = dataframe.withColumn(col_name,\
                               nameOuts(df, col_name, iqrx))
        return dataframe
    return inner

def balance_classes(df):
    '''
    fraction argument in .sample() misbehaves
    if it didn't should be able to return without while loop
    '''
    c0 = df.filter(df.Response==0).count()
    c1 = df.filter(df.Response==1).count()
    x = (float(c0)-c1)/c1
    if x < .25:
        return df
    else:
        while c1+df.filter(df.Response==1)\
        .sample(True,x,42).count() < .9*c0:
            x += x/2
    return df.union(df.filter(df.Response==1).sample(True,x,42))

def vectorize(df):
    numericCols = ['Obs', 'Outs', 'Outliers']
    assembler = VectorAssembler(inputCols=numericCols,\
                                outputCol='features')

    stages = [assembler]
    pipeline = Pipeline(stages=stages)
    pipelineModel = pipeline.fit(df)
    df = pipelineModel.transform(df)
    return df

if __name__ == '__main__':
    root = 'hdfs://ryans-macbook:9000/user/ryan/%s'
    file_name = 'train_numeric.csv'
    path = root % file_name

    X = myMunge(path)

    dt = datetime.now().time()
    munged_file_name = str(dt).replace(':', '_') + '_' + file_name
    munged_path = root % munged_file_name

    print('saving data >>> '.upper() + munged_path)
    X.write.csv(munged_path, header=True)

    '''
    all this goes to a new module
    '''
    # vectorize
    # X_train, X_test = df.randomSplit([.8, .2], 42)
    # lr = LogisticRegression(featuresCol='features',\
    #                         labelCol='Response',\
    #                         maxIter=2,\
    #                         regParam=.3,\
    #                         elasticNetParam=.8)
    #
    # lrModel = lr.fit(X_train)
    #
    # print('Coefficients: ' + str(lrModel.coefficients))
    # print('Intercept: ' + str(lrModel.intercept))
