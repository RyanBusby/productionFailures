import pyspark as ps
from datetime import datetime
from math import log1p
from pyspark.sql.functions import when

def munge(path, spark):
    df = spark.read.csv(path, header=True, inferSchema=True)
    print('aggregating row'.upper())
    df = df.rdd.map(rowaggs).toDF()

    print('labeling outliers'.upper())
    df = multi_nameOuts(df, 1.25)(df)
    df= df.rdd.map(lambda x:(x[0],x[1],x[2],x[3],sum(x[5:]),x[4]))\
    .toDF(['msrs', 'avg', 'avg2', 'ln', 'outs', 'Response'])

    df = balance_classes(df)

    return df

def rowaggs(x):
    r = filter(None, x[1:-1])
    avrg = 0
    if len(r) != 0:
        avrg = sum(r)/len(r)
    return len(r), avrg, avrg**2, log1p(avrg), x[-1]

def mungeNoLabel(path, spark):
    df = spark.read.csv(path, header=True, inferSchema=True)
    print('aggregating row'.upper())
    df = df.rdd.map(rowaggsNoLabel).toDF()

    print('labeling outliers'.upper())
    df = multi_nameOuts(df, 1.25, label=False)(df)
    df= df.rdd.map(lambda x:(x[0],x[1],x[2],x[3],x[4],sum(x[5:8])))\
    .toDF(['Id', 'msrs', 'avg', 'avg2', 'ln', 'outs'])

    return df

def rowaggsNoLabel(x):
    r = filter(None, x[1:])
    avrg = 0
    if len(r) != 0:
        avrg = sum(r)/len(r)
    return x[0], len(r), avrg, avrg**2, log1p(avrg)

def nameOuts(df, col_name, iqrx):
    quants = df.approxQuantile([col_name],[.25,.75],0)
    q1, q3 = quants[0][0], quants[0][1]
    iqr = q3 - q1
    lb = q1 - iqrx * iqr
    ub = q3 + iqrx * iqr
    return when((df[col_name]<lb) | (df[col_name]>ub),1).otherwise(0)

def multi_nameOuts(df, iqrx, label=True):
    # USE approxQuantile() TO CALCULATE THE IQR PER COLUMN AND LABEL OUTS
    end = -1
    if not label:
        end = None
    def inner(dataframe):
        for col_name in df.columns[1:end]:
            dataframe = dataframe\
            .withColumn('%s_ISOUT'%col_name,nameOuts(df, col_name, iqrx))
        return dataframe
    return inner

def balance_classes(df):
    # OVERSAMPLING SPECIFICALLY TO ADDRESS CLASS IMBALANCE OF BOSCHE DATA
    '''
    fraction argument in .sample() misbehaves
    if it didn't should be able to return without while loop
    '''
    c0 = df.filter(df.Response==0).count()
    c1 = df.filter(df.Response==1).count()
    diff = float(abs(c0 - c1))
    lrgrClss = max(c0, c1)
    smlrClss = min(c0, c1)
    if smlrClss == 0:
        smlrClss = 1
    x = diff / smlrClss
    f_label = 0
    if c0 > c1:
        f_label = 1
    if x < .25:
        return df
    else:
        while smlrClss+df.filter(df.Response==f_label)\
        .sample(True, x, 42).count() < .9*lrgrClss:
            x += x/2
    return df.union(df.filter(df.Response==f_label).sample(True,x,42))

def save_munged(X, file_name):
    dt = datetime.now().time()
    munged_file_name = str(dt).replace(':', '_') + '_' + file_name
    munged_path = root % munged_file_name
    print('saving data >>> '.upper() + munged_path)
    X.write.csv(munged_path, header=True)

if __name__ == '__main__':
    sparkContext = ps.SparkContext(master='spark://ryans-macbook:7077')
    spark = ps.sql.SparkSession(sparkContext)

    root = 'hdfs://ryans-macbook:9000/user/ryan/%s'

    # train_file_name = 'train_numeric.csv'
    # train_path = root % train_file_name
    # X = munge(train_path, spark)
    # X.show()
    # save_munged(X, train_file_name)
    #
    # test_file_name = 'test_numeric.csv'
    # test_path = root % test_file_name
    # X = mungeNoLabel(test_path, spark)
    # X.show()
    # save_munged(X, test_file_name)

    train_file_name = 'toyTrain.csv'
    train_path = root % train_file_name
    X = munge(train_path, spark)
    X.show()
    save_munged(X, train_file_name)

    test_file_name = 'toyTest.csv'
    test_path = root % test_file_name
    X = mungeNoLabel(test_path, spark)
    X.show()
    save_munged(X, test_file_name)
