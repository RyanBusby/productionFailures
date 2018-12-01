import pyspark as ps
from tqdm import tqdm
from pyspark.sql.functions import when
from datetime import datetime

def myMungeNoLabel(path, spark):
    df = spark.read.csv(path, header=True, inferSchema=True)

    print('counting observations'.upper())
    num_obs = df.rdd.map(lambda x:(x[0], counts(x, label=False)))\
    .toDF(['ii','Counts'])

    print('labeling outliers'.upper())
    df = multi_nameOuts(df, 1.5, label=False)(df)

    print('summing outliers'.upper())
    df = df.rdd.map(lambda x:(x[0],sum(x[1:]))).toDF(['Id','Outliers'])

    print('joining'.upper())
    df = df.join(num_obs, df.Id == num_obs.ii, 'left')\
    .select('Id','Outliers', 'Counts')

    cols = ['Id','Outliers', 'Obs', 'Outs']
    df = df.rdd.map(lambda x:(x[0],x[1],x[2][0],x[2][1])).toDF(cols)

    return df

def myMunge(path, spark):
    df = spark.read.csv(path, header=True, inferSchema=True)

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

def counts(x, label=True):
    obs, outs = 0, 0
    end = -1
    if not label:
        end = None
    for xx in x[1:end]:
        if xx:
            obs += 1
            if xx > .25 or xx < -.25:
                outs += 1
    return obs, outs

def nameOuts(df, col_name, iqrx):
    quants = df.approxQuantile([col_name],[.25,.75],.5)
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
        for col_name in tqdm(df.columns[1:end]):
            dataframe = dataframe.withColumn(col_name,\
                               nameOuts(df, col_name, iqrx))
        return dataframe
    return inner

def balance_classes(df):
    '''
    OVERSAMPLING SPECIFICALLY TO ADDRESS CLASS IMBALANCE OF BOSCHE DATA

    fraction argument in .sample() misbehaves
    if it didn't should be able to return without while loop
    '''
    c0 = df.filter(df.Response==0).count()
    c1 = df.filter(df.Response==1).count()
    biggerClass = max(c0, c1)

    x = (float(c0)-c1)/c1
    if x < .25: # yea but why though?
        return df
    else:
        while c1+df.filter(df.Response==1)\
        .sample(True, x, 13).count() < .9*c0:
            x += x/2 # research why fraction argument doesn't act right
    return df.union(df.filter(df.Response==1).sample(True,x,42))

if __name__ == '__main__':
    sparkContext = ps.SparkContext(master='spark://ryans-macbook:7077')
    spark = ps.sql.SparkSession(sparkContext)
    root = 'hdfs://ryans-macbook:9000/user/ryan/%s'
    file_name = 'test_numeric.csv'
    path = root % file_name

    X = myMunge(path, spark)

    dt = datetime.now().time()
    munged_file_name = str(dt).replace(':', '_') + '_' + file_name
    munged_path = root % munged_file_name

    print('saving data >>> '.upper() + munged_path)
    X.write.csv(munged_path, header=True)
