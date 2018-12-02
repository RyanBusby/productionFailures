import pyspark as ps
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from datetime import datetime

def run(spark, root, train_fname, test_fname):
    train_path = root % train_fname
    test_path = root % test_fname
    model, X_test = trainModel(spark, train_path)
    validate_save_model(model, X_test)
    make_save_preds(model, test_path)

def load_data(path, persisted=True, test=False):
    if not persisted:
        if test:
            df = Munge(path, spark, labeled=False)
            return df
        df = Munge(path, spark)
        return df
    else:
        df = spark.read.csv(path, header=True, inferSchema=True)
        return df

def vectorize(df, test=False):
    numericCols = ['msrs', 'avg', 'avg2', 'ln', 'outs']
    assembler = VectorAssembler(inputCols=numericCols,\
                                outputCol='features',\
                                handleInvalid='keep')

    stages = [assembler]
    pipeline = Pipeline(stages=stages)
    pipelineModel = pipeline.fit(df)
    cols = ['features', 'Response']
    if test:
        cols = ['Id', 'features']
    df = pipelineModel.transform(df).select(cols)
    return df

def trainModel(spark, train_path):
    train_df = load_data(train_path)
    train_df = vectorize(train_df)

    X_train, X_test = train_df.randomSplit([.8, .2], 42)

    # utilize pyspark.ml.tuning here to gridsearch and tune the model
    lr = LogisticRegression(featuresCol='features',\
                            labelCol='Response',\
                            maxIter=2,\
                            regParam=.3,\
                            elasticNetParam=.8)

    lrModel = lr.fit(X_train)
    return lrModel, X_test

def validate_save_model(model, X_test):
    bce = BinaryClassificationEvaluator(labelCol='Response')
    train_preds = model.transform(X_test)
    score = bce.evaluate(train_preds)
    print('The Model got a %s of %s' % (bce.getMetricName(), score))
    dt = datetime.now().time()
    date_name = str(dt).replace(':', '_')
    model.save('../models/%s_LR' % date_name)

def make_save_preds(model, test_path):
    test_df = load_data(test_path)
    test_df = vectorize(test_df, test=True)
    preds = model.transform(test_df)\
    .selectExpr('Id', 'CAST(prediction as INT) as Response')
    dt = datetime.now().time()
    date_name = str(dt).replace(':', '_')
    preds.write.csv('%s' % root % date_name + '_PREDS.csv', header=True)

def launchSpark(local=False):
    if local:
        root = '../data/%s'
        sparkContext = ps.SparkContext('local[1]')
        spark = ps.sql.SparkSession(sparkContext)
        return spark, root
    root = 'hdfs://ryans-macbook:9000/user/ryan/%s'
    sparkContext = ps.SparkContext(\
    master = 'spark://ryans-macbook:7077')
    spark = ps.sql.SparkSession(sparkContext)
    return spark, root

def get_ans():
    islocal = None
    while not islocal:
        islocal =\
        raw_input('launch locally? 1-yes, 0-no: ')
        if islocal != '1' and islocal != '0':
            islocal = None
            print('enter 0 for no 1 for yes'.upper())
    return bool(islocal)

if __name__ == '__main__':
    isloc = get_ans()
    spark, root = launchSpark(local=isloc)

    train_fname = ''
    test_fname = ''

    run(spark, root, train_fname, test_fname)
