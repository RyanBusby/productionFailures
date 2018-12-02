from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.classification import LogisticRegressionModel
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from datetime import datetime
from myLib import *
from munge import *

def run(root, smaster, train_fname, test_fname, ask):
    makeDirs()
    f = os.path.abspath(__file__)
    root, train_path, test_path, spark = \
    configure(root, smaster, train_fname, test_fname, f, ask)
    model, X_test = trainModel(spark, train_path)
    model_path = validate_save_model(model, X_test)
    make_save_preds(spark, root, test_path, model, model_path)

def trainModel(spark, train_path, persisted=True):
    train_df = load_data(spark, train_path, persisted=persisted)
    train_df.show()
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

def load_data(spark, path, persisted=True, test=False):
    if not persisted:
        if test:
            df = mungeNoLabel(path, spark)
            return df
        df = munge(path, spark)
        return df
    else:
        df = spark.read.csv(path, header=True, inferSchema=True)
        return df

def vectorize(df, test=False):
    numericCols = ['cntX', 'avgX', 'avg2X', '1plogavgX', 'O']
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


def validate_save_model(model, X_test):
    bce = BinaryClassificationEvaluator(labelCol='Response')
    train_preds = model.transform(X_test)
    score = bce.evaluate(train_preds)
    print('The Model got a %s of %s' % (bce.getMetricName(), score))
    dt = datetime.now().time()
    date_name = str(dt).replace(':', '_')
    model_dir = os.path.join(os.getcwd(), __file__, '..', 'models')
    model_dir = os.path.abspath(model_dir)
    model_path = os.path.join(model_dir, 'LR_%s' % date_name)
    model.save(model_path)
    return model_path

def make_save_preds(spark, root, test_path, model, model_path, persisted=True):
    if not model:
        model = LogisticRegressionModel.load(model_path)
    test_df = load_data(spark, test_path, persisted=persisted, test=True)
    test_df = vectorize(test_df, test=True)
    preds = model.transform(test_df)\
    .selectExpr('Id', 'CAST(prediction as INT) as Response')
    dt = datetime.now().time()
    date_name = str(dt).replace(':', '_')
    preds.write.csv('%s' % root % date_name + '_PREDS.csv', header=True)

if __name__ == '__main__':
    ask = False
    root = 'hdfs://ryans-macbook:9000/user/ryan/%s'
    smaster = 'spark://ryans-macbook:7077'
    train_fname = '12_51_48.275807_toyTrain.csv'
    test_fname = '12_51_49.871649_toyTest.csv'

    run(root, smaster, train_fname, test_fname, ask)
