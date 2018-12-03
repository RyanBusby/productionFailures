import os
from myLib import *
from fit_predict import *

def run(root, smaster, train_fname, test_fname, ask):
    makeDirs()
    make_toy_data()
    f = os.path.abspath(__file__)
    root, train_path, test_path, spark = \
    configure(root, smaster, train_fname, test_fname, f, ask)
    model, X_test = trainModel(spark, train_path, persisted=False)
    model_path = validate_save_model(model, X_test)
    args = (spark, root, test_path, model, model_path, False)
    make_save_preds(*args)

if __name__ == '__main__':

    root = 'hdfs://ryans-macbook:9000/user/ryan/%s'
    smaster = 'spark://ryans-macbook:7077'
    train_fname = 'toyTrain.csv'
    test_fname = 'toyTest.csv'
    ask = True

    run(root, smaster, train_fname, test_fname, ask)
