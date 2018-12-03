import os
from myLib import *
from fit_predict import *

def run(root, smaster, train_fname, test_fname, ask):
    makeDirs()
    make_toy_data('train')
    make_toy_data('test')
    f = os.path.abspath(__file__)
    root, train_path, test_path, spark = \
    configure(root, smaster, train_fname, test_fname, f, ask)
    model, X_test = trainModel(spark, train_path, persisted=False)
    model_path = validate_save_model(model, X_test)
    args = (spark, root, test_path, model, model_path, False)
    make_save_preds(*args)

if __name__ == '__main__':

    root = os.path.abspath(\
    os.path.join(os.getcwd(), __file__, '..', '..', 'data', '%s'))
    smaster = ''
    train_fname = 'toyTrain.csv'
    test_fname = 'toyTest.csv'
    ask = False

    run(root, smaster, train_fname, test_fname, ask)
