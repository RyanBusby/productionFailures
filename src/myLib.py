import os
import sys
import platform
import pyspark as ps
import csv
import random
from multiprocessing import cpu_count

def configure(root, smaster, train_file_name, test_file_name, f, ask):
    if ask:
        root, smaster, isLocal = get_ans(root, smaster)
        train_path = root % train_file_name
        test_path = root % test_file_name
        acceptConf(isLocal, root, smaster, f, train_path, test_path, ask)
    else:
        isLocal = False
        train_path = root % train_file_name
        test_path = root % test_file_name
        acceptConf(isLocal, root, smaster, f, train_path, test_path, ask)
    spark = launchSpark(smaster)
    return root, train_path, test_path, spark

def get_ans(root, smaster):
    isLocal = get_bool()
    if isLocal:
        cpus = cpu_count()/2
        if cpus == 0:
            cpus == 1
        workdir = os.getcwd()
        root = os.path.abspath(\
        os.path.join(workdir, __file__, '..', '..', 'data', '%s'))
        smaster = 'local[%s]' % cpus
        return root, smaster, isLocal
    return root, smaster, isLocal

def get_bool():
    q ='\nLaunch Spark locally and use local data store?\n1-yes, 0-no: '\
    .upper()
    islocal = False
    while not islocal:
        islocal = raw_input(q)
        if islocal == '1' or islocal == '0':
            return bool(int(islocal))
        else:
            islocal = False
            print('enter 0 for no 1 for yes'.upper())

def acceptConf(isLocal, root, smaster,confFile,train_path,test_path,ask):
    print('\nOS: '+platform.platform())
    print('Python Version: '+platform.python_version())
    print('PySpark Version: '+ps.__version__)
    os.system('java -version')
    os.system('scala -version')
    if isLocal:
        print('\npath to local data store: {}'.upper().format(root[:-2]))
        if train_path:
            print('training data location: {}'.upper()\
            .format(train_path))
        if test_path:
            print('test data location: {}'.upper().format(test_path))
    else:
        nn = '/'.join(root.split('/')[:-3])
        data_dir = '/'.join(root.split('/')[-3:-1])
        print('\nhdfs Namenode location: {}'.upper().format(nn))
        print('hdfs data directory: {}'.upper().format(data_dir))
        print('\nspark master location: {}'.upper().format(smaster))
        if train_path:
            print('\ntraining data location: {}'.upper()\
            .format(train_path))
        if test_path:
            print('test data location: {}'.upper().format(test_path))
    while ask:
        ans = raw_input('\n1-Continue, 0-Exit: ')
        if ans != '1' and ans != '0':
            print('enter 0 or 1'.upper())
        elif ans == '0':
            print('\x1b[6;37;41m'+\
            'enter proper configuration details in {}'\
            .upper().format(confFile)+'\x1b[0m')
            sys.exit()
        elif ans == '1':
            break

def launchSpark(smaster):
    sc = ps.SparkContext(master=smaster)
    sc.addFile('munge.py')
    sc.addFile('fit_predict.py')
    sc.addFile('munge_fit_predict.py')
    sc.addFile('myLib.py')
    spark = ps.sql.SparkSession(sc)
    return spark

def make_toy_data(type):
    if type == 'train':
        cols = ['Id','f1','f2','f3','Response']
        n = 'toyTrain.csv'
        save_path = os.path.join(os.getcwd(), '..', 'data', n)
        save_path = os.path.abspath(save_path)
    elif type == 'test':
        cols = ['Id','f1','f2','f3']
        n = 'toyTest.csv'
        save_path = os.path.join(os.getcwd(), '..', 'data', n)
        save_path = os.path.abspath(save_path)
    if os.path.exists(save_path):
        return
    writer = csv.DictWriter(open(save_path, 'w'),fieldnames=cols)
    writer.writerow(dict(zip(cols, cols)))
    for i in range(0, 100):
        idx = ('Id', i)
        f1  = ('f1', random.choice([None, random.gauss(0,1)]))
        f2  = ('f2', random.choice([None, random.gauss(0,1)]))
        f3  = ('f3', random.choice([None, random.gauss(0,1)]))
        if type == 'train':
            lbl = ('Response', random.randint(0,1))
            writer.writerow(dict([idx, f1, f2, f3, lbl]))
        elif type == 'test':
            writer.writerow(dict([idx, f1, f2, f3]))

def makeDirs():
    dataDir = os.path.join(os.getcwd(), __file__, '..', '..', 'data')
    dataDir = os.path.abspath(dataDir)
    if not os.path.isdir(dataDir):
        os.makedirs(dataDir)
    modsDir = os.path.join(os.getcwd(), __file__, '..', '..', 'models')
    modsDir = os.path.abspath(modsDir)
    if not os.path.isdir(modsDir):
        os.makedirs(modsDir)
