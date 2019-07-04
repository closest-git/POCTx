import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
myfont = FontProperties(fname=r'C:\Windows\Fonts\simhei.ttf', size=14)
from sklearn.metrics import mean_absolute_error,mean_squared_error
import numpy as np
import os
from datetime import datetime
import time
import pickle
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold

isMORT=False

class DetectionSet:
    def __init__(self):
        pass

class TumorSamples(DetectionSet):
    def __init__(self, config, xls_path):
        self.version="v0.2"
        self.config=config
        markers = config.tumor_markers.marker_dict
        self.testing_index = list(markers.keys())
        self.pkl_path="./data/TumorSamples_{}_.pickle".format(self.version)

        if os.path.isfile(self.pkl_path):
            pass
        else:
            self.scan_xls_path(config, xls_path)
        print("\n======load pickle file from {} ...".format(self.pkl_path))
        with open(self.pkl_path, "rb") as fp:  # Pickling
            [self.df] = pickle.load(fp)
        print("===============df={}\n{}".format(self.df.shape, self.df.head()))
        self.testing_index = list(set(self.testing_index) & set(self.df.columns))
        print("===============testing_index={}".format(self.testing_index))

    def EDA_pair(self):
        self.df.fillna(0.0, inplace=True)
        sns.set_style("whitegrid");
        sns.pairplot(self.df, hue="age", vars=['age', 'PGII'], size=4);
        plt.savefig("./result/pair_[{}]_.jpg".format(self.df.shape))
        plt.show()
        print("")

    def EDA(self):
        if True:
            self.df.fillna(0.0, inplace=True)
            for col in self.df.columns:
                if col in ['id','date','sex']:
                    continue
                self.df[col] = self.df[col].astype(np.float)
                sns.FacetGrid(self.df, hue="sex", height=5).map(sns.distplot, col).add_legend();

                path = "./result/dist_{}.jpg".format(col.replace('/' , '_'))
                plt.savefig(path)
                plt.show();
            return

        nz=0

        for index in self.testing_index:
            if index not in self.df.columns:
                continue

            target,x_var,y_var = 'sex','age',index
            sns.set_style("whitegrid");
            g = sns.FacetGrid(self.df, hue=target, height=5) .map(plt.scatter, x_var,y_var, s=2 ) ;
            for ax in g.axes.flat:
                labels = ax.get_xticklabels()  # get x labels
                for i, l in enumerate(labels):
                    if (i % 5 != 0): labels[i] = ''  # skip even labels
                ax.set_xticklabels(labels)  # set new labels
            path="./result/{}.jpg".format(y_var.replace('/' , '_') )       #"./result/{}_'{}'_[{}]_.jpg".format(x_var,nz,self.df.shape)
            plt.savefig(path)
            plt.show()
            nz=nz+1

    def scan_xls_path(self,config, xls_path):
        markers = config.tumor_markers.marker_dict
        nz,extensions = 0,['.xls']
        files = os.listdir(xls_path)
        samples={}
        t0=time.time()
        for file in files:
            name, extension = os.path.splitext(file)
            if extension not in extensions:            continue
            df = pd.read_excel("{}{}".format(xls_path,file), usecols ='B:D,F:G,M',dtype=str)
            for index, row in df.iterrows():
                id,tect_str,sex,age,time_str=row['病人ID'],row['ID类型'],row['性别'],row['年龄'],row['报告时间']
                if age=="未知":
                    continue
                if '天' in age or '月' in age:
                    age=0.0
                else:
                    age = age.replace("岁", "")
                    age = float(age)
                if sex == "女":
                    sex = "F"
                elif sex == "男":
                    sex = "M"
                else:
                    continue
                tect_tokens = tect_str.strip( ).split(' ')
                tect={}
                if len(tect_tokens)==2:
                    try:
                        k_id,k_val=tect_tokens[0],(float)(tect_tokens[1])
                    except:
                        continue
                    if k_id not in markers:
                        continue
                else:
                    continue
                try:
                    if self.version=="v0.2":
                        date = datetime.strptime(time_str, '%Y/%m/%d')
                    else:
                        date = datetime.strptime(time_str, '%Y-%m-%d')
                except:
                    print("failed to parse date@{}".format(time_str))
                    if time_str is not np.nan:
                        continue
                if (id, date) not in samples:
                    samples[(id, date)] = {'sex':sex,'age':age}
                samples[(id, date)][k_id]=k_val
                nz = nz + 1
                if nz%1000==0:
                    print("{}\tnItem={} time={:.3g}".format(nz,len(samples),time.time()-t0))
                    #break
        df = pd.DataFrame.from_dict(samples, orient='index').reset_index()
        df = df.rename(columns={ df.columns[0]: "id",df.columns[1]: "date" }).sort_values('date')
        print ("===============df={}\n{}".format(df.shape,df.head()))
        with open(self.pkl_path, "wb") as fp:  # Pickling
            pickle.dump([df], fp)
        pass

    def Split(self,target="age"):
        df_valid = self.df[self.df[target].notnull()]
        print("df_valid[{}]=[{}]".format(target,df_valid.shape))

        cols = df_valid.columns
        x_cols = [e for e in cols if e not in (target, 'id','date','sex')]
        self.y = df_valid[target].astype(np.float)
        self.X = df_valid[x_cols].astype(np.float)
        self.X['sex'] = df_valid['sex'].astype('category')

        pass


class TumorMarkers(DetectionSet):
    def __init__(self,config,xls_path):

        self.df = pd.read_excel(xls_path)
        self.df.set_index('项目代码',inplace=True)
        nMark=self.df.shape[0]-1
        self.marker_dict = self.df[0:nMark].to_dict(orient='index')
        if False:
            import pprint
            pprint.pprint(markers)
        else:
            print("{" + "\n".join("{}: {}".format(k, v) for k, v in self.marker_dict.items()) + "}")
        return

def arg_parser():
    parser = argparse.ArgumentParser(description='POCT by Xiamen University')
    return parser
    parser.add_argument('data', metavar='DIR', help='path to dataset')
    parser.add_argument('--arch', '-a', metavar='ARCH', default=model_name,
                        choices=model_names,
                        help='model architecture: ' +
                             ' | '.join(model_names) +
                             ' (default: resnet34)')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=1000, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=32, type=int,
                        metavar='N', help='mini-batch size (default: 32)')
    parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--print-freq', '-p', default=300, type=int,
                        metavar='N', help='print frequency (default: 100)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='gloo', type=str,
                        help='distributed backend')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    print(parser)
    return parser

def runLgb(X, y, test=None, num_rounds=1000, max_depth=1, eta=0.5, subsample=0.8,
           colsample=0.8, min_child_weight=1, early_stopping_rounds=50, seeds_val=2017):
    print("X={} y={}".format(X.shape,y.shape))
    param = {'task': 'train',
             'min_data_in_leaf': 32,
             'boosting_type': 'gbdt',
             'objective': 'regression',
             'learning_rate': eta,
             # 'metric': {'multi_logloss'},
             'metric': 'mae',
             'max_depth': max_depth,
             # 'min_child_weight':min_child_weight,
             'bagging_fraction': subsample,
             'feature_fraction': colsample,
             'bagging_seed': seeds_val,
             'num_iterations': num_rounds,
             'num_leaves': 2,
             'min_data_in_leaf': 60,
             'lambda_l1': 1.0,
             'verbose': 0,
             'nthread': -1}
    n_fold = 5
    folds = KFold(n_splits=n_fold, shuffle=True, random_state=11)
    for fold_n, (train_index, valid_index) in enumerate(folds.split(X)):
        t0 = time.time()

        if type(X) == np.ndarray:
            X_train, X_valid = X[train_index], X[valid_index]
            y_train, y_valid = y[train_index], y[valid_index]
        else:
            X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
            y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

        if isMORT:
            # model = LiteMORT(params).fit(X_train, y_train, eval_set=[(X_valid, y_valid)])
            # model = LiteMORT(param).fit_1(Xtr, ytr, eval_set=[(Xvl, yvl)])
            pass
        else:
            lgtrain = lgb.Dataset(X_train, y_train)
            lgval = lgb.Dataset(X_valid, y_valid)
            model = lgb.train(param, lgtrain, num_rounds, valid_sets=lgval,
                              early_stopping_rounds=early_stopping_rounds, verbose_eval=100)
        pred_val = model.predict(X_valid, num_iteration=model.best_iteration)
        fold_score = mean_absolute_error(pred_val, y_valid)
        if test is not None:
            pred_test = model.predict(test, num_iteration=model.best_iteration)
        else:
            pred_test = None
        break
    return fold_score

def main():
    global args
    args = arg_parser()
    markers = TumorMarkers(args, "./data/肿瘤标志物.xls")
    args.tumor_markers = markers
    #samples = TumorSamples(args,"./data/蛋白检测样本/")
    samples = TumorSamples(args, "./data/LSW/")
    samples.EDA()
    results={}
    for index in samples.testing_index:
        samples.Split(target=index)
        score = runLgb(samples.X, samples.y)
        results[index]=score
    print(results)
    os._exit(-1)


if __name__ == '__main__':
    main()