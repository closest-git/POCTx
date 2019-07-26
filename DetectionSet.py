import argparse
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
myfont = FontProperties(fname=r'C:\Windows\Fonts\simhei.ttf', size=14)
import numpy as np
import os
from datetime import datetime
import time
import pickle
from TumorMarkers import *
from GBDT import *

class DetectionSet:
    def __init__(self):
        pass

class TumorSamples(DetectionSet):
    def __init__(self, config, xls_path):
        self.config=config
        source = self.config.data_source
        tumor_markers = config.tumor_markers        #可能会被修改
        self.pkl_path="./data/TumorSamples_{}_.pickle".format(source)
        if os.path.isfile(self.pkl_path):
            pass
        else:
            self.scan_xls_path(config, xls_path)

        print("\n======load pickle file from {} ...".format(self.pkl_path))
        with open(self.pkl_path, "rb") as fp:  # Pickling
            [self.df,self.tumor_markers] = pickle.load(fp)
        self.testing_index = list(self.tumor_markers.marker_dict.keys())
        print("===============df={} tumor_markers={}\n{}".format(self.df.shape,self.testing_index, self.df.head(20)))
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
            df_EDA = self.df[self.df['id'].notnull()].reset_index(drop=True)
            df_EDA.fillna(0.0, inplace=True)
            df_EDA['in_hospital'] = df_EDA.apply(lambda x: len(x['id']) <= 6, axis=1)
            print(df_EDA['in_hospital'].value_counts())
            for col in df_EDA.columns:
                if col in ['id','date','sex','in_hospital']:
                    continue

                df_EDA[col] = df_EDA[col].astype(np.float)
                sns.FacetGrid(df_EDA, hue="in_hospital", height=5).map(sns.distplot, col).add_legend();
                #sns.distplot(df_EDA[col],kde=False )
                path = "./result/hospital_{}.jpg".format(col.replace('/' , '_'))
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

    def OnAge(self,age):
        if age == "未知":
            return None
        if '天' in age or '月' in age:
            age = 0.0
        else:
            age = age.replace("岁", "")
            age = float(age)
        return age

    def OnTect(self,tect_str):
        source = self.config.data_source
        markers = self.tumor_markers.marker_dict
        k_id, k_val = None,None
        tect_tokens = tect_str.strip( ).split(' ')
        tect={}
        if len(tect_tokens)!=2:
            return k_id, k_val
        try:
            k_id,k_val=tect_tokens[0],(float)(tect_tokens[1])
            assert(k_val>0)
            k_val = np.log(k_val)
        except:
            return k_id, k_val
        isUpdate = source == "LSW"
        if k_id not in markers:
            if isUpdate:     #允许添加
                markers[k_id]={"样本量":1,	"平均值":0.,	"中值":0.,	"标准差":0.,	"变异系数":0.,
                               "方差":0.,	"最小值":k_val,	"最大值":k_val}

            else:
                pass
        else:
            if isUpdate:
                markers[k_id]["样本量"] = (int)(markers[k_id]["样本量"]+1)
                markers[k_id]["最小值"] = min((markers[k_id]["最小值"]),k_val)
                markers[k_id]["最大值"] = max((markers[k_id]["最大值"]),k_val)
        return k_id,k_val

    def markers_stat(self,df):
        markers = self.tumor_markers.marker_dict
        for key,info in markers.items():
            feat = df[df[key].notnull()][key]
            markers[key]['样本量'] = len(feat)     #有重复啊
            markers[key]['平均值'] = feat.mean()
            markers[key]['中值'] = feat.median()
            markers[key]['方差'] = feat.var()
            markers[key]['标准差'] = feat.std()
            markers[key]['变异系数'] = feat.skew()

        df_markers = pd.DataFrame.from_dict(markers, orient='index')
        df_markers.to_excel("./data/tumor_marks_{}_.xls".format(len(markers)))
        print(df_markers)

    def scan_xls_path(self,config, xls_path):
        source = config.data_source
        self.tumor_markers = config.tumor_markers
        if source == "LSW":
            date_format = '%Y/%m/%d'
        else:
            date_format = '%Y-%m-%d'

        nz,extensions = 0,['.xls']
        files = os.listdir(xls_path)
        samples={}
        t0=time.time()
        for file in files:
            name, extension = os.path.splitext(file)
            if extension not in extensions:            continue
            df = pd.read_excel("{}{}".format(xls_path,file), usecols ='B:D,F:G,M',dtype=str)
            nRow, nSample = 0, 0
            for index, row in df.iterrows():
                nRow=nRow+1
                id,tect_str,sex,age,time_str=row['病人ID'],row['ID类型'],row['性别'],row['年龄'],row['报告时间']
                age = self.OnAge(age)
                if age == None:     continue
                if sex == "女":
                    sex = "F"
                elif sex == "男":
                    sex = "M"
                else:
                    continue

                k_id,k_val = self.OnTect(tect_str)
                if k_id is None:        continue
                try:
                    date = datetime.strptime(time_str, date_format)
                except:
                    #print("failed to parse date@{}".format(time_str))
                    if time_str is not np.nan:
                        continue
                if (id, date) not in samples:
                    samples[(id, date)] = {'sex':sex,'age':age}
                samples[(id, date)][k_id]=k_val
                nz = nz + 1;    nSample=nSample+1
                if nz%1000==0:
                    print("{}\tnItem={} time={:.3g}".format(nz,len(samples),time.time()-t0))
                    #break
            print("====== nRow={} nSample={} @\"{}\"......".format(nRow,nSample,file))
            #break
        df = pd.DataFrame.from_dict(samples, orient='index').reset_index()
        if source == "LSW":
            self.markers_stat(df)
        df = df.rename(columns={ df.columns[0]: "id",df.columns[1]: "date" }).sort_values('date')
        print ("===============df={}\n{}".format(df.shape,df.head()))
        with open(self.pkl_path, "wb") as fp:  # Pickling
            pickle.dump([df,self.tumor_markers], fp)
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

    def HospitalOnID(self):
        df_valid = self.df[self.df['id'].notnull()].reset_index(drop=True)
        print("df_valid=[{}]".format(df_valid.shape))
        df_valid['in_hospital'] = df_valid.apply(lambda x: len(x['id'])<=6, axis=1)
        self.y = df_valid['in_hospital'].astype(np.int)

        cols = df_valid.columns
        x_cols = [e for e in cols if e not in ('id', 'date', 'sex', 'in_hospital')]
        #sugestion by Dr.Lin
        #x_cols = [e for e in cols if e in ('age', 'CA199','CYFRA211', 'AFP.')]
        #x_cols = [e for e in cols if e in ('age', 'CA199', 'CA125', 'CYFRA211', 'PSA', 'NSE')]


        self.X = df_valid[x_cols].astype(np.float)
        #self.X['sex'] = df_valid['sex'].astype('category')
        self.X['sex'] = df_valid.apply(lambda x: 0 if x['sex']=='M' else 1, axis=1)
        print("HospitalOnID X=[{}],{}\ny={}".format(self.X.shape,self.X.head(),self.y.shape ))
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

def InHospital(samples):
    samples.HospitalOnID( )
    score = runLgb(samples.X, samples.y)

def main():
    global args
    args = arg_parser()
    args.data_source = "LSW"
    markers = TumorMarkers(args, None if args.data_source == "LSW" else "./data/肿瘤标志物.xls")
    args.tumor_markers = markers
    #samples = TumorSamples(args,"./data/蛋白检测样本/")
    samples = TumorSamples(args, "./data/LSW/")
    #samples.EDA()
    InHospital(samples)

    os._exit(-1)


if __name__ == '__main__':
    main()