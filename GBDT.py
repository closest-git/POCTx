import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold
from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn.metrics import roc_auc_score, roc_curve,auc
import time
import numpy as np
isMORT=False
import matplotlib.pyplot as plt

def ROC_plot(features,X_,y_, pred_,title):
    fpr_, tpr_, _ = roc_curve(y_, pred_)
    auc_ = auc(fpr_, tpr_)
    title = "{} auc=".format(title)
    print("{} auc={} ".format(title, auc_))
    plt.plot(fpr_, tpr_, label="{}:{:.4g}".format(title, auc_))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('SMPLEs={} Features={}'.format(X_.shape[0],len(features)))
    plt.legend(loc='best')
    plt.show()

def runLgb(X, y, test=None, num_rounds=10000, max_depth=-1, eta=0.01, subsample=0.8,
           colsample=0.8, min_child_weight=1, early_stopping_rounds=50, seeds_val=2017):
    features = list(X.columns)
    print("X={} y={}".format(X.shape,y.shape))
    param = {'task': 'train',
             'min_data_in_leaf': 32,
             'boosting_type': 'gbdt',
             'objective': 'binary',
             'learning_rate': eta,
             # 'metric': {'multi_logloss'},
             'metric': 'auc',
             'max_depth': max_depth,
             # 'min_child_weight':min_child_weight,
             'bagging_fraction': subsample,
             'feature_fraction': colsample,
             'bagging_seed': seeds_val,
             'num_iterations': num_rounds,
             'num_leaves': 32,
             'lambda_l1': 1.0,
             'verbose': 0,
             'nthread': -1}
    n_fold = 5
    folds = KFold(n_splits=n_fold, shuffle=True, random_state=11)
    y_pred=np.zeros(y.shape[0])
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
        y_pred[valid_index] = pred_val

        if test is not None:
            pred_test = model.predict(test, num_iteration=model.best_iteration)
        else:
            pred_test = None

    ROC_plot(features,X, y, y_pred, "")
    cv_score = 0    #roc_auc_score(target, oof)
    print("CV score: {:<8.5f}".format(cv_score))
    return cv_score