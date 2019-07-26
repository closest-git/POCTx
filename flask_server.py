# https://gist.github.com/kylehounslow/767fb72fde2ebdd010a0bf4242371594
# Send and receive images using Flask, Numpy and OpenCV
from flask import Flask,request,make_response
from werkzeug.serving import run_simple
import json
import threading
import uuid
import time
from datetime import datetime
import lightgbm as lgb
import os
import numpy as np
'''
1 @before_first_request
https://blog.zengrong.net/post/2632.html

2 reload Flask app in runtime
https://gist.github.com/joshuapowell/73655f55a6669ac2a580016dcc57d812
https://gist.github.com/nguyenkims/ff0c0c52b6a15ddd16832c562f2cae1d

3 processing requests 1 by 1
https://stackoverflow.com/questions/42325105/flask-processing-requests-1-by-1
'''

### http://127.0.0.1:5000/face/verify/'ss_card_file/4562de24aa800a944ce0650632e93c6a/1559524130035.jpg'
#http://127.0.0.1:5000/face/verify/{"url": "http://dig404.com"}
#app = Flask(__name__)   #__name__是固定写法，主要是方便flask框架去寻找资源 ，也方便flask插件出现错误时，去定位问题

version = '0.2'
to_reload = False
lgb_model = None

class FlaskConfig():
    def __init__(self, fix_seed=None):
        self.CUR_DIR = os.path.abspath(".")
        self.source = ""
        self.env = 'default'

def generate_request_id(original_id=''):
    new_id = uuid.uuid4()
    if original_id:
        new_id = "{},{}".format(original_id, new_id)
    return new_id

def initialize():
    global photor
    config = FlaskConfig(42)
    sTime = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S.%f")
    print("\n====================flask_server initialize@{} config={}".format(sTime,config.__dict__))

def LoadModel(model_file = './model/model_lgb_poct_1_.txt'):
    print(f'Loading model@\'{model_file}\' to predict...')
    model = lgb.Booster(model_file=model_file)
    print("\n====================lgb_model={}".format(model))
    return model

'''
格式
age  CA153     CA199  CYFRA211    CA724     CA125  NSE       PSA  SCCA  HE4      AFP.  CA211  sex
66.0    NaN  3.439456       NaN  0.48858       NaN  NaN       NaN   NaN  NaN       NaN    NaN    0
'''
def json2ndarray(json_data):
    i,features=0,['age','CA153', 'CA199', 'CYFRA211',  'CA724', 'CA125',  'NSE', 'PSA', 'SCCA','HE4','AFP.', 'CA211','sex']
    map={}
    arr=np.ndarray((1,len(features)))
    arr[:,:] = np.NAN
    for feat in features:
        map[feat]=i
        i=i+1
    for item in json_data:
        if item in features:
            no = map[item]
            arr[0,no]=float(json_data[item])
        else:
            print(f"item{item} is ...")
            pass
    return arr

def POCT_app():
    now = datetime.now()
    print("create app({}) now@{}".format(__name__,now))
    initialize()
    app = Flask(__name__)

    # to make sure of the new app instance

    @app.route('/')
    def hello_world():
        sTime = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S.%f")
        return "POCT Server on Flask!"

    @app.route('/reload')
    def reload():
        global to_reload
        to_reload = True
        return "reloaded"

    #some process on result
    def on_result(result):
        print(result)
        result = json.dumps(result, ensure_ascii=False)
        return result

    @app.route('/poct/detect/', methods=['POST'])
    def detect():
        global lgb_model
        optimal_threshold=0.34
        t0 = time.time()
        sTime = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S.%f")
        result = {'status': -1000, 'visit_0': sTime}
        content = request.get_json()
        test = json2ndarray(content)
        if lgb_model is None:
            lgb_model=LoadModel()
        if test is None or lgb_model is None:
            return result

        pred_test = lgb_model.predict(test)
        result['status'] = 0
        result['in_hospital'] = float(pred_test[0])<optimal_threshold
        result['OT'] = optimal_threshold
        result['pred'] = float(pred_test[0])
        print(">>>> detect={}......".format(result))
        #result['visit_1'] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S.%f")
        result['time'] = "{:.3g}".format(time.time() - t0)
        # print("<<<< result_64_len={} result={}".format(len(result['result_64']),result))
        result = json.dumps(result, ensure_ascii=False)
        return result

    return app

class AppReloader(object):
    def __init__(self, create_app):
        self.create_app = create_app
        self.app = create_app()

    def get_application(self):
        global to_reload
        if to_reload:
            self.app = self.create_app()
            to_reload = False

        return self.app

    def __call__(self, environ, start_response):
        #print("_____________AppReloader__call______________".format())
        global to_reload
        #to_reload = True
        app = self.get_application()
        return app(environ, start_response)

application = AppReloader(POCT_app)

def SomeTest():
    lgb_model = LoadModel()
    record = {'age': 66, 'sex': 0, 'status': -1000}
    record['CA199'] = 3.439456
    record['CA724'] = 0.488580
    test=json2ndarray(record)
    pred_test = lgb_model.predict(test)
    print(">>>> detect={}......".format(pred_test))

if __name__ == '__main__':
    SomeTest()

    if application is not None:
        run_simple('0.0.0.0', 5000, application,use_reloader=False, use_debugger=True, use_evalex=True, threaded=True)
        #run_simple('0.0.0.0', 5000, application,use_reloader=True, use_debugger=True, use_evalex=True, threaded=True)
    else:
        sTime = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S.%f")
        #app.run( debug=True,host='0.0.0.0', use_reloader=False, threaded=True)
        #app.run( host='0.0.0.0')
