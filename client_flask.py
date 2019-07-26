import urllib.request
import json
import numpy as np

'''
返回json
    {'status': 0, 'visit_0': '2019-07-26 05:39:10.744215', 'in_hospital': True, 'OT': 0.34, 'pred': 0.11711524114040654, 'time': '0.00498'}
 
其中
    'status': 0 调用成功，其它为错误代码
    'in_hospital': True 则需要住院
'''
'''
一些样本数据
               id       date sex   age     CA153     CA199  CYFRA211     CA724     CA125       NSE       PSA      SCCA       HE4      AFP.  CA211
6225   3000487296 2019-06-01   M  66.0       NaN  3.439456       NaN  0.488580       NaN       NaN       NaN       NaN       NaN       NaN    NaN
13098      407225 2019-06-01   M  58.0       NaN       NaN       NaN       NaN       NaN       NaN -1.666008       NaN       NaN       NaN    NaN
12437  1000071937 2019-06-01   F  47.0       NaN       NaN       NaN       NaN       NaN       NaN       NaN       NaN       NaN  0.837248    NaN
12435  3003214322 2019-06-01   F   9.0       NaN       NaN       NaN       NaN       NaN       NaN       NaN       NaN       NaN  0.587787    NaN
6239       756058 2019-06-01   F  64.0       NaN  2.445819       NaN -0.128970  3.178054       NaN       NaN       NaN       NaN       NaN    NaN
6243       762332 2019-06-01   F  60.0       NaN  2.823757  1.169381  0.371564  1.410987  1.231101       NaN  0.053541       NaN       NaN    NaN
6244       762357 2019-06-01   M  54.0       NaN  2.871868  0.760806  2.225704  1.410987  0.991027       NaN  0.613563       NaN       NaN    NaN
12433         NaN 2019-06-01   F  52.0       NaN       NaN       NaN       NaN       NaN       NaN       NaN       NaN       NaN  0.947789    NaN
12626  3004802081 2019-06-01   F  36.0       NaN       NaN       NaN       NaN  3.613617       NaN       NaN       NaN       NaN       NaN    NaN
6248       747454 2019-06-01   M  68.0       NaN  3.359681       NaN  0.916291  2.028148       NaN       NaN       NaN       NaN       NaN    NaN
6254       660122 2019-06-01   M  65.0       NaN  4.118549       NaN  1.413423       NaN       NaN       NaN       NaN       NaN       NaN    NaN
6376       762347 2019-06-01   M  64.0       NaN  1.927164       NaN       NaN  2.282382       NaN       NaN       NaN       NaN       NaN    NaN
6374       762326 2019-06-01   M  46.0       NaN  3.875359       NaN       NaN       NaN       NaN       NaN       NaN       NaN       NaN    NaN
6759       762179 2019-06-01   M  32.0       NaN       NaN  1.108563       NaN       NaN       NaN       NaN  0.574927       NaN       NaN    NaN
6373       512387 2019-06-01   F  75.0       NaN  3.282038       NaN       NaN  1.722767       NaN       NaN       NaN       NaN       NaN    NaN
6261       762345 2019-06-01   M  64.0       NaN  2.805782  1.423108  1.329724  2.714695  0.897719       NaN  0.403463       NaN       NaN    NaN
522        760021 2019-06-01   F  30.0  1.667707  3.496204  0.559616 -0.143870  2.708050  0.630740       NaN  0.229523  3.769999       NaN    NaN
12430  1000087243 2019-06-01   M  77.0       NaN       NaN       NaN       NaN       NaN       NaN       NaN       NaN       NaN  1.854734    NaN
13091      761830 2019-06-01   M  72.0       NaN       NaN       NaN       NaN       NaN       NaN  0.215111       NaN       NaN       NaN    NaN
6230   6010832478 2019-06-01   F  50.0       NaN  2.469793       NaN       NaN  2.230014       NaN       NaN       NaN  3.492865       NaN    NaN
'''
if __name__ == '__main__':
    #record = {'age': 66, 'sex': 0, 'CA199': 3.439456, 'CA724': 0.488580}
    #record = {'age':58.0,'sex':0,   'PSA':-1.666008}
    record = {'age':47, 'sex': 1, 'AFP.': 0.837248}
    #record = {'age': 9, 'sex': 1, 'AFP.': 0.587787}
    #record = {'age': 50, 'sex': 1, 'CA199': 2.469793,'CA125': 2.230014,'HE4': 3.492865}

    print(f"{record}")
    server_url = "http://localhost:5000/poct/detect/"

    nTest=1
    for i in range(nTest):
        req = urllib.request.Request(server_url)
        req.add_header('Content-Type', 'application/json; charset=utf-8')
        jsondata = json.dumps(record)
        jsondataasbytes = jsondata.encode('utf-8')  # needs to be bytes
        req.add_header('Content-Length', len(jsondataasbytes))
        #print(jsondataasbytes)
        # https://stackoverflow.com/questions/37616460/how-to-print-out-http-response-header-in-python
        with urllib.request.urlopen(req, jsondataasbytes) as response:
            resp_body = response.read().decode('utf-8')
            #print("{} resp_body={}".format(i, result))
            result = json.loads(resp_body)
            if False and (result['status'] == 0):
                pass
            print("{} result={}".format(i, result))
        #break
