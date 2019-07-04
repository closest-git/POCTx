import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


class TumorMarkers(object):
    def __init__(self,config,xls_path):
        #self.config = config
        self.marker_dict = {}

        if xls_path is not None:
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