# -*- coding: utf-8 -*-
"""

@author: ainnmzln
"""

import re
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler


class ExploratoryDataAnalysis():
    
    def __init__(self):
        pass
    
    def remove_char(self,data):
        
        for index,text in enumerate(data):
            data[index]=re.sub("[^0-9]", "",text)
            
        return data
    
    def change_numeric(self,data):
        
        data_numeric=pd.to_numeric(data,errors='coerce')

        return data_numeric
    
    def imputer(self,data):
        imputer=SimpleImputer()
        data_imputed=imputer.fit_transform(np.expand_dims(data,axis=1))

        return data_imputed
    
    def scaler(self,data):
        
        scaler=MinMaxScaler()
        data_scaled=scaler.fit_transform(data)
        
        return data_scaled






    

if __name__=='__main__':
    pass


