import pandas as pd
import numpy as np

def process_sngl_data(folder, file):
	"""
	这里单品种的dataframe的格式example如下：
    =============================
       date      open     high    low    close   volume
       
    2017-07-28   256.3   263.2   249.3   262.4   121234
    2017-07-29   259.5   260.1   253.5   258.2   543542
    =============================

	"""
	asset = pd.read_csv(folder+file, engine="python")
	dt_index = pd.to_datetime(asset['date'])
	asset.set_index(dt_index, inplace = True)
	return asset



