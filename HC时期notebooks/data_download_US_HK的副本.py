import pandas as pd, numpy as np
from datetime import datetime
import requests
from bs4 import BeautifulSoup
import statsmodels.api as sm
import yfinance as yf
import matplotlib
import time, urllib
import glob
import re
import os



class US_TICKERS_POOL():

#---------------------------------0305---------------------------------
	def us_consumer_discretionary_stocks():#非必需消费品
		tickers = list(set(['HD','PDD','NKE','MCD','JD','SBUX','LOW','LULU','LVS',
			'RACE','PTON','CHWY','CMG','EBAY','AZO','BBY','CZR']))
		return tickers

	def us_consumer_staples_stocks():#必需消费品
		tickers = list(set(['WMT','PG','KO','PEP','COST','PM','EL','TGT','CL','STZ',
			'KHC','TSN','SAM']))
		return tickers	

	def us_industrial_stocks():
		tickers = list(set(['HON','UPS','UNP','BA','RTX','CAT','MMM','LMT','GE']))
		return tickers	

	def us_real_estate_stocks():
		tickers = list(set(['AMT','PLD','CCI','EQIX','DLR','PSA','SBAC']))
		return tickers	

	def us_tele_entertainment_stocks():
		tickers = list(set(['GOOGL','FB','DIS','VZ','NFLX','CMCSA','T','TMUS',
			'CHTR','ATVI','SPOT','BILI','TME']))
		return tickers	


#--------------------------------0304--------------------------------
	def us_materials_stocks():
		tickers = list(set(['LIN','SHW','APD','ECL','DD','SCCO','NEM']))
		return tickers		

	def us_utilities_stocks():
		tickers = list(set(['NEE','DUK','SO','D','AEP','EXC','SRE','XEL','ES','WES']))
		return tickers			

	def us_fin_stocks():
		tickers = list(set(['JPM','BAC','C','WFC','MS','BLK','GS','AXP','BX','CB','MET']))
		return tickers	

	def us_healthcare_stocks():
		tickers = list(set(['JNJ','UNH','PFE','MRK','ABT','TMO','ABBV','DHR','LLY','MDT','BMY',
			'AMGN','ISRG','CVS']))
		return tickers			
#-------------------------------pre 0304-----------------------------
	def us_tech_stocks():
		tickers = list(set(['V','MA','NFLX','CSCO','ADBE','MSFT','CRM','TSLA','GOOG','FB','GOOGL','TSM',
				'AMZN','AAPL','PYPL','INTC','NVDA','QCOM','MU','BABA','BIDU']))
		return tickers

	def us_blockchain_stocks():
		tickers = list(set(['EBON','CAN','AMD','FB','NVDA','OSTK',
					'RIOT','GBTC','MARA']))
		return tickers


	def us_saas_stocks():
		tickers = list(set(['HUBS','CDAY','ORCL','WDAY','COUP','INTU','SMAR','OKTA','ADP','PAYC','WORK',
		'ADBE','MSFT','SPLK','ZM','RNG','CRM','SAP','GDDY','DBX','VEEV','NOW','TEAM',
		'ZEN','ADSK','TTD','DDOG','TWLO','SHOP','DOCU','SQ','CRWD']))
		return tickers

	def us_biotech_stocks():
		tickers = list(set(['INO','MRNA','VIR','GILD','AZN','PFE','JNJ','BNTX','ABT',
		'VCNX','NVAX']))
		return tickers

	def us_new_energy_stocks():
		tickers = list(set(['TSLA','NIO','LI','XPEV','SQM','ALB','LTHM','FMC','LAC','QS','VLDR','LAZR',
					'SBE','BLNK','XL','ENPH','PLUG','RUN','FLS','NOVA','FCEL']))
		return tickers

	def other_us_stocks():
		tickers = list(set(['OPEN']))
		return tickers

	def global_etfs():
		# http://www.360doc.com/content/20/1207/15/72758660_949970677.shtml
		tickers = list(set(["SPY",
					"QQQ",
					"VXX",
					"UVXY",
					"DBA",#综合型商品ETF
					"DBC",#
					"IEF",#
					"TLT",#
					"GLD",#
					"GDX",#
					"UUP",#美元
					"FXE",#欧元
					"FXF",#瑞郎
					"FXY",#日元
					"FXB",#英镑
					"CYB",#人民币

					"EWA",#澳大利亚
					"EWJ",#日本
					"EWY",#韩国
					"EWH",#香港
					"EWT",#台湾
					"FXI",#中国富时A50
					"INDA",#印度
					"EWG",#德国
					"EWL",#瑞士
					"EWU",#英国
					"RSX",#俄罗斯
					"EWZ",#巴西
					"EWW",#墨西哥
					"EWC",#加拿大

					"TQQQ",##纳指100x3
					"UPRO",#标普x3                        
					"UDOW",#道指x3
					"SMH",# 半导体行业x1
					"USD", # 半导体行业x2
					"SOXL", # 半导体行业x3
					"UYG",# 金融双倍做多
					"URE",# 房地产双倍做多
					"DRN",#x3
					"DIG",# 能源双倍做多
					"ERX",#x3
					"UYM",# 原材料双倍做多 
					"MATL",#x3
					"NUGT",#黄金矿业三倍做多
					"EET",# 新兴市场指数双倍做多
					"EDC",#x3
					"XPP",# 中国股票指数x2
					"UCO",# 石油x2
					"DGP",# 黄金x2
					"UGL",# 黄金x2
					"AGQ",# 白银x2
					"EUO",# 美元x2
					"UBT",# 政府长期债券x2
					"TMF",# 政府长期债券x3
					"TNA",# 罗素2000	ETF三倍做多
					"YINN", #三倍做多中国ETF
					"PBW", #清洁能源
					"FAS", # 三倍做多罗素金融股指数ETF
					"PAVE",#美股基建ETF
					"SCO"#两倍做空彭博原油ETF


					]))
		return tickers



	def global_indices():
		tickers = list(set(['^GSPC',
					'^DJI',
					'^IXIC',
					'^HSI',
					'000001.SS']))
		return tickers



	def sector_etf():
		tickers = list(set(['XLE',#能源
					'XLV',#医疗保健
					'XLI',#工业
					'XLF',#金融
					'XLY',#可选消费
					'XLP',#必选消费
					'XLB',#原材料
					'XLK',#科技
					'XLU',#公共事业
					'XLRE',#房地产
					'XLC',#通讯服务
					
					'ARKG',#ARK系列 - 生物科技&基因工程
					'ARKK',#ARK系列 - 颠覆式科技
					'ARKF',#ARK系列 - 金融科技
					'ARKW',#
					'ARKQ']))
		return tickers


		
class CN_TICKERS_POOL():
	def CN_fund_tickers():
		tickers = list(set(['161725','009865','003096','011609','005827','001838','110011',
							'006792','010849','161005','010358','000452','001508','004233',
							'001888','004851']))
		return tickers


class HK_TICERS_POOL():
	def HK_tickers():
		tickers = list(set(['9988.HK','6618.HK','3888.HK','0418.HK',
		'9999.HK','1909.HK','2318.HK','0941.HK','0005.HK','9999.HK',
		'1070.HK','6969.HK','0241.HK','1137.HK','6055.HK','6862.HK',
		'0700.HK','2051.HK','6993.HK','6618.HK','9992.HK','3690.HK',
		'1810.HK','1024.HK','1119.HK','8029.HK','0493.HK','2382.HK',
		'0175.HK','6969.HK','1610.HK','2400.HK','0788.HK','0981.HK',
		'0728.HK','0762.HK','1833.HK','0285.HK','0268.HK','0763.HK',
		'6823.HK','6060.HK','0151.HK','1347.HK','3799.HK','2018.HK',
		'0522.HK','0008.HK','2400.HK','0552.HK','1458.HK','6088.HK',
		'1478.HK','0303.HK','1070.HK','1310.HK','9990.HK','0799.HK',
		'3738.HK','1675.HK','0142.HK','0777.HK','1883.HK','2038.HK',
		'0419.HK','6869.HK','1385.HK','0302.HK','0797.HK','3798.HK',
		'0751.HK','1475.HK','0215.HK','2342.HK','2100.HK','0315.HK',
		'6820.HK','1909.HK','1415.HK','8032.HK','1119.HK','0732.HK',
		'1588.HK','2309.HK','0698.HK','2369.HK','0596.HK','1566.HK',
		'0186.HK','2255.HK','0439.HK','6188.HK','1979.HK'
		]))
		return tickers


class FUTURES():
	def futures():
		tickers = list(set(['NQ=F','ES=F','YM=F','GC=F','CL=F','XAU=F',
							'RB=F','CT=F','ALI=F'
							]))
		return tickers



def all_tickers():

	tickers = list(set(US_TICKERS_POOL.us_tech_stocks() + \
			              US_TICKERS_POOL.us_blockchain_stocks() + \
			              US_TICKERS_POOL.us_saas_stocks() + \
			              US_TICKERS_POOL.us_biotech_stocks() + \
			              US_TICKERS_POOL.us_new_energy_stocks() + \
			              US_TICKERS_POOL.global_etfs() + \
			              US_TICKERS_POOL.sector_etf() + \
			              US_TICKERS_POOL.global_indices() +\
			              US_TICKERS_POOL.us_healthcare_stocks() +\
			              US_TICKERS_POOL.us_fin_stocks() +\
			              US_TICKERS_POOL.us_utilities_stocks() +\
			              US_TICKERS_POOL.us_materials_stocks() +\
			              US_TICKERS_POOL.other_us_stocks() +\
							US_TICKERS_POOL.us_consumer_discretionary_stocks() +\
							US_TICKERS_POOL.us_consumer_staples_stocks() +\
							US_TICKERS_POOL.us_industrial_stocks() +\
							US_TICKERS_POOL.us_real_estate_stocks() +\
							US_TICKERS_POOL.us_tele_entertainment_stocks() +\
			              HK_TICERS_POOL.HK_tickers()
			              ))
	print("All Tickers: {}".format(len(tickers)))
	return tickers



class LOAD_DATA():
	#all_csvs = "C:/Users/Administrator/CE_github_2/data_pipeline/Data/*.csv"
	# Valid intervals: [1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo]
	def read_downloaded_data(all_csvs): 
		load_count = 1
		data_list = []
		len_tickers = len(glob.glob(all_csvs)[:])
		for fname in glob.glob(all_csvs)[:]:
			if load_count%10000==0:
				print("=======================Sleeping======================")
				time.sleep(30)
			else:
				print ("Loadig Data from DB: No.{} / {}: {}".format(load_count, len_tickers, fname))
				data = pd.read_csv(fname)
				data_list.append(data)
			load_count +=1
		print("All Data Loaded")
		return data_list


	def read_data_from_folders(ticker_list, csv_folder):
		# csv_folder = "C:/Users/Administrator/CE_github_2/data_pipeline/Data/"
		load_count = 1
		data_list = []
		len_tickers = len(ticker_list)
		for ticker in ticker_list:
			try:
				print ("Loadig Data from DB: No.{} / {}: {}".format(load_count, len_tickers, ticker))
				data = pd.read_csv(csv_folder+ticker+".csv")
				data_list.append(data)
			except Exception as e:
				print(ticker, "Error:", e)
			load_count +=1
		return data_list
		

	def read_data_from_yf_5m(tickers_list):
		load_count = 1
		data_list = []
		len_tickers = len(tickers_list)
		for ticker in tickers_list:
			if load_count%100==0:
				print("=======================Sleeping======================")
				time.sleep(30)
			else:
				print("Loading from YahooFinance: No.{} / {}: {}".format(load_count, len_tickers, ticker))
				data = yf.download(
			            tickers=[ticker],
			            # use "period" instead of start/end
			            # valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
			            # (optional, default is '1mo')
			            period="5d",
			            # fetch data by interval (including intraday if period < 60 days)
			            # valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
			            # (optional, default is '1d')
			            interval="5m")
				data.dropna(inplace = True)
				data.reset_index(inplace = True)
				data["Ticker"] = ticker
				data_list.append(data)
			load_count+=1
		print("All Data Loaded")
		return data_list
		


	def read_data_from_yf_15m(tickers_list):
		load_count = 1
		data_list = []
		len_tickers = len(tickers_list)
		for ticker in tickers_list:
			if load_count%100==0:
				print("=======================Sleeping======================")
				time.sleep(30)
			else:
			    print("Loading from YahooFinance: No.{} / {}: {}".format(load_count, len_tickers, ticker))
			    data = yf.download(
			            tickers=[ticker],
			            # use "period" instead of start/end
			            # valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
			            # (optional, default is '1mo')
			            period="5d",
			            # fetch data by interval (including intraday if period < 60 days)
			            # valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
			            # (optional, default is '1d')
			            interval="15m")
			    data.dropna(inplace = True)
			    data.reset_index(inplace = True)
			    data["Ticker"] = ticker
			    data_list.append(data)
			load_count+=1
		print("All Data Loaded")
		return data_list


	def read_data_from_yf_D(tickers_list):
		load_count = 1
		data_list = []
		len_tickers = len(tickers_list)
		for ticker in tickers_list:
			if load_count%100==0:
				print("=======================Sleeping======================")
				time.sleep(30)
			else:
			    print("Loading from YahooFinance: No.{} / {}: {}".format(load_count, len_tickers, ticker))
			    data = yf.download(
			            tickers=[ticker],
			            # use "period" instead of start/end
			            # valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
			            # (optional, default is '1mo')
			            period="252d",
			            # fetch data by interval (including intraday if period < 60 days)
			            # valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
			            # (optional, default is '1d')
			            interval="1d")
			    data.dropna(inplace = True)
			    data.reset_index(inplace = True)
			    data["Ticker"] = ticker
			    data_list.append(data)
			load_count+=1
		print("All Data Loaded")
		return data_list
		


	def read_data_from_yf_W(tickers_list):
		load_count = 1
		data_list = []
		len_tickers = len(tickers_list)
		for ticker in tickers_list:
			if load_count%100==0:
				print("=======================Sleeping======================")
				time.sleep(30)
			else:
			    print("Loading from YahooFinance: No.{} / {}: {}".format(load_count, len_tickers, ticker))
			    data = yf.download(
			            tickers=[ticker],
			            # use "period" instead of start/end
			            # valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
			            # (optional, default is '1mo')
			            period="500d",
			            # fetch data by interval (including intraday if period < 60 days)
			            # valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
			            # (optional, default is '1d')
			            interval="1wk")
			    data.dropna(inplace = True)
			    data.reset_index(inplace = True)
			    data["Ticker"] = ticker
			    data_list.append(data)
			load_count+=1
		print("All Data Loaded")
		return data_list
		


	#指定默认字体
	matplotlib.rcParams['font.sans-serif'] = ['SimHei']
	matplotlib.rcParams['font.family']='sans-serif'
	#解决负号'-'显示为方块的问题
	matplotlib.rcParams['axes.unicode_minus'] = False

	# 抓取网页
	def get_url(url, params=None, proxies=None):
	    rsp = requests.get(url, params=params, proxies=proxies)
	    rsp.raise_for_status()
	    return rsp.text

	# 从网页抓取数据
	def get_fund_data(code,per=10,sdate='',edate='',proxies=None):
	    url = 'http://fund.eastmoney.com/f10/F10DataApi.aspx'
	    params = {'type': 'lsjz', 'code': code, 'page':1,'per': per, 'sdate': sdate, 'edate': edate}
	    html = LOAD_DATA.get_url(url, params, proxies)
	    soup = BeautifulSoup(html, 'html.parser')

	    # 获取总页数
	    pattern=re.compile(r'pages:(.*),')
	    result=re.search(pattern,html).group(1)
	    pages=int(result)

	    # 获取表头
	    heads = []
	    for head in soup.findAll("th"):
	        heads.append(head.contents[0])

	    # 数据存取列表
	    records = []

	    # 从第1页开始抓取所有页面数据
	    page=1
	    while page<=pages:
	        params = {'type': 'lsjz', 'code': code, 'page':page,'per': per, 'sdate': sdate, 'edate': edate}
	        html = LOAD_DATA.get_url(url, params, proxies)
	        soup = BeautifulSoup(html, 'html.parser')

	        # 获取数据
	        for row in soup.findAll("tbody")[0].findAll("tr"):
	            row_records = []
	            for record in row.findAll('td'):
	                val = record.contents

	                # 处理空值
	                if val == []:
	                    row_records.append(np.nan)
	                else:
	                    row_records.append(val[0])

	            # 记录数据
	            records.append(row_records)

	        # 下一页
	        page=page+1

	    # 数据整理到dataframe
	    np_records = np.array(records)
	    data= pd.DataFrame()
	    for col,col_name in enumerate(heads):
	        data[col_name] = np_records[:,col]

	    return data


	def read_data_from_Sina(ticker_list, start_date, end_date):
		data_list = []
		load_count = 1
		len_tickers = len(ticker_list)
		for ticker in ticker_list:
			if load_count%100==0:
				print("=======================Sleeping======================")
				time.sleep(30)
			else:
				print("Loading from SinaFinance: No.{} / {}: {}".format(load_count, len_tickers, ticker))
				data=LOAD_DATA.get_fund_data(ticker,per=49,sdate=start_date,edate=end_date)
				# 修改数据类型
				data['净值日期']=pd.to_datetime(data['净值日期'],format='%Y/%m/%d')
				data['单位净值']= data['单位净值'].astype(float)
				data['Open'] = data['单位净值'].astype(float)
				data['High'] = data['单位净值'].astype(float)
				data['Low'] = data['单位净值'].astype(float)
				data['Close'] = data['单位净值'].astype(float)
				data['Volume'] = data['单位净值'].astype(float)

				# 按照日期升序排序并重建索引
				data=data.sort_values(by='净值日期',axis=0,ascending=True).reset_index(drop=True)
				data = data[['净值日期','Open','High','Low','Close','单位净值','Volume']]
				data.columns = ['Date', 'Open','High','Low','Close','Adj Close', 'Volume']
				data['Ticker'] = ticker
				data_list.append(data)
			load_count+=1
		print("All Data Loaded")
		return data_list


	def download_data_from_Sina(ticker_list, start_date, end_date, csv_folder):
		data_list = []
		load_count = 1
		len_tickers = len(ticker_list)
		for ticker in ticker_list:
			if load_count%100==0:
				print("=======================Sleeping======================")
				time.sleep(30)
			else:
				print("Downloading from SinaFinance: No.{} / {}: {}".format(load_count, len_tickers, ticker))
				data=LOAD_DATA.get_fund_data(ticker,per=49,sdate=start_date,edate=end_date)
				# 修改数据类型
				data['净值日期']=pd.to_datetime(data['净值日期'],format='%Y/%m/%d')
				data['单位净值']= data['单位净值'].astype(float)
				data['Open'] = data['单位净值'].astype(float)
				data['High'] = data['单位净值'].astype(float)
				data['Low'] = data['单位净值'].astype(float)
				data['Close'] = data['单位净值'].astype(float)
				data['Volume'] = data['单位净值'].astype(float)
				# 按照日期升序排序并重建索引
				data=data.sort_values(by='净值日期',axis=0,ascending=True).reset_index(drop=True)
				data = data[['净值日期','Open','High','Low','Close','单位净值','Volume']]
				data.columns = ['Date', 'Open','High','Low','Close','Adj Close', 'Volume']
				data['Ticker'] = ticker
				data.to_csv(csv_folder+ticker+".csv")
			load_count+=1
		print("All Data Downloaded")
		# return data_list

	def today_dt():
		today = str(datetime.now().date())
		return today

	def easy_download(start, end, us_db_path, ticker_list):
		load_count = 1
		for ticker in ticker_list:
			if load_count%100==0:
				print("=======================Sleeping======================")
				time.sleep(30)
			else:
				try:
					data = yf.download(ticker, start=start, end=end)
					data.reset_index(inplace=True)
					data['Ticker'] = ticker
					data.to_csv(us_db_path + ticker + ".csv", index=False)
					print("No.{}: {} data file created: {}".format(load_count, ticker, end))
				except Exception as e:
					print("Failed to download: {}".format(ticker))
					print(e)
			load_count +=1
		print("-----------------------------")
		print("All done for {}!".format(end))



# class US_TICKERS_POOL():

# #---------------------------------0305---------------------------------
# 	def us_consumer_discretionary_stocks():#非必需消费品
# 		tickers = list(set(['HD','PDD','NKE','MCD','JD','SBUX','LOW','LULU','LVS',
# 			'RACE','PTON','CHWY','CMG','EBAY','AZO','BBY','CZR']))
# 		return tickers

# 	def us_consumer_staples_stocks():#必需消费品
# 		tickers = list(set(['WMT','PG','KO','PEP','COST','PM','EL','TGT','CL','STZ',
# 			'KHC','TSN','SAM']))
# 		return tickers	

# 	def us_industrial_stocks():
# 		tickers = list(set(['HON','UPS','UNP','BA','RTX','CAT','MMM','LMT','GE']))
# 		return tickers	

# 	def us_real_estate_stocks():
# 		tickers = list(set(['AMT','PLD','CCI','EQIX','DLR','PSA','SBAC']))
# 		return tickers	

# 	def us_tele_entertainment_stocks():
# 		tickers = list(set(['GOOGL','FB','DIS','VZ','NFLX','CMCSA','T','TMUS',
# 			'CHTR','ATVI','SPOT','BILI']))
# 		return tickers	


# #--------------------------------0304--------------------------------
# 	def us_materials_stocks():
# 		tickers = list(set(['LIN','SHW','APD','ECL','DD','SCCO','NEM']))
# 		return tickers		

# 	def us_utilities_stocks():
# 		tickers = list(set(['NEE','DUK','SO','D','AEP','EXC','SRE','XEL','ES','WES']))
# 		return tickers			

# 	def us_fin_stocks():
# 		tickers = list(set(['JPM','BAC','C','WFC','MS','BLK','GS','AXP','BX','CB','MET']))
# 		return tickers	

# 	def us_healthcare_stocks():
# 		tickers = list(set(['JNJ','UNH','PFE','MRK','ABT','TMO','ABBV','DHR','LLY','MDT','BMY',
# 			'AMGN','ISRG','CVS']))
# 		return tickers			
# #-------------------------------pre 0304-----------------------------
# 	def us_tech_stocks():
# 		tickers = list(set(['V','MA','NFLX','CSCO','ADBE','MSFT','CRM','TSLA','GOOG','FB','GOOGL','TSM',
# 				'AMZN','AAPL','PYPL','INTC','NVDA','QCOM','MU','BABA']))
# 		return tickers

# 	def us_blockchain_stocks():
# 		tickers = list(set(['EBON','CAN','LFIN','CNET','AMD','FB','NVDA','XNET','SRAX','RENN','OSTK',
# 					'RIOT','GBTC','MARA','NCTY']))
# 		return tickers


# 	def us_saas_stocks():
# 		tickers = list(set(['HUBS','CDAY','ORCL','WDAY','COUP','INTU','SMAR','OKTA','ADP','PAYC','WORK',
# 		'ADBE','MSFT','SPLK','ZM','RNG','CRM','SAP','GDDY','DBX','VEEV','NOW','TEAM',
# 		'ZEN','ADSK','TTD','DDOG','TWLO','SHOP','DOCU','SQ','CRWD']))
# 		return tickers

# 	def us_biotech_stocks():
# 		tickers = list(set(['AHPI','INO','IBIO','MRNA','VIR','NNVC','GILD','AZN','PFE','JNJ','BNTX','ABT',
# 		'CBLI','ARPO','VCNX','NVAX','VXRT','CRVS','APT','CODX','EQ','ACOR','LAKE',
# 		'AEMD','DFFN']))
# 		return tickers

# 	def us_new_energy_stocks():
# 		tickers = list(set(['TSLA','NIO','LI','XPEV','SQM','ALB','LTHM','FMC','LAC','QS','VLDR','LAZR',
# 					'SBE','BLNK','XL','ENPH','PLUG','RUN','FLS','NOVA','FCEL']))
# 		return tickers

# 	def other_us_stocks():
# 		tickers = list(set(['OPEN']))
# 		return tickers

# 	def global_etfs():
# 		tickers = list(set(["SPY",
# 					"QQQ",
# 					"VXX",
# 					"UVXY",
# 					"DBA",#综合型商品ETF
# 					"DBC",#
# 					"IEF",#
# 					"TLT",#
# 					"GLD",#
# 					"GDX",#
# 					"UUP",#美元
# 					"FXE",#欧元
# 					"FXF",#瑞郎
# 					"FXY",#日元
# 					"FXB",#英镑
# 					"CYB",#人民币
# 					"TQQQ"
# 					]))
# 		return tickers


# 	def global_indices():
# 		tickers = list(set(['^GSPC',
# 					'^DJI',
# 					'^IXIC',
# 					'^HSI',
# 					'000001.SS']))
# 		return tickers



# 	def sector_etf():
# 		tickers = list(set(['XLE',#能源
# 					'XLV',#医疗保健
# 					'XLI',#工业
# 					'XLF',#金融
# 					'XLY',#可选消费
# 					'XLP',#必选消费
# 					'XLB',#原材料
# 					'XLK',#科技
# 					'XLU',#公共事业
# 					'XLRE',#房地产
# 					'ARKG',#ARK系列 - 生物科技&基因工程
# 					'ARKK',#ARK系列 - 颠覆式科技
# 					'ARKF',#ARK系列 - 金融科技
# 					'ARKW',#
# 					'ARKQ']))
# 		return tickers