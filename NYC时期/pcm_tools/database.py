import urllib.request, urllib.parse, urllib.error
import os
import xlrd # module for excel file reading
import pandas as pd

def sptickers():
	fName = 'spy_holdings.xls'

	if not os.path.exists('data'):os.mkdir('data')
	urllib.request.urlretrieve("https://www.spdrs.com/site-content/xls/SPY_All_Holdings.xls?fund=SPY&docname=All+Holdings&onyx_code1=1286&onyx_code2=1700", 
	                          fName)

	wb = xlrd.open_workbook(fName) # open xls file, create a workbook
	sh = wb.sheet_by_index(0) # select first sheet

	data = {'name':[], 'symbol':[], 'weight':[], 'sector':[]} # create a new and empty dictionary for storing the later data
	for rowNr in range(5, 505): # cycle through the rows
	    v = sh.row_values(rowNr) # get all row values
	    data['name'].append(v[0])
	    data['symbol'].append(v[1]) #symbol is in the second column, append it to the list
	    data['weight'].append(float(v[2])) 
	    data['sector'].append(v[3])
	    
	holdings = pd.DataFrame(data)



	# Categorize the dataframe based on sectors:
	tech = holdings[holdings['sector'] == 'Information Technology']
	healthcare = holdings[holdings['sector'] == 'Health Care']
	utilities = holdings[holdings['sector'] == 'Utilities']
	energy = holdings[holdings['sector'] == 'Energy']
	tele = holdings[holdings['sector'] == 'Telecommunication Services']
	consumer_d = holdings[holdings['sector'] == 'Consumer Discretionary']
	financials = holdings[holdings['sector'] == 'Financials']
	consumer_staples = holdings[holdings['sector'] == 'Consumer Staples']
	industrials = holdings[holdings['sector'] == 'Industrials']
	realestate = holdings[holdings['sector'] == 'Real Estate']


	# create a whole list for all 500 stocks:
	sp500_tickers = holdings['symbol'].tolist()

	# Create each industry's ticker list
	healthcare_tickers = healthcare['symbol'].tolist()
	tech_tickers = tech['symbol'].tolist()
	utilities_tickers = utilities['symbol'].tolist()
	energy_tickers = energy['symbol'].tolist()
	tele_tickers = tele['symbol'].tolist()
	consumer_d_tickers = consumer_d['symbol'].tolist()
	financials_tickers = financials['symbol'].tolist()
	consumer_staples_tickers = consumer_staples['symbol'].tolist()
	industrials_tickers = industrials['symbol'].tolist()
	realestate_tickers = realestate['symbol'].tolist()

	return sp500_tickers 

	# tele_tickers, tech_tickers, healthcare_tickers, utilities_tickers, energy_tickers, 
	# financials_tickers, consumer_staples_tickers, consumer_d_tickers, realestate_tickers, industrials_tickers
