from coinbase.wallet.client import Client
import datetime as dt
import pandas as pd
import cbpro

all_dates, all_prices = [], []
coinbase_API_key = '8zFgQlg6TZzUJQLP'
coinbase_API_secret = 'ATRT7EHe4tLUSrKPhWOY9S3A7ztQzao2'
client = Client(coinbase_API_key, coinbase_API_secret)

def daterange(start_date, end_date):
	for date in range(int((end_date - start_date).days)):
		yield start_date + dt.timedelta(date)

start_date = dt.date(2016, 1, 1) # CHANGE FOR DIFFERENT CRYPTOS
end_date = dt.date.today()

for single_date in daterange(start_date, end_date):
	price = client.get_spot_price(currency_pair='BTC-USD', date=single_date)
	all_dates.append(single_date)
	all_prices.append(float(price.amount))

next_date = dt.date.today() - dt.timedelta(days=1)
price = client.get_spot_price(currency_pair='BTC-USD', date=next_date)  
all_prices.append(float(price.amount))
price = client.get_spot_price(currency_pair='BTC-USD', date=dt.date.today())
all_prices.append(float(price.amount))

prices_df = pd.DataFrame(list(zip(all_dates, all_prices)), columns=['Date', 'Price'])
prices_df.style.hide_index

prices_df.to_csv('BTC.csv')