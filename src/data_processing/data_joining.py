import pandas as pd
from datetime import timedelta, date

# Read review data as Pandas data frames
fashion = pd.read_csv('AMAZON_FASHION_2018.csv')
electronics = pd.read_csv('Electronics_2018.csv')
beauty = pd.read_csv('Luxury_Beauty_2018.csv')
toy = pd.read_csv('Toys_and_Games_2018.csv')


# Fetch all calendar days in 2018
def dateRange(date1, date2):
    for n in range(int((date2 - date1).days) + 1):
        yield date1 + timedelta(n)


year2018 = []
ret = []
start_dt = date(2018, 1, 1)
end_dt = date(2018, 12, 31)
for dt in dateRange(start_dt, end_dt):
    year2018.append(dt.strftime('%Y/%m/%d'))
    ret.append(float("NaN"))

# Create a dummy data frame containing all calender dates in 2018 and temporarily set return as 'Null'
D = pd.DataFrame()
D['Date'] = year2018
D['stockReturn'] = ret


# Read daily return data as Pandas data frames
sp = pd.read_csv('SP_Daily_Return.csv')
sp['Date'] = pd.to_datetime(sp['Date']).dt.strftime('%Y/%m/%d')

# Concatenate the full date dummy frame with SP daily return, and drop duplicates in date
# i.e. missing date from sp data frame would be filled with "NaN" in "stockReturn" column
# while if the date is already existing in sp, the row from dummy frame would be dropped
res = pd.concat([sp, D]).drop_duplicates(subset=['Date'])
# Sort the result by date
res = res.sort_values(by=['Date'])
# Reset index for the data frame
res = res.reset_index(drop=True)
# Fill missing stock returns with previous value
res = res.fillna(method='ffill')


# Reformat the 'reivewTime' in raw data in to YYYY/MM/DD and join with S&P return data frame
def join_process(df):
    df['Date'] = pd.to_datetime(df['reviewTime']).dt.strftime('%Y/%m/%d')
    return pd.merge(df, res, on='Date', how='left')

# A list of 4 joined data frames
joined = [join_process(df) for df in [fashion, electronics, beauty, toy]]

