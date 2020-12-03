import sys
import os
import wget
import requests
import gzip
import shutil
import pandas as pd
from datetime import datetime, timedelta, date

pd.set_option('display.max_columns', None)
review_date_threshold = datetime.strptime('01 01 2018', '%m %d %Y').timestamp()

# Download raw data
directory = sys.argv[1]
if directory == None:
    directory = os.getcwd()
    print("Downloading data files to current working directory: ", directory)
if not directory.endswith('/'):
    directory = directory + '/'

data_source = ['http://deepyeti.ucsd.edu/jianmo/amazon/categoryFiles/Luxury_Beauty.json.gz',
               'http://deepyeti.ucsd.edu/jianmo/amazon/categoryFiles/Electronics.json.gz',
               'http://deepyeti.ucsd.edu/jianmo/amazon/categoryFiles/AMAZON_FASHION.json.gz',
               'http://deepyeti.ucsd.edu/jianmo/amazon/categoryFiles/Toys_and_Games.json.gz']
sp500_url = "https://query1.finance.yahoo.com/v7/finance/download/%5EGSPC?period1=1509408000&period2=1572912000&interval=1mo&events=history&includeAdjustedClose=true"

file_names = []
for url in data_source:
    file_name = url.split('/')[-1]
    file_names.append(file_name)
    full_dir = directory + file_name
    if os.path.exists(full_dir):
        print("File already exists, skip downloading for ", file_name)
    else:
        print("Downloading to file ", full_dir)
        wget.download(url, out=full_dir)

sp500_name = directory + 'sp500.csv'
if os.path.exists(sp500_name):
    print("File already exists, skip downloading for ", sp500_name)
else:
    r = requests.get(sp500_url, allow_redirects=True)
    open('sp500_name', 'wb').write(r.content)

print("Finished downloading data...")


# Unzip and remove useless columns
def process_data(names):
    # unzip
    tmp = []
    for name in names:
        print("Start unzip and processing...", name)

        with gzip.open(name, 'rb') as f_in:
            json_file_name = name.replace('.gz', '')
            tmp.append(json_file_name)
            with open(json_file_name, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
    names = tmp
    chunk_list = []
    i = 1
    for name in names:
        for chunk in pd.read_json('{}{}.json'.format(directory, name), lines=True,
                                  chunksize=100000):
            df_chunk = chunk[chunk['unixReviewTime'] > review_date_threshold]
            chunk_list.append(df_chunk.drop(['image', 'summary', 'style', 'verified'], axis=1))
            print("processing chunk ", i)
            i += 1
        data_df = pd.concat(chunk_list)
        print('data size ', len(data_df))

        meta_df = pd.read_json('{}meta_{}.json'.format(directory, name), lines=True)
        meta_df = meta_df[['title', 'also_buy', 'rank', 'main_cat', 'price', 'asin']]
        print('metadata size ', len(meta_df))
        df = pd.merge(data_df, meta_df, how='left', on='asin')
        df = df.loc[df.astype(str).drop_duplicates().index]  # remove duplicates
        df.to_csv('{}{}_2018.csv'.format(directory, name))


process_data(file_names)

# Data joining
# Read review data as Pandas data frames
df_list = []
for name in file_names:
    if '.gz' in name:
        name = name.replace('json.gz', '_2018.csv')
    print("Loaded csv:", directory + name)
    df_list.append(pd.read_csv(directory + name))


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

sp = pd.read_csv(sp500_name)
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
joined = [join_process(df) for df in df_list]
