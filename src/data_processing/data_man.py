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


def download(directory):
    data_source = ['http://deepyeti.ucsd.edu/jianmo/amazon/categoryFiles/Luxury_Beauty.json.gz',
                   'http://deepyeti.ucsd.edu/jianmo/amazon/categoryFiles/Electronics.json.gz',
                   'http://deepyeti.ucsd.edu/jianmo/amazon/categoryFiles/AMAZON_FASHION.json.gz',
                   'http://deepyeti.ucsd.edu/jianmo/amazon/categoryFiles/Toys_and_Games.json.gz',
                   'http://deepyeti.ucsd.edu/jianmo/amazon/metaFiles/meta_Luxury_Beauty.json.gz',
                   'http://deepyeti.ucsd.edu/jianmo/amazon/metaFiles/meta_Electronics.json.gz',
                   'http://deepyeti.ucsd.edu/jianmo/amazon/metaFiles/meta_AMAZON_FASHION.json.gz',
                   'http://deepyeti.ucsd.edu/jianmo/amazon/metaFiles/meta_Toys_and_Games.json.gz']
    sp500_url = "https://query1.finance.yahoo.com/v7/finance/download/%5EGSPC?period1=1509408000&period2=1572912000&interval=1mo&events=history&includeAdjustedClose=true"

    file_names = []
    for url in data_source:
        file_name = url.split('/')[-1]
        file_names.append(file_name.split('.')[0])
        full_dir = directory + file_name
        if os.path.exists(full_dir):
            print("File already exists, skip downloading for ", file_name)
        else:
            print("Downloading to file {}\n".format(full_dir))
            wget.download(url, out=full_dir)

    sp500_name = directory + 'sp500.csv'
    if os.path.exists(sp500_name):
        print("File already exists, skip downloading for ", sp500_name)
    else:
        r = requests.get(sp500_url, allow_redirects=True)
        open('sp500_name', 'wb').write(r.content)

    print("Finished downloading data...")
    return file_names, sp500_name


# Unzip and remove useless columns
def process_data(names, directory):
    # unzip
    tmp = []
    for name in names:
        file_to_unzip = directory + name + '.json.gz'
        json_file_name = directory + name + '.json'
        if os.path.exists(json_file_name):
            print("Already has unzipped file " + json_file_name)
            continue
        else:
            print("Start unzip and processing...", name)
            with gzip.open(file_to_unzip, 'rb') as f_in:
                with open(json_file_name, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)

    chunk_list = []
    i = 1
    for name in names:
        expected_name = '{}{}_2018.csv'.format(directory, name)
        if 'meta' not in name:
            if os.path.exists(expected_name):
                print("File {} already exists".format(expected_name))
                continue
            file_name = '{}{}.json'.format(directory, name)
            for chunk in pd.read_json(file_name, lines=True, chunksize=100000):
                df_chunk = chunk[chunk['unixReviewTime'] > review_date_threshold]
                chunk_list.append(df_chunk.drop(['image', 'summary', 'style', 'verified'], axis=1))
                print("processing chunk ", i)
                i += 1
            data_df = pd.concat(chunk_list)
            print('data size ', len(data_df))
            meta_df = pd.read_json('{}meta_{}.json'.format(directory, name), lines=True)
            if 'main_cat' not in meta_df.columns:
                meta_df = meta_df[['title', 'price', 'asin']]
                meta_df['main_cat'] = name
            else:
                meta_df = meta_df[['title', 'main_cat', 'price', 'asin']]
            meta_df = meta_df[meta_df['title'].str.contains('\n')==False]
            print('metadata size ', len(meta_df))
            df = pd.merge(data_df, meta_df, how='left', on='asin')
            df = df.loc[df.astype(str).drop_duplicates().index]  # remove duplicates
            df = df[(df['price'].str.len() < 20) & (df['price'].str.len() > 0)]
            df = df.dropna(subset=['price'])
            df.to_csv(expected_name)


# Fetch all calendar days in 2018
def dateRange(date1, date2):
    for n in range(int((date2 - date1).days) + 1):
        yield date1 + timedelta(n)


# Data joining
def data_join(file_names, sp500_name, directory):
    # Read review data as Pandas data frames
    df_list = []
    for name in file_names:
        if 'meta' in name:
            continue
        name = name + '_2018.csv'
        print("Loaded csv:", directory + name)
        df_list.append(pd.read_csv(directory + name))
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
    return joined


def __main():
    directory = os.getcwd()
    # Download raw data
    if len(sys.argv) > 1:
        directory = sys.argv[1]
    else:
        print("Downloading data files to current working directory: ", directory)

    if not directory.endswith('/'):
        directory = directory + '/'

    file_names, sp500_name = download(directory)
    process_data(file_names, directory)
    data_join(file_names, sp500_name, directory)


__main()