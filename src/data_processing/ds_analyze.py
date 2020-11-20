import os
import json
import gzip
import pandas as pd
from datetime import datetime

review_date_threshold = datetime.strptime('01 01 2018', '%m %d %Y').timestamp()
### load the data
pd.set_option('display.max_columns', None)

# names = ['Luxury_Beauty', 'Electronics', 'AMAZON_FASHION']
names = ['Toys_and_Games']


def process_data(name):
    print("Processing data ", name)
    chunk_list = []
    i = 1
    for chunk in pd.read_json('/Users/yolanda/PycharmProjects/DataScience/project/{}.json'.format(name), lines=True,
                              chunksize=100000):
        df_chunk = chunk[chunk['unixReviewTime'] > review_date_threshold]
        chunk_list.append(df_chunk.drop(['image', 'summary', 'style', 'verified'], axis=1))
        print("processing chunk ", i)
        i += 1
    data_df = pd.concat(chunk_list)
    print('data size ', len(data_df))

    meta_df = pd.read_json('/Users/yolanda/PycharmProjects/DataScience/project/meta_{}.json'.format(name), lines=True)
    meta_df = meta_df[['title', 'also_buy', 'rank', 'main_cat', 'price', 'asin']]
    print('metadata size ', len(meta_df))
    # print(meta_df.head())
    df = pd.merge(data_df, meta_df, how='left', on='asin')
    df = df.loc[df.astype(str).drop_duplicates().index]  # remove duplicates
    df.to_csv('/Users/yolanda/PycharmProjects/DataScience/project/{}_2018.csv'.format(name))


for n in names:
    process_data(n)
