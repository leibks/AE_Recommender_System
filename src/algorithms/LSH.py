import numpy as np
import pandas as pd
import re
import time
from datasketch import MinHash, MinHashLSHForest

#Preprocess will split a string of text into individual tokens/shingles based on whitespace.
def preprocess(text):
    text = re.sub(r'[^\w\s]','',text)
    tokens = text.lower()
    tokens = tokens.split()
    return tokens

# text = 'The devil went down to Georgia'
# print('The shingles (tokens) are:', preprocess(text))

#Number of Permutations
permutations = 128

def get_forest(data, perms):
    start_time = time.time()
    
    minhash = []
    
    for text in data['reviewText']:
        tokens = preprocess(text)
        m = MinHash(num_perm=perms)
        for s in tokens:
            m.update(s.encode('utf8'))
        minhash.append(m)
        
    forest = MinHashLSHForest(num_perm=perms)
    
    for i,m in enumerate(minhash):
        forest.add(i,m)
        
    forest.index()
    
    print('It took %s seconds to build forest.' %(time.time()-start_time))
    
    return forest

def predict(text, database, perms, num_results, forest):
    start_time = time.time()
    
    tokens = preprocess(text)
    m = MinHash(num_perm=perms)
    for s in tokens:
        m.update(s.encode('utf8'))
        
    idx_array = np.array(forest.query(m, num_results))
    if len(idx_array) == 0:
        return None # if your query is empty, return none
    
    result = database.iloc[idx_array]['asin']
    
    print('It took %s seconds to query forest.' %(time.time()-start_time))
    
    return result

raw_reviews = pd.read_csv('resource\sample_data\sample_electronics.csv')
product_reviews = raw_reviews.groupby("asin", as_index=False).agg(list).eval("reviewText = reviewText.str.join(' ')")
forest = get_forest(product_reviews, permutations)

num_recommendations = 5
asin = 'This is not a plug and play external dvd/cd drive.  One must download software to make it work which I cannot figure out'
result = predict(asin, product_reviews, permutations, num_recommendations, forest)
print('\n Top Recommendation(s) is(are) \n', result)