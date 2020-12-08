import numpy as np
from sklearn import random_projection
from sklearn.metrics.pairwise import cosine_similarity


# Generate the hashtable for easier finding similar items.
# Same hash value means a list of items under this
# hash value are most similar
# random_type="gau" means using gaussian random projection
# random_type="sparse" means using sparse random projection
class HashTable:

    def __init__(self, input_dim, random_type="gau", hash_size=3):
        self.input_dim = input_dim
        # key: hash value, value: a list of item names
        self.hash_table = {}
        # key: item_name, value: generated hash value for this item
        self.hash_values = {}
        self.hash_size = hash_size
        self.projections = np.random.randn(self.hash_size, self.input_dim)
        # if random_type == "gau":
        #     self.projections = random_projection.GaussianRandomProjection()
        # elif random_type == "sparse":
        #     self.projections = random_projection.SparseRandomProjection()

    def set_hash_value(self, input_vec, item_name):
        # new_vec = self.projections.fit_transform(input_vec)
        new_vec = np.dot(input_vec, self.projections.T)[0]
        bitwise = (new_vec > 0).astype('int')
        hash_value = ''.join(bitwise.astype('str'))
        self.hash_values[item_name] = hash_value
        if hash_value not in self.hash_table.keys():
            self.hash_table[hash_value] = []
        self.hash_table[hash_value].append(item_name)

    def build_hash_table(self, input_matrix, limit=None):
        count = 0
        for item_name in input_matrix.keys():
            vec = np.array([input_matrix[item_name]])
            self.set_hash_value(vec, item_name)
            count += 1
            # set the limit number of items we want to cover
            # from a huge input matrix if it is not None
            if limit is not None and count == limit:
                break

    def fetch_similar_items(self, item_name):
        return self.hash_table[self.hash_values[item_name]]


# Build a number of hash tables to adjust the trade-offer between recall and precision.
# It is worth that multiple tables generalize the high
# dimensional space better and amortize the contribution of bad random vectors.
# So, by building multiple hash tables and providing one item
# we collect all similar items from all hash tables
# (any item appears in any one of tables' similarity fetching can be regard as the similar item)
class LSH:

    def __init__(self, input_matrix, input_dim, num_tables=10, random_type="gau"):
        self.input_matrix = input_matrix
        self.num_tables = num_tables
        self.random_type = random_type
        self.hash_tables = []
        for i in range(self.num_tables):
            ht = HashTable(input_dim)
            ht.build_hash_table(input_matrix)
            self.hash_tables.append(ht)

    # build the similar dictionary: key: the similar items' names, value: similar value
    # by finding all similar items with given item
    def build_similar_dict(self, given_item):
        similar_dic = {}
        given_item_vec = np.array([self.input_matrix[given_item]])
        for ht in self.hash_tables:
            list_sim_items = ht.fetch_similar_items(given_item)
            for item in list_sim_items:
                if item not in similar_dic and item != given_item:
                    compare_product_vec = np.array([self.input_matrix[item]])
                    cos_sim_value = cosine_similarity(given_item_vec, compare_product_vec).item(0)
                    if cos_sim_value > 0:
                        similar_dic[item] = cos_sim_value

        return similar_dic
