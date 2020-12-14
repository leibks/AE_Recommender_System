import os
import sys
path = os.getcwd()
sys.path.append(path)
from src.modules.RMSystemModel import *
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--USER", type=str, default=False, help="the user who is recommended")
parser.add_argument("--NUM_REC", type=int, default=10, help="how many items provided for recommendation")
parser.add_argument("--ALGO", type=str, default="user", help="selected algorithms to use for prediction")
parser.add_argument("--HIGH_RATE", type=float, default=0.9, help="identify rate of determining high value products")
parser.add_argument("--LOW_RATE", type=float, default=0.1, help="identify rate of determining low value products")
parser.add_argument("--ECO", type=str, default="True", help="consider economic factors")
parser.add_argument("--LSH", type=str, default="True", help="whether use the locality sensitive hashing")
parser.add_argument("--REDUCE", type=str, default="False", help="whether reduce the matrix")

args = parser.parse_args()

if __name__ == '__main__':
    file_name = "resource/cleaned_data/Luxury_Beauty_stock.csv"
    user_id = args.USER  # A2HOI48JK8838M
    algo = args.ALGO
    eco = True if args.ECO == "True" else False
    do_lsh = True if args.LSH == "True" else False
    reduce = True if args.REDUCE == "True" else False
    print([user_id, algo, eco, do_lsh, reduce])
    m = RMSystemModel()
    m.set_up_matrix(file_name, algo=algo, reduce=reduce, hash_size=8, num_tables=2, eco=eco)
    # print(m.predict_utility("A2HOI48JK8838M", "B00004U9V2", algo, lsh=False))
    res = m.find_recommended_products(user_id, algo=algo, do_lsh=do_lsh)

