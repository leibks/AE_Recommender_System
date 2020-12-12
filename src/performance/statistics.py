
def calculate_filled_utilities(rated_products, product_num):
    count_size = 0
    for user_id in rated_products:
        count_size += len(rated_products[user_id])
    return count_size / (len(rated_products)) / product_num
