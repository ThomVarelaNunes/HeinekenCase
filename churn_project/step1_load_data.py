"""
STEP 1 -- Load the raw files into four tidy tables.

    orders    one row per order    (account, date, status, delivery dates, buyer)
    items     one row per product  (order, account, date, category, price, freight)
    reviews   one row per review   (order, account, score, when it was answered, comment)
    payments  one row per payment  (order, account, type, instalments, amount)

Every table gets `account_id` and the order date `order_date`, so later steps can
filter "everything before cutoff X" with one line.

Run on its own to see a short summary:   python step1_load_data.py
"""
import pandas as pd
import config


def load_tables(data_dir=config.DATA_DIR):
    keep_text = {"account_id": str}  # keep leading zeros in account ids

    orders = pd.read_csv(
        data_dir + "orders.csv", dtype=keep_text,
        parse_dates=["order_purchase_timestamp", "order_delivered_customer_date",
                     "order_estimated_delivery_date"])
    orders["order_date"] = orders.order_purchase_timestamp.dt.normalize()

    # which individual buyer placed the order (one account = many buyers)
    customers = pd.read_csv(data_dir + "customers.csv", dtype=keep_text)
    orders = orders.merge(customers[["customer_id", "customer_unique_id", "customer_state"]],
                          on="customer_id", how="left")

    order_keys = orders[["order_id", "account_id", "order_date"]]

    items = pd.read_csv(data_dir + "order_items.csv")
    products = pd.read_csv(data_dir + "products.csv", usecols=["product_id", "product_category"])
    items = (items.merge(products, on="product_id", how="left")
                  .merge(order_keys, on="order_id"))
    items["product_category"] = items.product_category.fillna("unknown")

    reviews = pd.read_csv(data_dir + "order_reviews.csv", parse_dates=["review_answer_timestamp"])
    reviews = reviews.merge(order_keys, on="order_id")

    payments = pd.read_csv(data_dir + "order_payments.csv").merge(order_keys, on="order_id")

    return {"orders": orders, "items": items, "reviews": reviews, "payments": payments}


if __name__ == "__main__":
    t = load_tables()
    o = t["orders"]
    print(f"orders:   {len(o):>7,}  from {o.order_date.min().date()} to {o.order_date.max().date()}")
    print(f"accounts: {o.account_id.nunique():>7,}")
    print(f"items:    {len(t['items']):>7,}")
    print(f"reviews:  {len(t['reviews']):>7,}")
    print(f"payments: {len(t['payments']):>7,}")
