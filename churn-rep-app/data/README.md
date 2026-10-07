# HEINEKEN × AISO Challenge: Dataset

This folder holds about 91,000 real, anonymised orders placed between **January 2017 and August 2018**. It is not HEINEKEN data, but it holds the same behavioural signals. Assume HEINEKEN sells directly to these customers.

**Analysis date ("today"): 31 August 2018.** There are no orders after this date.

## Accounts

Treat each zip code area as one HEINEKEN account, such as a bar, restaurant or shop. All files share the key `account_id`, a text value such as `A01037`, so its leading zeros are kept.

- There are 14,989 accounts.
- About 2,600 accounts have 10 or more orders. Together they place 52% of all orders. They order in a median of 10 different months and buy about 11 product categories each.
- Small accounts often have random gaps between orders. Telling real decline apart from noise is part of the challenge.

## Files at a glance

| File | One row per | Rows | Use it for |
|---|---|---|---|
| `order_lines.csv` | product line in an order | 103,646 | **Start here.** Everything joined in one Excel-ready table |
| `orders.csv` | order | 90,961 | Order status and all timestamps |
| `customers.csv` | order's customer record | 90,961 | Linking orders to accounts and locations |
| `order_items.csv` | product line in an order | 102,942 | Products, prices and freight per order |
| `order_payments.csv` | payment | 95,048 | How each order was paid |
| `order_reviews.csv` | reviewed order | 90,263 | Review scores and written comments |
| `products.csv` | product | 32,951 | Product category, size and weight |
| `geolocation.csv` | zip code area | 19,010 | Map coordinates for each account |

### How the files connect

```
customers ──customer_id── orders ──order_id── order_items ──product_id── products
                            │
                            ├──order_id── order_payments
                            └──order_id── order_reviews

account_id links orders, customers, order_lines and geolocation directly.
```

---

## `order_lines.csv`: the ready-made table

This file already joins orders, items, products, customers and reviews. It opens directly in Excel, Google Sheets or pandas.

- Orders without products, which are mostly cancelled or unavailable orders, appear as one row with the product fields left empty.
- An order with several products has several rows. To count orders, count distinct `order_id` values, not rows.

| Variable | Description |
|---|---|
| `account_id` | Account (zip code area) that placed the order |
| `city` | City of the account |
| `state` | State of the account (two-letter code) |
| `order_id` | Unique order identifier |
| `order_date` | Date the order was placed (YYYY-MM-DD) |
| `order_month` | Month the order was placed (YYYY-MM) |
| `order_status` | Current status: `delivered`, `shipped`, `canceled`, `unavailable`, `invoiced`, `processing`, `created` or `approved` |
| `order_item_id` | Line number of the product within the order (1, 2, 3, …) |
| `product_id` | Unique product identifier |
| `product_category` | Product category in English (`unknown` if missing) |
| `price` | Price of this product line, in local currency units |
| `freight_value` | Delivery cost charged for this product line |
| `delivered_date` | Date the order reached the account (empty if not delivered) |
| `estimated_delivery_date` | Delivery date promised when the order was placed |
| `days_late` | Days between the promised and the actual delivery date. A positive value means late; zero or negative means on time or early |
| `is_late` | `True` if the order was delivered after the promised date, `False` if on time. Empty if the order was never delivered; check `order_status` to see why |
| `review_score` | Score from 1 (very bad) to 5 (very good) that the account gave the order. Empty if there was no review |

---

## `orders.csv`

| Variable | Description |
|---|---|
| `order_id` | Unique order identifier |
| `customer_id` | Customer record for this order (links to `customers.csv`). Each order has its own `customer_id` |
| `account_id` | Account that placed the order |
| `order_status` | Current status: `delivered`, `shipped`, `canceled`, `unavailable`, `invoiced`, `processing`, `created` or `approved` |
| `order_purchase_timestamp` | Date and time the order was placed |
| `order_approved_at` | Date and time the payment was approved |
| `order_delivered_carrier_date` | Date and time the order was handed to the delivery carrier |
| `order_delivered_customer_date` | Date and time the order reached the account |
| `order_estimated_delivery_date` | Delivery date promised when the order was placed |

## `customers.csv`

| Variable | Description |
|---|---|
| `customer_id` | Customer record for one order (links to `orders.csv`) |
| `customer_unique_id` | Identifies the individual buyer. One account can have many buyers |
| `account_id` | Account (zip code area) this buyer belongs to |
| `customer_zip_code_prefix` | First five digits of the zip code, stored as text |
| `customer_city` | City |
| `customer_state` | State (two-letter code) |

## `order_items.csv`

| Variable | Description |
|---|---|
| `order_id` | Order the product belongs to |
| `order_item_id` | Line number of the product within the order (1, 2, 3, …) |
| `product_id` | Product ordered (links to `products.csv`) |
| `price` | Price of this product line, in local currency units |
| `freight_value` | Delivery cost charged for this product line |

## `order_payments.csv`

An order can be paid in more than one payment, for example a card plus a voucher.

| Variable | Description |
|---|---|
| `order_id` | Order the payment belongs to |
| `payment_sequential` | Number of the payment within the order (1, 2, …) |
| `payment_type` | Method: `credit_card`, `boleto` (bank invoice), `voucher`, `debit_card` or `not_defined` |
| `payment_installments` | Number of instalments the payment was split into |
| `payment_value` | Amount paid in this payment |

## `order_reviews.csv`

Each order has at most one review. Comments are in **Portuguese**, and about 40% of reviews include text; any LLM can translate them. Some comments span several lines, so open the file with a CSV reader rather than counting lines.

| Variable | Description |
|---|---|
| `review_id` | Review identifier |
| `order_id` | Order that was reviewed |
| `review_score` | Score from 1 (very bad) to 5 (very good) |
| `review_comment_title` | Short title of the review (often empty) |
| `review_comment_message` | Written comment (often empty) |
| `review_creation_date` | Date the review request was sent, shortly after delivery |
| `review_answer_timestamp` | Date and time the review was submitted |

## `products.csv`

| Variable | Description |
|---|---|
| `product_id` | Unique product identifier |
| `product_category` | Product category in English (`unknown` if missing) |
| `product_name_length` | Number of characters in the product name |
| `product_description_length` | Number of characters in the product description |
| `product_photos_qty` | Number of product photos |
| `product_weight_g` | Weight in grams |
| `product_length_cm` | Length in cm |
| `product_height_cm` | Height in cm |
| `product_width_cm` | Width in cm |

## `geolocation.csv`

There is one point per zip code area, at the centre of that area. A few accounts (158) have no coordinates.

| Variable | Description |
|---|---|
| `account_id` | Account (zip code area); links to every other file |
| `zip_code_prefix` | First five digits of the zip code, stored as text |
| `lat` | Latitude |
| `lng` | Longitude |
| `city` | City |
| `state` | State (two-letter code) |

---

## Good to know

- **Money:** all amounts are in local currency units. Treat them as relative values.
- **Growth:** order volume grows strongly during 2017, so be careful when comparing periods.
- **Zip codes:** keep `account_id` and the zip code columns as text. If Excel or pandas turns them into numbers, the leading zeros are lost and the joins break.
