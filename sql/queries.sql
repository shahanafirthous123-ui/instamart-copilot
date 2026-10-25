-- Q1
-- WHERE filters rows BEFORE grouping; HAVING filters groups AFTER.
SELECT customer_id, SUM(order_amount) AS total
FROM orders
WHERE status = 'completed' AND order_amount > 500
GROUP BY customer_id
HAVING SUM(order_amount) > 8000;

-- Q2
-- Three-table join: revenue by city and category.
SELECT c.city, p.category, ROUND(SUM(o.order_amount), 2) AS revenue
FROM orders o
JOIN customers c ON c.customer_id = o.customer_id
JOIN products  p ON p.product_id  = o.product_id
WHERE o.status = 'completed'
GROUP BY c.city, p.category;

-- Q3
-- Scalar subquery: one value (the average price) used inside WHERE.
SELECT product_id, product_name, price_inr
FROM products
WHERE price_inr > (SELECT AVG(price_inr) FROM products);

-- Q4
-- The status test is in ON, not WHERE. In WHERE it would remove the NULL rows and defeat the LEFT JOIN.
SELECT c.customer_id
FROM customers c
LEFT JOIN orders o ON o.customer_id = c.customer_id AND o.status = 'completed'
WHERE o.order_id IS NULL;

-- Q5
WITH spend AS (
  SELECT customer_id, SUM(order_amount) AS total
  FROM orders WHERE status = 'completed' GROUP BY customer_id
),
avg_spend AS (SELECT AVG(total) AS avg_total FROM spend)
SELECT s.customer_id, s.total, a.avg_total
FROM spend s CROSS JOIN avg_spend a
WHERE s.total > a.avg_total;

-- Q6
WITH spend AS (
  SELECT c.customer_id, c.city, SUM(o.order_amount) AS total_spend
  FROM orders o JOIN customers c ON c.customer_id = o.customer_id
  WHERE o.status = 'completed'
  GROUP BY c.customer_id, c.city
)
SELECT customer_id, city, total_spend,
       RANK()       OVER (PARTITION BY city ORDER BY total_spend DESC) AS rnk,
       DENSE_RANK() OVER (PARTITION BY city ORDER BY total_spend DESC) AS drnk,
       ROW_NUMBER() OVER (PARTITION BY city ORDER BY total_spend DESC) AS rn
FROM spend;

-- Q7
WITH daily AS (
  SELECT substr(order_ts, 1, 10) AS day, SUM(order_amount) AS revenue
  FROM orders WHERE status = 'completed' GROUP BY day
)
SELECT day, revenue, SUM(revenue) OVER (ORDER BY day) AS cumulative_revenue
FROM daily;