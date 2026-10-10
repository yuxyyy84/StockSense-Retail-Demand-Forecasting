/*
================================================================================
SCRIPT: Sales_revenue_metrics.sql
PURPOSE: Comprehensive Sales & Revenue Performance Analytics
BUSINESS QUESTIONS ANSWERED:
  1. Which individual products drive the highest sales volume vs. revenue?
  2. What is the category-level contribution to total revenue and volume?
  3. How does store-level revenue compare across geographical footprints?
  4. What is the total discount exposure relative to gross realized revenue?
  5. What is the average unit price realized per category?
================================================================================
*/

-- 1. Product-Level Sales Distribution & Revenue Drivers
-- Business Question: Identify top-selling SKUs by volume vs high-value revenue drivers.
SELECT 
    Product_ID,
    ROUND(SUM(Units_Sold), 2) AS total_units_sold,
    ROUND(AVG(Price), 2) AS avg_selling_price,
    ROUND(SUM(Revenue), 2) AS total_revenue,
    ROUND((SUM(Revenue) / (SELECT SUM(Revenue) FROM Sales_Database..sales_data)) * 100, 2) AS revenue_contribution_pct
FROM Sales_Database..sales_data
GROUP BY Product_ID
ORDER BY total_revenue DESC;

-- 2. Category-Level Performance Breakdown
-- Business Question: Evaluate revenue concentration and SKU breadth per product category.
SELECT 
    Category,
    COUNT(DISTINCT Product_ID) AS unique_sku_count,
    ROUND(SUM(Units_Sold), 2) AS total_units_sold,
    ROUND(SUM(Revenue), 2) AS total_revenue,
    ROUND(AVG(Price), 2) AS avg_category_price,
    ROUND((SUM(Revenue) / (SELECT SUM(Revenue) FROM Sales_Database..sales_data)) * 100, 2) AS revenue_share_pct
FROM Sales_Database..sales_data
GROUP BY Category
ORDER BY total_revenue DESC;

-- 3. Store-Level Revenue & Sales Benchmark
-- Business Question: Compare regional store performance to identify top & bottom retail locations.
SELECT 
    Store_ID,
    Region,
    ROUND(SUM(Units_Sold), 2) AS total_units_sold,
    ROUND(SUM(Revenue), 2) AS total_revenue,
    ROUND(AVG(Revenue), 2) AS avg_daily_store_revenue
FROM Sales_Database..sales_data
GROUP BY Store_ID, Region
ORDER BY total_revenue DESC;

-- 4. Aggregate Revenue, Volume & Discount Summary
-- Business Question: Measure total business revenue, units sold, and aggregate discount percentage.
SELECT 
    ROUND(SUM(Revenue), 2) AS total_gross_revenue,
    SUM(Units_Sold) AS total_units_sold,
    ROUND(SUM(Discount), 2) AS total_discount_percentage_points,
    ROUND(AVG(Discount), 2) AS avg_discount_rate_pct
FROM Sales_Database..sales_data;

-- 5. Category Pricing & Revenue Elasticity Summary
-- Business Question: Determine the pricing hierarchy across categories to guide merchandising.
SELECT 
    Category,
    ROUND(MIN(Price), 2) AS min_price,
    ROUND(MAX(Price), 2) AS max_price,
    ROUND(AVG(Price), 2) AS avg_price,
    ROUND(SUM(Revenue), 2) AS total_revenue
FROM Sales_Database..sales_data
GROUP BY Category
ORDER BY avg_price DESC;
