/*
========================================================================================================
COMPREHENSIVE RETAIL STORE & INVENTORY SQL ANALYSIS
Database: Sales_Database
Table: sales_data

MODULE OVERVIEW:
  Section 1: Sales and Revenue Performance Metrics
  Section 2: Inventory Turnover, Stock Coverage & Demand Analysis
  Section 3: Order Replenishment and Fulfillment Analytics
  Section 4: Seasonal, Meteorological & Environmental Effects
  Section 5: Exogenous Shocks: Epidemic Impact Assessment
  Section 6: Temporal, Cumulative & Rolling Trend Analytics
  Section 7: Category & Geographic Part-to-Whole Breakdown (Window Functions)
  Section 8: Dynamic Product Performance Segmentation
========================================================================================================
*/

--------------------------------------------------------------------------------------------------------
-- SECTION 1: SALES AND REVENUE METRICS
--------------------------------------------------------------------------------------------------------

-- 1.1 Product-level performance ranked by revenue
SELECT 
    Product_ID,
    ROUND(SUM(Units_Sold), 2) AS total_units_sold,
    ROUND(AVG(Price), 2) AS avg_selling_price,
    ROUND(SUM(Revenue), 2) AS total_revenue
FROM Sales_Database..sales_data
GROUP BY Product_ID
ORDER BY total_revenue DESC;

-- 1.2 Category-level sales and revenue breakdown
SELECT 
    Category,
    COUNT(DISTINCT Product_ID) AS unique_skus,
    ROUND(SUM(Units_Sold), 2) AS total_units_sold,
    ROUND(SUM(Revenue), 2) AS total_revenue,
    ROUND(AVG(Price), 2) AS avg_category_price
FROM Sales_Database..sales_data
GROUP BY Category
ORDER BY total_revenue DESC;

-- 1.3 Store-level volume and revenue comparison
SELECT 
    Store_ID,
    Region,
    ROUND(SUM(Units_Sold), 2) AS total_units_sold,
    ROUND(SUM(Revenue), 2) AS total_revenue
FROM Sales_Database..sales_data
GROUP BY Store_ID, Region
ORDER BY total_revenue DESC;

-- 1.4 Global business aggregate: Revenue, Units Sold, and Discounts
SELECT 
    ROUND(SUM(Revenue), 2) AS total_gross_revenue,
    SUM(Units_Sold) AS total_units_sold,
    ROUND(SUM(Discount), 2) AS total_discount_points,
    ROUND(AVG(Discount), 2) AS avg_discount_rate
FROM Sales_Database..sales_data;

--------------------------------------------------------------------------------------------------------
-- SECTION 2: INVENTORY & DEMAND METRICS
--------------------------------------------------------------------------------------------------------

-- 2.1 Store and Category Inventory Turnover
SELECT 
    Store_ID,
    Category, 
    SUM(Inventory_Level) AS total_inventory_held,
    SUM(Units_Sold) AS total_units_sold,
    SUM(Demand) AS total_demand,
    ROUND(
        CASE 
            WHEN SUM(Inventory_Level) > 0 
            THEN (CAST(SUM(Units_Sold) AS FLOAT) / SUM(Inventory_Level)) * 100 
            ELSE 0 
        END, 2
    ) AS inventory_turnover_pct
FROM Sales_Database..sales_data
GROUP BY Store_ID, Category
ORDER BY Store_ID, inventory_turnover_pct DESC;

-- 2.2 Unmet Demand / Stockout Gap by Category
SELECT
    Category,
    SUM(Demand) AS total_customer_demand,
    SUM(Units_Sold) AS total_realized_sales,
    SUM(Demand) - SUM(Units_Sold) AS unmet_demand_units,
    ROUND(
        CASE 
            WHEN SUM(Demand) > 0 
            THEN ((CAST(SUM(Demand) - SUM(Units_Sold) AS FLOAT)) / SUM(Demand)) * 100 
            ELSE 0 
        END, 2
    ) AS lost_sales_pct
FROM Sales_Database..sales_data
GROUP BY Category
ORDER BY unmet_demand_units DESC;

--------------------------------------------------------------------------------------------------------
-- SECTION 3: ORDER REPLENISHMENT & FULFILLMENT
--------------------------------------------------------------------------------------------------------

-- 3.1 Fulfillment rate by Region and Category
SELECT 
    Region,
    Category,
    SUM(Units_Sold) AS total_units_sold,
    SUM(Units_Ordered) AS total_units_ordered,	
    ROUND(
        CASE 
            WHEN SUM(Units_Ordered) > 0 
            THEN (CAST(SUM(Units_Sold) AS FLOAT) / SUM(Units_Ordered)) * 100 
            ELSE NULL 
        END, 2
    ) AS fulfillment_rate_pct
FROM Sales_Database..sales_data
GROUP BY Region, Category
ORDER BY Region, fulfillment_rate_pct DESC;

--------------------------------------------------------------------------------------------------------
-- SECTION 4: SEASONAL & WEATHER IMPACT
--------------------------------------------------------------------------------------------------------

-- 4.1 Weather impact on sales and revenue contribution
SELECT 
    Weather,
    SUM(Units_Sold) AS total_units_sold,
    ROUND(SUM(Revenue), 2) AS total_revenue,
    ROUND(
        (SUM(Revenue) / (SELECT SUM(Revenue) FROM Sales_Database..sales_data)) * 100, 2
    ) AS revenue_contribution_pct
FROM Sales_Database..sales_data
GROUP BY Weather
ORDER BY total_revenue DESC;

-- 4.2 Seasonality vs Weather Matrix
SELECT 
    Seasonality,
    Weather,
    SUM(Units_Sold) AS total_units_sold,
    ROUND(SUM(Revenue), 2) AS total_revenue,
    ROUND(AVG(Demand), 2) AS avg_daily_demand
FROM Sales_Database..sales_data
GROUP BY Seasonality, Weather
ORDER BY Seasonality, total_revenue DESC;

--------------------------------------------------------------------------------------------------------
-- SECTION 5: EPIDEMIC IMPACT ASSESSMENT
--------------------------------------------------------------------------------------------------------

SELECT
    Epidemic,
    COUNT(*) AS total_daily_records,
    ROUND(AVG(Demand), 2) AS avg_daily_demand,
    ROUND(AVG(Units_Sold), 2) AS avg_daily_units_sold,
    ROUND(SUM(Revenue), 2) AS total_realized_revenue
FROM Sales_Database..sales_data
GROUP BY Epidemic;

--------------------------------------------------------------------------------------------------------
-- SECTION 6: TIME-SERIES & CUMULATIVE SALES ANALYTICS
--------------------------------------------------------------------------------------------------------

-- 6.1 Monthly sales progression and running total
WITH monthly_metrics AS (
    SELECT 
        DATETRUNC(MONTH, Date) AS sales_month,
        SUM(Units_Sold) AS monthly_units_sold,
        ROUND(SUM(Revenue), 2) AS monthly_revenue,
        ROUND(AVG(Price), 2) AS avg_price
    FROM Sales_Database..sales_data
    GROUP BY DATETRUNC(MONTH, Date)
)
SELECT
    sales_month,
    monthly_units_sold,
    ROUND(SUM(monthly_units_sold) OVER (ORDER BY sales_month), 2) AS cumulative_units_sold,
    monthly_revenue,
    ROUND(SUM(monthly_revenue) OVER (ORDER BY sales_month), 2) AS cumulative_revenue,
    avg_price,
    ROUND(AVG(avg_price) OVER (ORDER BY sales_month ROWS BETWEEN 2 PRECEDING AND CURRENT ROW), 2) AS rolling_3mo_avg_price
FROM monthly_metrics
ORDER BY sales_month;

--------------------------------------------------------------------------------------------------------
-- SECTION 7: CATEGORY & REGIONAL PART-TO-WHOLE (WINDOW FUNCTIONS)
--------------------------------------------------------------------------------------------------------

-- 7.1 Category contribution to global business
SELECT 
    Category,
    ROUND(SUM(Revenue), 2) AS total_sales,
    SUM(Units_Sold) AS total_units_sold,
    ROUND((SUM(Revenue) / (SELECT SUM(Revenue) FROM Sales_Database..sales_data)) * 100, 2) AS revenue_share_pct,
    ROUND((CAST(SUM(Units_Sold) AS FLOAT) / (SELECT SUM(Units_Sold) FROM Sales_Database..sales_data)) * 100, 2) AS unit_share_pct
FROM Sales_Database..sales_data
GROUP BY Category
ORDER BY revenue_share_pct DESC;

-- 7.2 Top 3 Categories per Geographic Region (Rank Window Function)
WITH ranked_regional_categories AS (
    SELECT 
        Region,
        Category,
        ROUND(SUM(Revenue), 2) AS category_revenue,
        SUM(Units_Sold) AS category_units_sold,
        RANK() OVER (
            PARTITION BY Region 
            ORDER BY SUM(Revenue) DESC
        ) AS category_rank
    FROM Sales_Database..sales_data
    GROUP BY Region, Category
)
SELECT 
    Region,
    Category,
    category_revenue,
    category_units_sold,
    category_rank
FROM ranked_regional_categories
WHERE category_rank <= 3
ORDER BY Region, category_rank;

--------------------------------------------------------------------------------------------------------
-- SECTION 8: DYNAMIC PRODUCT PERFORMANCE SEGMENTATION
--------------------------------------------------------------------------------------------------------

-- 8.1 Monthly Product Demand Segmentation relative to monthly benchmark
WITH monthly_product_demand AS (
    SELECT 
        DATETRUNC(MONTH, Date) AS sales_month,
        Product_ID,
        SUM(Demand) AS total_monthly_demand,
        SUM(Units_Sold) AS total_monthly_units_sold
    FROM Sales_Database..sales_data
    GROUP BY DATETRUNC(MONTH, Date), Product_ID
),
monthly_benchmarks AS (
    SELECT 
        sales_month,
        AVG(total_monthly_demand) AS avg_monthly_product_demand
    FROM monthly_product_demand
    GROUP BY sales_month
)
SELECT 
    mpd.sales_month,
    mpd.Product_ID,
    mpd.total_monthly_demand,
    ROUND(mb.avg_monthly_product_demand, 2) AS market_benchmark_demand,
    CASE
        WHEN mpd.total_monthly_demand > mb.avg_monthly_product_demand * 1.15 THEN 'High Demand Star'
        WHEN mpd.total_monthly_demand < mb.avg_monthly_product_demand * 0.85 THEN 'Low Demand Underperformer'
        ELSE 'Core Performer'
    END AS product_demand_segment
FROM monthly_product_demand mpd
JOIN monthly_benchmarks mb 
    ON mpd.sales_month = mb.sales_month
ORDER BY mpd.sales_month, mpd.Product_ID;
