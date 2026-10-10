/*
================================================================================
SCRIPT: inventory_demand_metrics.sql
PURPOSE: Inventory Turnover, Stock Efficiency, and Demand vs Supply Metrics
BUSINESS QUESTIONS ANSWERED:
  1. What is the stock turnover ratio across stores and categories?
  2. Are certain categories overstocked relative to true customer demand?
  3. What is the unfulfilled demand gap (Demand - Units Sold)?
================================================================================
*/

-- 1. Store & Category Inventory Turnover Ratio
-- Business Question: Evaluate how effectively inventory is converted into sales across stores.
SELECT 
    Store_ID,
    Category, 
    SUM(Inventory_Level) AS total_inventory_held,
    SUM(Units_Sold) AS total_units_sold,
    SUM(Demand) AS total_customer_demand,
    -- Inventory Turnover = (Units Sold / Inventory Level) * 100
    ROUND(
        CASE 
            WHEN SUM(Inventory_Level) > 0 
            THEN (CAST(SUM(Units_Sold) AS FLOAT) / SUM(Inventory_Level)) * 100 
            ELSE 0 
        END, 2
    ) AS inventory_turnover_pct,
    -- Stock-to-Demand Coverage Ratio
    ROUND(
        CASE 
            WHEN SUM(Demand) > 0 
            THEN (CAST(SUM(Inventory_Level) AS FLOAT) / SUM(Demand)) 
            ELSE 0 
        END, 2
    ) AS stock_to_demand_ratio
FROM Sales_Database..sales_data
GROUP BY Store_ID, Category
ORDER BY Store_ID, inventory_turnover_pct DESC;

-- 2. Category-Level Aggregate Demand & Lost Sales Gap
-- Business Question: Identify categories suffering from stockouts / lost sales where Demand > Units Sold.
SELECT
    Category,
    SUM(Demand) AS total_unconstrained_demand,
    SUM(Units_Sold) AS total_realized_sales,
    SUM(Demand) - SUM(Units_Sold) AS total_lost_sales_units,
    ROUND(
        CASE 
            WHEN SUM(Demand) > 0 
            THEN ((CAST(SUM(Demand) - SUM(Units_Sold) AS FLOAT)) / SUM(Demand)) * 100 
            ELSE 0 
        END, 2
    ) AS unmet_demand_pct,
    SUM(Inventory_Level) AS total_inventory_held,
    ROUND(
        CASE 
            WHEN SUM(Inventory_Level) > 0 
            THEN (CAST(SUM(Units_Sold) AS FLOAT) / SUM(Inventory_Level)) * 100 
            ELSE 0 
        END, 2
    ) AS category_turnover_pct
FROM Sales_Database..sales_data
GROUP BY Category
ORDER BY total_lost_sales_units DESC;