/*
================================================================================
SCRIPT: order_fulfillment_metrics.sql
PURPOSE: Order Fulfillment and Replenishment Efficiency Analytics
BUSINESS QUESTIONS ANSWERED:
  1. What is the replenishment fulfillment rate across regional operations?
  2. How do orders placed compare against realized sales volume?
  3. Are there supply chain bottlenecks in specific product categories?
================================================================================
*/

-- 1. Regional & Category Fulfillment Rate
-- Business Question: Measure the ratio of sold units to replenishment orders placed.
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

-- 2. Category Level Restocking Alignment
-- Business Question: Evaluate restocking efficiency for active sales periods.
SELECT
    Category,
    SUM(Units_Sold) AS total_units_sold,
    SUM(Units_Ordered) AS total_units_ordered,
    SUM(Units_Ordered) - SUM(Units_Sold) AS net_inventory_replenishment_buffer,
    ROUND(
        CASE 
            WHEN SUM(Units_Ordered) > 0 
            THEN (CAST(SUM(Units_Sold) AS FLOAT) / CAST(SUM(Units_Ordered) AS FLOAT)) * 100 
            ELSE 0 
        END, 2
    ) AS overall_fulfillment_rate_pct
FROM Sales_Database..sales_data
WHERE Units_Sold > 0 OR Units_Ordered > 0
GROUP BY Category
ORDER BY overall_fulfillment_rate_pct DESC;
