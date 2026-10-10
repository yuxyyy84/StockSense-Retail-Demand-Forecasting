/*
================================================================================
SCRIPT: weather_impact.sql
PURPOSE: Environmental & Meteorological Sales Impact Analytics
BUSINESS QUESTIONS ANSWERED:
  1. How do different weather conditions affect overall revenue and unit volume?
  2. Which product categories dominate under specific weather conditions per region?
  3. What is the seasonal revenue distribution across the retail chain?
================================================================================
*/

-- 1. Sales & Revenue Distribution by Weather Condition
-- Business Question: Quantify how weather extremes impact foot traffic and revenue generation.
SELECT 
    Weather,
    SUM(Units_Sold) AS total_units_sold,
    ROUND(MIN(Revenue), 2) AS min_transaction_revenue,
    ROUND(MAX(Revenue), 2) AS max_transaction_revenue,
    ROUND(AVG(Revenue), 2) AS avg_transaction_revenue,
    ROUND(SUM(Revenue), 2) AS total_revenue,
    ROUND(
        (SUM(Revenue) / (SELECT SUM(Revenue) FROM Sales_Database..sales_data)) * 100, 2
    ) AS revenue_contribution_pct
FROM Sales_Database..sales_data
WHERE Revenue > 0
GROUP BY Weather
ORDER BY total_revenue DESC;

-- 2. Regional Category Leaders by Weather Condition (Window Ranking)
-- Business Question: Identify top-performing category for each combination of weather and region.
WITH regional_weather_sales AS (
    SELECT 
        Weather, 
        Region,
        Category,
        ROUND(SUM(Revenue), 2) AS category_revenue,
        SUM(Units_Sold) AS category_units_sold,
        RANK() OVER (
            PARTITION BY Weather, Region 
            ORDER BY SUM(Revenue) DESC
        ) AS region_rank
    FROM Sales_Database..sales_data
    GROUP BY Weather, Region, Category
)
SELECT 
    Weather,
    Region,
    Category,
    category_revenue,
    category_units_sold
FROM regional_weather_sales
WHERE region_rank = 1
ORDER BY Weather, Region;

-- 3. Combined Seasonality and Weather Performance Matrix
-- Business Question: Identify synergistic seasonal weather combinations that drive maximum sales.
SELECT 
    Seasonality,
    Weather,
    SUM(Units_Sold) AS total_units_sold,
    ROUND(SUM(Revenue), 2) AS total_revenue,
    ROUND(AVG(Demand), 2) AS avg_daily_demand,
    ROUND(
        (SUM(Revenue) / (SELECT SUM(Revenue) FROM Sales_Database..sales_data)) * 100, 2
    ) AS sales_contribution_pct
FROM Sales_Database..sales_data
GROUP BY Seasonality, Weather
ORDER BY Seasonality, total_revenue DESC;