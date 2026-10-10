/*
================================================================================
SCRIPT: epidemic_effect_demand_sales.sql
PURPOSE: Exogenous Shock / Pandemic Impact Analysis on Demand and Sales
BUSINESS QUESTIONS ANSWERED:
  1. Did epidemic periods cause significant deviations in daily customer demand and sales volume?
  2. How did realized revenue and transaction frequency shift during epidemic periods?
  3. Which categories demonstrated resilience vs vulnerability during health disruptions?
================================================================================
*/

-- 1. Macro Impact of Epidemic on Demand, Sales Volume, and Order Activity
-- Business Question: Measure average sales, demand, and transaction counts across epidemic vs non-epidemic periods.
SELECT
    Epidemic,
    COUNT(*) AS total_daily_records,
    ROUND(AVG(Demand), 2) AS avg_daily_demand,
    ROUND(AVG(Units_Sold), 2) AS avg_daily_units_sold,
    SUM(Units_Sold) AS total_units_sold,
    ROUND(AVG(Revenue), 2) AS avg_revenue_per_record,
    ROUND(SUM(Revenue), 2) AS total_realized_revenue
FROM Sales_Database..sales_data
GROUP BY Epidemic;

-- 2. Category-Specific Epidemic Resilience Analysis
-- Business Question: Evaluate how different product categories responded to epidemic conditions.
SELECT
    Category,
    Epidemic,
    ROUND(AVG(Demand), 2) AS avg_demand,
    ROUND(AVG(Units_Sold), 2) AS avg_units_sold,
    ROUND(SUM(Revenue), 2) AS total_revenue,
    ROUND(AVG(Price), 2) AS avg_price
FROM Sales_Database..sales_data
GROUP BY Category, Epidemic
ORDER BY Category, Epidemic;