-- Each query starts with a "-- name:" line so sqlstore.py can find it.
-- The table is `collisions` (one row per police-reported injury collision).
-- ksi = 1 for fatal or serious, 0 for slight.

-- name: severity_by_speed_and_area
-- Share of fatal/serious collisions by speed limit and urban/rural.
-- HAVING drops tiny groups, where a percentage would just be noise.
SELECT speed_limit,
       CASE urban_or_rural_area WHEN '1' THEN 'Urban' WHEN '2' THEN 'Rural' ELSE 'Other' END AS area,
       COUNT(*)                          AS collisions,
       SUM(ksi)                          AS fatal_or_serious,
       ROUND(100.0 * SUM(ksi) / COUNT(*), 1) AS pct_fatal_or_serious
FROM collisions
GROUP BY speed_limit, area
HAVING COUNT(*) >= 200
ORDER BY speed_limit, area;

-- name: year_on_year
-- Collisions per year and change against the previous year, using LAG().
WITH yearly AS (
    SELECT collision_year AS year, COUNT(*) AS collisions, SUM(ksi) AS ksi
    FROM collisions GROUP BY collision_year
)
SELECT year, collisions, ksi,
       ROUND(100.0 * ksi / collisions, 1) AS pct_ksi,
       collisions - LAG(collisions) OVER (ORDER BY year) AS change_in_collisions
FROM yearly
ORDER BY year;

-- name: worst_hours
-- Hours of the day ranked by fatal/serious share (needs >= 1000 collisions).
SELECT hour, collisions, pct_ksi,
       RANK() OVER (ORDER BY pct_ksi DESC) AS severity_rank
FROM (
    SELECT hour, COUNT(*) AS collisions,
           ROUND(100.0 * SUM(ksi) / COUNT(*), 1) AS pct_ksi
    FROM collisions GROUP BY hour HAVING COUNT(*) >= 1000
)
ORDER BY severity_rank
LIMIT 8;

-- name: dark_vs_daylight_by_limit
-- Does darkness change the picture at the same speed limit?
SELECT speed_limit,
       ROUND(100.0 * AVG(CASE WHEN light_conditions = '1' THEN ksi END), 1) AS pct_ksi_daylight,
       ROUND(100.0 * AVG(CASE WHEN light_conditions <> '1' THEN ksi END), 1) AS pct_ksi_dark,
       COUNT(*) AS collisions
FROM collisions
GROUP BY speed_limit
HAVING COUNT(*) >= 1000
ORDER BY speed_limit;
