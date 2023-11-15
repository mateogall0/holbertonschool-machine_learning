-- task 15
CREATE TEMPORARY TABLE temp_band_fans AS
SELECT origin, COUNT(*) AS nb_fans
FROM metal_bands
GROUP BY origin;

SELECT origin, nb_fans
FROM temp_band_fans
ORDER BY nb_fans DESC;