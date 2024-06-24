-- task 20
DROP PROCEDURE IF EXISTS ComputeAverageScoreForUser;

DELIMITER //
CREATE PROCEDURE ComputeAverageScoreForUser(IN user_id_param INT)
BEGIN
  DECLARE total_score INT;
  DECLARE total_projects INT;
  DECLARE average_score FLOAT;

  SELECT SUM(score), COUNT(DISTINCT project_id)
  INTO total_score, total_projects
  FROM corrections
  WHERE user_id = user_id_param;

  IF total_projects > 0 THEN
      SET average_score = total_score / total_projects;
      UPDATE users SET average_score = average_score WHERE id = user_id_param;
  ELSE
      UPDATE users SET average_score = 0 WHERE id = user_id_param;
  END IF;
END //
DELIMITER ;