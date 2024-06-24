-- task 19
DROP PROCEDURE IF EXISTS AddBonus;

DELIMITER //
CREATE PROCEDURE AddBonus(IN user_id_param INT, IN project_name_param VARCHAR(255), IN score_param INT)
BEGIN
    DECLARE project_id INT;

    IF EXISTS (SELECT id FROM projects WHERE name = project_name_param) THEN
        SELECT id INTO project_id FROM projects WHERE name = project_name_param;
    ELSE
        INSERT INTO projects (name) VALUES (project_name_param);
        SET project_id = LAST_INSERT_ID();
    END IF;

    INSERT INTO corrections (user_id, project_id, score) VALUES (user_id_param, project_id, score_param);
END //
DELIMITER ;