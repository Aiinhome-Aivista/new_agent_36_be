-- =====================================================================
-- DevKnow - Multi-Language AI Developer Knowledge Assistant
-- Relational Database Schema (MySQL 8.0+)
-- =====================================================================

CREATE DATABASE IF NOT EXISTS `devknow_db` 
CHARACTER SET utf8mb4 
COLLATE utf8mb4_unicode_ci;

USE `devknow_db`;

-- 1. Knowledge Base Files (Audit & Upload Tracking)
CREATE TABLE IF NOT EXISTS `knowledge_base_files` (
    `id` INT AUTO_INCREMENT PRIMARY KEY,
    `file_name` VARCHAR(255) NOT NULL,
    `file_path` VARCHAR(500) NOT NULL,
    `uploaded_at` DATETIME DEFAULT CURRENT_TIMESTAMP,
    `total_records` INT DEFAULT 0,
    `successful_records` INT DEFAULT 0,
    `failed_records` INT DEFAULT 0,
    `skipped_records` INT DEFAULT 0,
    `languages_detected` VARCHAR(500) DEFAULT '',
    `ingestion_status` VARCHAR(50) DEFAULT 'pending'
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 2. Functions (Single Source of Truth for Organizational Code)
CREATE TABLE IF NOT EXISTS `functions` (
    `id` INT AUTO_INCREMENT PRIMARY KEY,
    `rule_version_name` VARCHAR(255) DEFAULT '',
    `rule_version_description` TEXT,
    `function_name` VARCHAR(255) NOT NULL,
    `function_header` TEXT NOT NULL,
    `function_body` LONGTEXT NOT NULL,
    `programming_language` VARCHAR(100) NOT NULL,
    `framework` VARCHAR(100) DEFAULT '',
    `version` VARCHAR(50) DEFAULT '1.0.0',
    `class_name` VARCHAR(255) DEFAULT '',
    `namespace` VARCHAR(255) DEFAULT '',
    `description` TEXT,
    `additional_metadata` TEXT,
    `knowledge_source` VARCHAR(50) DEFAULT 'excel',
    `source_file_id` INT NULL,
    `created_at` DATETIME DEFAULT CURRENT_TIMESTAMP,
    `updated_at` DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX `idx_func_name` (`function_name`),
    INDEX `idx_func_lang` (`programming_language`),
    INDEX `idx_func_ver` (`version`),
    INDEX `idx_func_class` (`class_name`),
    INDEX `idx_func_source` (`knowledge_source`),
    CONSTRAINT `fk_functions_file` 
        FOREIGN KEY (`source_file_id`) 
        REFERENCES `knowledge_base_files` (`id`) 
        ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 3. LLM Generated Knowledge (Self-Growing Knowledge Base Tracking)
CREATE TABLE IF NOT EXISTS `llm_generated_knowledge` (
    `id` INT AUTO_INCREMENT PRIMARY KEY,
    `function_id` INT NOT NULL,
    `original_query` TEXT NOT NULL,
    `llm_answer` LONGTEXT NOT NULL,
    `user_decision` VARCHAR(50) DEFAULT 'accepted',
    `similarity_score_at_query` FLOAT DEFAULT 0.0,
    `stored_at` DATETIME DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT `fk_llm_function` 
        FOREIGN KEY (`function_id`) 
        REFERENCES `functions` (`id`) 
        ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 4. Ingestion Logs (Row-by-Row Ingestion Audit Trail)
CREATE TABLE IF NOT EXISTS `ingestion_logs` (
    `id` INT AUTO_INCREMENT PRIMARY KEY,
    `file_id` INT NOT NULL,
    `record_identifier` VARCHAR(255) DEFAULT '',
    `status` VARCHAR(50) NOT NULL,
    `error_message` TEXT,
    `created_at` DATETIME DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT `fk_logs_file` 
        FOREIGN KEY (`file_id`) 
        REFERENCES `knowledge_base_files` (`id`) 
        ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 5. Conversations (Chat Sessions)
CREATE TABLE IF NOT EXISTS `conversations` (
    `id` INT AUTO_INCREMENT PRIMARY KEY,
    `session_id` VARCHAR(100) UNIQUE NOT NULL,
    `title` VARCHAR(255) DEFAULT 'New Conversation',
    `created_at` DATETIME DEFAULT CURRENT_TIMESTAMP,
    `updated_at` DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX `idx_session_id` (`session_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 6. Conversation Messages (Chat Turns & Observability)
CREATE TABLE IF NOT EXISTS `conversation_messages` (
    `id` INT AUTO_INCREMENT PRIMARY KEY,
    `conversation_id` INT NOT NULL,
    `role` VARCHAR(20) NOT NULL,
    `message` LONGTEXT NOT NULL,
    `attached_code` LONGTEXT NULL,
    `answer_source` VARCHAR(50) DEFAULT 'none',
    `retrieved_function_ids` TEXT NULL,
    `similarity_score` FLOAT NULL,
    `query_intent` VARCHAR(50) NULL,
    `created_at` DATETIME DEFAULT CURRENT_TIMESTAMP,
    INDEX `idx_conv_id` (`conversation_id`),
    CONSTRAINT `fk_messages_conv` 
        FOREIGN KEY (`conversation_id`) 
        REFERENCES `conversations` (`id`) 
        ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

