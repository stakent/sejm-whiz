-- PostgreSQL initialization script for sejm-whiz
-- Creates required extensions for vector database functionality

-- Create pgvector extension for vector operations
CREATE EXTENSION IF NOT EXISTS vector;

-- Create uuid-ossp extension for UUID generation
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Grant necessary permissions to the application user
GRANT ALL PRIVILEGES ON DATABASE sejm_whiz TO sejm_whiz_user;
GRANT ALL PRIVILEGES ON SCHEMA public TO sejm_whiz_user;
