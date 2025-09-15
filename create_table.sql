-- Create the sales_calls table
CREATE TABLE sales_calls (
    id BIGSERIAL PRIMARY KEY,
    transcription TEXT NOT NULL,
    analysis TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
); 