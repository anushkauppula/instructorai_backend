-- SQL script to set up Supabase for Majors Exploration AI Assistant

-- 1. Create the academic_explorations table
CREATE TABLE IF NOT EXISTS academic_explorations (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    transcription TEXT NOT NULL,
    recommendations TEXT,
    relevant_programs JSONB,
    detailed_analysis JSONB,
    career_guidance TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 2. Create an index on created_at for faster queries
CREATE INDEX IF NOT EXISTS idx_academic_explorations_created_at 
ON academic_explorations(created_at DESC);

-- 3. Create RLS (Row Level Security) policy if needed
ALTER TABLE academic_explorations ENABLE ROW LEVEL SECURITY;

-- 4. Create a policy to allow all operations (adjust based on your needs)
CREATE POLICY "Allow all operations on academic_explorations" 
ON academic_explorations FOR ALL 
USING (true);

-- 5. Create the storage bucket for audio files (run this in Supabase Dashboard Storage section)
-- Bucket name: academic-audio-files
-- Public: false (for privacy)
-- File size limit: 50MB
-- Allowed MIME types: audio/m4a, audio/mp3, audio/wav, audio/ogg

-- Instructions for creating the storage bucket:
-- 1. Go to your Supabase Dashboard
-- 2. Navigate to Storage section
-- 3. Click "New bucket"
-- 4. Name: academic-audio-files
-- 5. Make it private (not public)
-- 6. Set file size limit to 50MB
-- 7. Add allowed MIME types: audio/m4a, audio/mp3, audio/wav, audio/ogg
-- 8. Click "Create bucket"
