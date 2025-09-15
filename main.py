import os
import openai
import shutil
import logging
import traceback
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from supabase import create_client
from datetime import datetime
from typing import Optional
from pydub import AudioSegment

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load and validate environment variables
load_dotenv()
required_env_vars = {
    "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
    "SUPABASE_URL": os.getenv("SUPABASE_URL"),
    "SUPABASE_KEY": os.getenv("SUPABASE_KEY")
}

missing_vars = [key for key, value in required_env_vars.items() if not value]
if missing_vars:
    error_msg = f"Missing required environment variables: {', '.join(missing_vars)}"
    logger.error(error_msg)
    raise ValueError(error_msg)

client = openai.OpenAI(api_key=required_env_vars["OPENAI_API_KEY"])

# Initialize Supabase client
try:
    supabase = create_client(required_env_vars["SUPABASE_URL"], required_env_vars["SUPABASE_KEY"])
    supabase.auth.get_user()
    logger.info("Successfully connected to Supabase")
except Exception as e:
    error_msg = f"Failed to initialize Supabase client: {str(e)}\n{traceback.format_exc()}"
    logger.error(error_msg)
    raise ValueError(error_msg)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def is_silent(audio_file_path, silence_threshold=-50.0):
    audio = AudioSegment.from_file(audio_file_path)
    return audio.dBFS < silence_threshold

@app.post("/analyze_sales_call")
async def analyze_sales_call(file: UploadFile = File(...)):
    temp_file_path = "temp_audio.m4a"
    wav_file_path = "temp_audio.wav"  # <-- Initialize here
    bucket_name = "audio-files"

    try:
        if not file.filename.endswith(('.m4a', '.mp3', '.wav', '.ogg')):
            raise HTTPException(
                status_code=400,
                detail="Invalid file format. Please upload an audio file (m4a, mp3, wav, or ogg)."
            )

        # Save uploaded file temporarily
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(f"Successfully saved temporary file: {temp_file_path}")

        # Silence detection
        if is_silent(temp_file_path):
            logger.info("No audio detected in uploaded file")
            return {
                "transcription": "",
                "analysis": "No audio detected"
            }

        # Convert audio to WAV format for consistency
        try:
            audio = AudioSegment.from_file(temp_file_path)
            audio.export(wav_file_path, format="wav")
            logger.info(f"Audio converted to WAV format: {wav_file_path}")
        except Exception as e:
            error_msg = f"Failed to convert audio to WAV format: {str(e)}"
            logger.error(error_msg)
            raise HTTPException(status_code=500, detail=error_msg)

        # Check if the audio is silent
        if is_silent(wav_file_path):
            raise HTTPException(
                status_code=400,
                detail="The audio file is too silent. Please provide a clearer audio file."
            )

        # Generate a unique file name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_extension = os.path.splitext(file.filename)[1]
        file_name = f"audio_{timestamp}{file_extension}"
        storage_path = f"audio-files/{file_name}"

        # Upload to Supabase Storage
        try:
            with open(temp_file_path, "rb") as audio_file:
                response = supabase.storage.from_(bucket_name).upload(
                    path=storage_path,
                    file=audio_file
                )
            logger.info(f"File uploaded to Supabase Storage: {file_name}")
        except Exception as e:
            error_msg = f"Failed to upload to Supabase Storage: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            raise HTTPException(status_code=500, detail=f"Failed to upload recording to storage: {error_msg}")

        # Transcribe using Whisper
        with open(wav_file_path, "rb") as audio_file:
            logger.info("Starting audio transcription with Whisper")
            transcript = client.audio.transcriptions.create(
                file=audio_file,
                model="whisper-1"
            )
            logger.info("Transcription completed successfully")

        transcription_text = transcript.text

        # Generate feedback using GPT-4
        prompt = f"""
You're a supportive and encouraging sales coach.

Please analyze the following sales call and provide friendly, constructive feedback directly to the salesperson. Focus on what they did well, areas they can improve, and give specific, practical tips to help them boost their sales performance.

Your response should be:
- Empathetic and motivating
- Easy to understand
- Actionable and not too formal

Transcript:
\"\"\"
{transcription_text}
\"\"\"
"""

        logger.info("Starting GPT-4 analysis")
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a professional sales coach who gives supportive feedback."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0
        )
        logger.info("GPT-4 analysis completed successfully")

        analysis = response.choices[0].message.content

        # Store data in Supabase database
        logger.info("Storing data in Supabase database")
        data = {
            "transcription": transcription_text,
            "analysis": analysis,
            "created_at": datetime.now().isoformat()
        }

        result = supabase.table("sales_calls").insert(data).execute()
        logger.info("Data stored successfully in database")

        return {
            "transcription": transcription_text,
            "analysis": analysis
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        error_msg = f"Unexpected error during processing: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)
    finally:
        if os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
                logger.info(f"Cleaned up temporary file: {temp_file_path}")
            except Exception as e:
                logger.error(f"Failed to clean up temporary file: {str(e)}")
        if os.path.exists(wav_file_path):
            try:
                os.remove(wav_file_path)
                logger.info(f"Cleaned up temporary WAV file: {wav_file_path}")
            except Exception as e:
                logger.error(f"Failed to clean up temporary WAV file: {str(e)}")
