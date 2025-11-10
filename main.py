import os
import openai
import aiohttp
import json
import logging
import traceback
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from supabase import create_client
from datetime import datetime
from typing import Optional
from pydub import AudioSegment
from pydantic import BaseModel
import shutil

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
    "SUPABASE_KEY": os.getenv("SUPABASE_KEY"),
    "PINECONE_API_KEY": os.getenv("PINECONE_API_KEY")
}

# Define Pinecone host here so Pylance sees it and it's available at runtime
PINECONE_INDEX_HOST = os.getenv("PINECONE_INDEX_HOST") or "https://salesassistai-98b2rdq.svc.aped-4627-b74a.pinecone.io"

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

# Pydantic models for request bodies
class UserDetailsRequest(BaseModel):
    user_id: str
    first_name: str
    last_name: str
    phone_number: str
    email: str

def is_silent(audio_file_path, silence_threshold=-50.0):
    audio = AudioSegment.from_file(audio_file_path)
    return audio.dBFS < silence_threshold

@app.post("/analyze_sales_call")
async def analyze_sales_call(file: UploadFile = File(...), user_id: Optional[str] = None):
    temp_file_path = "temp_audio.m4a"
    wav_file_path = "temp_audio.wav"  # <-- Initialize here
    bucket_name = "audio-files"

    try:
        # Log request details for debugging
        logger.info(f"Received request - File: {file.filename}, Content-Type: {file.content_type}, Size: {file.size if hasattr(file, 'size') else 'unknown'}")
        
        # Log user_id if provided
        if user_id:
            logger.info(f"Received request with user_id: {user_id}")
        else:
            logger.warning("No user_id provided in request")
            
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
            "user_id": user_id,
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

@app.post("/user_details")
async def save_user_details(user_data: UserDetailsRequest):
    """Save user details to the user_details table in Supabase."""
    try:
        logger.info(f"Received request to save user details for user_id: {user_data.user_id}")
        
        # Prepare data for insertion
        data = {
            "user_id": user_data.user_id,
            "first_name": user_data.first_name,
            "last_name": user_data.last_name,
            "phone_number": user_data.phone_number,
            "email": user_data.email,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
        
        # Insert into Supabase
        result = supabase.table("user_details").insert(data).execute()
        logger.info(f"User details saved successfully for user_id: {user_data.user_id}")
        
        return {
            "message": "User details saved successfully",
            "user_id": user_data.user_id
        }
    except Exception as e:
        error_msg = f"Error saving user details: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        
        # Check if it's a unique constraint violation
        if "duplicate key" in str(e).lower() or "unique constraint" in str(e).lower():
            raise HTTPException(
                status_code=409,
                detail="User with this user_id or email already exists"
            )
        
        raise HTTPException(status_code=500, detail=f"Error saving user details: {str(e)}")

@app.get("/user_details/{user_id}")
async def get_user_details(user_id: str):
    """Get user details by user_id from Supabase."""
    try:
        logger.info(f"Fetching user details for user_id: {user_id}")
        
        # Query Supabase for user details
        response = supabase.table("user_details").select("*").eq("user_id", user_id).execute()
        
        if response.data and len(response.data) > 0:
            user = response.data[0]
            logger.info(f"User details found for user_id: {user_id}")
            return {
                "first_name": user["first_name"],
                "last_name": user["last_name"],
                "phone_number": user["phone_number"],
                "email": user["email"]
            }
        else:
            logger.warning(f"User not found for user_id: {user_id}")
            raise HTTPException(status_code=404, detail="User not found")
    except HTTPException as he:
        raise he
    except Exception as e:
        error_msg = f"Error fetching user details: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=f"Error fetching user details: {str(e)}")

@app.post("/user_details/update")
async def update_user_details(request: UserDetailsRequest):
    """Update user details in Supabase."""
    try:
        logger.info(f"Updating user details for user_id: {request.user_id}")
        
        # Prepare data for update
        data = {
            "first_name": request.first_name,
            "last_name": request.last_name,
            "phone_number": request.phone_number,
            "email": request.email,
            "updated_at": datetime.now().isoformat()
        }
        
        # Update user details in Supabase
        response = supabase.table("user_details").update(data).eq("user_id", request.user_id).execute()
        
        if response.data and len(response.data) > 0:
            logger.info(f"User details updated successfully for user_id: {request.user_id}")
            return {
                "message": "User details updated successfully",
                "status": "success"
            }
        else:
            logger.warning(f"User not found for update, user_id: {request.user_id}")
            raise HTTPException(status_code=404, detail="User not found")
    except HTTPException as he:
        raise he
    except Exception as e:
        error_msg = f"Error updating user details: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        
        # Check if it's a unique constraint violation (for email)
        if "duplicate key" in str(e).lower() or "unique constraint" in str(e).lower():
            raise HTTPException(
                status_code=409,
                detail="Email already exists for another user"
            )
        
        raise HTTPException(status_code=500, detail=f"Error updating user details: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.get("/transcript/{call_id}")
async def get_transcript(call_id: str):
    """Fetch transcript by call_id."""
    try:
        response = supabase.table("sales_calls").select("transcription").eq("id", call_id).execute()
        transcript = response.data[0]["transcription"] if response.data else ""
        return {"transcript": transcript}
    except Exception as e:
        logger.error(f"Error fetching transcript: {e}")
        raise HTTPException(status_code=500, detail="Error fetching transcript")

@app.post("/context")
async def retrieve_context(query: str, top_k: int = 5):
    """Retrieve top-K passages from Pinecone."""
    try:
        embedding_resp = await client.embeddings.create(
            input=query,
            model="text-embedding-3-small"
        )
        embedding = embedding_resp.data[0].embedding

        headers = {
            "Api-Key": required_env_vars["PINECONE_API_KEY"],
            "Content-Type": "application/json"
        }
        payload = {
            "vector": embedding,
            "topK": top_k,
            "includeMetadata": True
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{PINECONE_INDEX_HOST}/query", headers=headers, data=json.dumps(payload)) as resp:
                if resp.status != 200:
                    logger.error(f"Pinecone query failed: {resp.status} {await resp.text()}")
                    return {"passages": []}
                result = await resp.json()
                passages = []
                for match in result.get("matches", []):
                    meta = match.get("metadata", {})
                    passage = meta.get("text") or meta.get("passage") or ""
                    if passage:
                        passages.append(passage)
                return {"passages": passages}
    except Exception as e:
        logger.error(f"Error retrieving context: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving context")

@app.post("/generate_tips")
async def generate_tips(call_id: str):
    """Generate summary and tips for a sales call."""
    try:
        # Fetch transcript (get_transcript returns {"transcript": "..."} via the endpoint)
        transcript_resp = await get_transcript(call_id)
        transcript = transcript_resp.get("transcript") if isinstance(transcript_resp, dict) else transcript_resp
        if not transcript:
            raise HTTPException(status_code=404, detail="Transcript not found")

        # Retrieve relevant context from Pinecone
        context = await retrieve_context(transcript, top_k=5)
        passages = context.get("passages", []) if isinstance(context, dict) else context or []

        # Generate summary and tips
        summary, tips = await generate_summary_and_tips(transcript, passages)

        # Save results to Supabase
        await save_results(call_id, summary, tips)

        return {"summary": summary, "tips": tips}
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error generating tips: {e}")
        raise HTTPException(status_code=500, detail="Error generating tips")


async def generate_summary_and_tips(transcript_text, passages):
    """Generate summary and tips using GPT-4."""
    try:
        logger.info("Connecting to Pinecone index")
        logger.info("Generating vector embedding for transcript")
        embedding = await get_embedding(transcript_text)
        logger.info("Vector embedding generated")
        logger.info(f"Transcript embedding: {embedding[:5]}...")  # Log first 5 values

        logger.info("Retrieving top 3 passages from Pinecone")
        # If passages were provided, use them; otherwise query Pinecone
        if not passages:
            passages = await query_pinecone(embedding)
        passages_text = "\n\n".join(passages)
        logger.info(f"Retrieved {len(passages)} passages from Pinecone")
        for i, passage in enumerate(passages, 1):
            logger.info(f"Pinecone Passage {i}: {passage[:100]}...")

        logger.info("Generating summary and tips with GPT-4o")
        prompt = f"""
You are a supportive and encouraging sales coach.

Below is a transcript of a sales call. Also included are relevant passages from top sales books and playbooks.

Please generate:
- A concise summary of the sales call.
- Actionable, friendly tips for the salesperson, referencing both the transcript and the passages.

Transcript:
\"\"\"{transcript_text}\"\"\"

Relevant Passages:
\"\"\"{passages_text}\"\"\"

Format your response as:

Summary:
<your summary here>

Tips:
<your tips here>
"""

        logger.info("Starting GPT-4 analysis for summary and tips")
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a professional sales coach who gives supportive feedback."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0
        )
        logger.info("GPT-4 analysis for summary and tips completed successfully")

        result = response.choices[0].message.content

        # Split result into summary and tips (robust parsing)
        if "Tips:" in result and "Summary:" in result:
            try:
                parts = result.split("Tips:")
                summary = parts[0].replace("Summary:", "").strip()
                tips = parts[1].strip()
            except Exception:
                summary = result.strip()
                tips = ""
        else:
            summary = result.strip()
            tips = ""

        # Log Pinecone retrieval summary
        logger.info(f"Pinecone returned {len(passages)} passages for this transcript")
        for i, p in enumerate(passages, start=1):
            logger.info(f"Pinecone passage {i} preview: {p[:120]}...")

        return summary, tips
    except Exception as e:
        logger.error(f"Error in generate_summary_and_tips: {e}")
        raise HTTPException(status_code=500, detail="Error generating summary and tips")

async def get_embedding(text):
    """Get embedding for the text using OpenAI API."""
    try:
        response = await client.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        raise HTTPException(status_code=500, detail="Error generating embedding")

async def query_pinecone(embedding, top_k=3):
    """Query Pinecone with the embedding."""
    try:
        headers = {
            "Api-Key": required_env_vars["PINECONE_API_KEY"],
            "Content-Type": "application/json"
        }
        payload = {
            "vector": embedding,
            "topK": 3,
            "includeMetadata": True
        }
        logger.info(f"Pinecone query payload: {payload}")
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{PINECONE_INDEX_HOST}/query", headers=headers, data=json.dumps(payload)) as resp:
                if resp.status != 200:
                    logger.error(f"Pinecone query failed: {resp.status} {await resp.text()}")
                    return []
                result = await resp.json()
                passages = []
                for match in result.get("matches", []):
                    meta = match.get("metadata", {})
                    passage = meta.get("text") or meta.get("passage") or ""
                    if passage:
                        passages.append(passage)
                return passages
    except Exception as e:
        logger.error(f"Error querying Pinecone: {e}")
        raise HTTPException(status_code=500, detail="Error querying Pinecone")

async def save_results(call_id, summary, tips):
    """Save the generated summary and tips to Supabase."""
    try:
        data = {
            "id": call_id,
            "summary": summary,
            "tips": tips,
            "updated_at": datetime.now().isoformat()
        }
        result = supabase.table("sales_calls").upsert(data).execute()
        logger.info("Results saved successfully")
    except Exception as e:
        logger.error(f"Error saving results to Supabase: {e}")
        raise HTTPException(status_code=500, detail="Error saving results")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
