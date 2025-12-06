import os
import openai
import aiohttp
import json
import logging
import traceback
import asyncio
import base64
import tempfile
import re
from fastapi import FastAPI, File, UploadFile, HTTPException, WebSocket, WebSocketDisconnect
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
                "analysis": "No audio detected",
                "audio_base64": None,
                "audio_format": None
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

        # Generate audio response using TTS
        audio_response = None
        audio_base64 = None
        try:
            logger.info("Generating TTS audio response")
            tts_response = client.audio.speech.create(
                model="tts-1",
                voice="alloy",
                input=analysis
            )
            audio_response = tts_response.content
            audio_base64 = base64.b64encode(audio_response).decode('utf-8')
            logger.info(f"TTS audio generated, size: {len(audio_response)} bytes")
        except Exception as e:
            logger.error(f"TTS error: {str(e)}")
            # Continue without audio if TTS fails

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

        response_data = {
            "transcription": transcription_text,
            "analysis": analysis
        }
        
        # Add audio response if available
        if audio_base64:
            response_data["audio_base64"] = audio_base64
            response_data["audio_format"] = "mp3"

        return response_data

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

async def process_audio_chunks(audio_chunks, user_id: str):
    """Process accumulated audio chunks: transcribe and generate response.
    
    Note: This function does NOT save data to Supabase. It only processes
    audio in real-time and returns responses. For saving sales call data,
    use the /analyze_sales_call endpoint instead.
    """
    if not audio_chunks or len(audio_chunks) == 0:
        return None, None, None
    
    # Create temporary files
    temp_audio_file = tempfile.NamedTemporaryFile(delete=False, suffix=".webm")
    wav_file_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    temp_audio_file.close()
    wav_file_path.close()
    
    try:
        # Write all chunks to file
        with open(temp_audio_file.name, "wb") as f:
            for chunk in audio_chunks:
                f.write(chunk)
        
        logger.info(f"Processing audio file: {temp_audio_file.name}, size: {os.path.getsize(temp_audio_file.name)} bytes")
        
        # Check if audio is silent
        if is_silent(temp_audio_file.name):
            logger.info("Audio is silent, skipping processing")
            return None, None, None
        
        # Convert to WAV format
        try:
            audio = AudioSegment.from_file(temp_audio_file.name)
            audio.export(wav_file_path.name, format="wav")
            logger.info(f"Audio converted to WAV: {wav_file_path.name}")
        except Exception as e:
            logger.error(f"Failed to convert audio: {str(e)}")
            return None, None, None
        
        # Transcribe using Whisper
        transcription_text = ""
        try:
            with open(wav_file_path.name, "rb") as audio_file:
                logger.info("Starting Whisper transcription")
                transcript = client.audio.transcriptions.create(
                    file=audio_file,
                    model="whisper-1"
                )
                transcription_text = transcript.text
                logger.info(f"Transcription completed: {transcription_text[:100]}...")
        except Exception as e:
            logger.error(f"Transcription error: {str(e)}")
            return None, None, None
        
        # Query Pinecone for relevant context (Northwest Missouri State University course catalog)
        passages = []
        try:
            # Try multiple query variations for better semantic search
            query_variations = [
                transcription_text,  # Original query
                f"Northwest Missouri State University {transcription_text}",  # With university name
                f"course catalog {transcription_text}",  # With course catalog prefix
            ]
            
            for query_variant in query_variations:
                logger.info(f"Trying query variant: '{query_variant[:150]}...'")
                embedding = await get_embedding(query_variant)
                logger.info(f"Embedding generated, dimension: {len(embedding)}")
                logger.info("Querying Pinecone for relevant course catalog passages")
                # Query with higher top_k to get more results
                passages = await query_pinecone(embedding, top_k=15)
                logger.info(f"Query variant returned {len(passages)} passages")
                if passages:
                    logger.info(f"Successfully retrieved {len(passages)} passages with query: '{query_variant[:100]}...'")
                    for i, passage in enumerate(passages, 1):
                        logger.info(f"Pinecone passage {i} (first 200 chars): {passage[:200]}...")
                    break  # Stop trying other variants if we got results
                else:
                    logger.warning(f"No passages found for query variant: '{query_variant[:100]}...'")
            
            if not passages:
                logger.error("All query variations failed to retrieve passages from Pinecone")
                logger.error("This might indicate:")
                logger.error("1. Pinecone index might be empty or not properly loaded")
                logger.error("2. Embedding model mismatch (data might be embedded with different model)")
                logger.error("3. Metadata structure mismatch")
        except Exception as e:
            logger.error(f"Pinecone query error: {str(e)}\n{traceback.format_exc()}")
            logger.warning("Continuing without context")
            passages = []
        
        # Generate response using GPT-4 with Pinecone context
        response_text = ""
        try:
            logger.info("Generating GPT-4 response with Pinecone course catalog context")
            
            # Build prompt with context from Pinecone course catalog
            context_text = ""
            if passages:
                # Clean and deduplicate passages
                cleaned_passages = []
                seen = set()
                for passage in passages:
                    cleaned = clean_passage(passage)
                    # Simple deduplication based on first 100 chars
                    passage_key = cleaned[:100].lower().strip()
                    if cleaned and len(cleaned) > 50 and passage_key not in seen:
                        cleaned_passages.append(cleaned)
                        seen.add(passage_key)
                
                if cleaned_passages:
                    context_text = "\n\nRelevant Course Catalog Information from Northwest Missouri State University:\n" + "\n\n".join([f"{i+1}. {p}" for i, p in enumerate(cleaned_passages)])
                    logger.info(f"Using {len(cleaned_passages)} cleaned passages as context for GPT-4")
                else:
                    context_text = "\n\n[No relevant course catalog information was found in the database for this query.]"
                    logger.warning("All passages were filtered out after cleaning")
            else:
                context_text = "\n\n[No relevant course catalog information was found in the database for this query.]"
                logger.warning("No passages found - GPT-4 will respond without course catalog context")
            
            prompt = f"""You are an academic assistant for Northwest Missouri State University. You help students by providing accurate information about courses, majors, and academic programs based on the official course catalog.

User's Question:
\"\"\"{transcription_text}\"\"\"
{context_text}

Based on the course catalog information provided above, provide a clear and accurate response that:
- If course catalog information is provided: Directly answer using that information with specific course details (course codes, names, descriptions, prerequisites, credits, etc.)
- If NO course catalog information is provided: Politely inform the user that you couldn't find specific information in the course catalog for their question, and suggest they:
  * Try rephrasing their question
  * Contact the registrar's office
  * Check the official Northwest Missouri State University website
  * Be more specific about the course, major, or program they're asking about
- Use the exact course codes, names, and details from the catalog when available
- Be specific and accurate - only provide information that is in the course catalog context

Important: If course catalog information is provided above, use it. If not, clearly state that you couldn't find the information and provide helpful guidance."""
            
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an academic assistant for Northwest Missouri State University. You help students by providing accurate, specific information about courses, majors, programs, and academic requirements from the official Northwest Missouri State University course catalog. You always base your answers on the provided course catalog information and cite specific courses, codes, and details when available."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,  # Lower temperature for more accurate, factual responses
                max_tokens=1000
            )
            response_text = response.choices[0].message.content
            logger.info(f"Response generated: {response_text[:100]}...")
        except Exception as e:
            logger.error(f"GPT response error: {str(e)}")
            return transcription_text, None, None
        
        # Generate audio response using TTS
        audio_response = None
        try:
            logger.info("Generating TTS audio response")
            tts_response = client.audio.speech.create(
                model="tts-1",
                voice="alloy",
                input=response_text
            )
            audio_response = tts_response.content
            logger.info(f"TTS audio generated, size: {len(audio_response)} bytes")
        except Exception as e:
            logger.error(f"TTS error: {str(e)}")
            return transcription_text, response_text, None
        
        return transcription_text, response_text, audio_response
    
    except Exception as e:
        logger.error(f"Error processing audio: {str(e)}\n{traceback.format_exc()}")
        return None, None, None
    finally:
        # Clean up temporary files
        for file_path in [temp_audio_file.name, wav_file_path.name]:
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except Exception as e:
                    logger.error(f"Failed to clean up {file_path}: {str(e)}")

@app.websocket("/ws/realtime")
async def websocket_realtime(websocket: WebSocket):
    """WebSocket endpoint for real-time audio streaming and analysis.
    
    This endpoint processes audio in real-time and returns responses
    but does NOT save data to Supabase. For saving sales call analysis,
    use the /analyze_sales_call endpoint instead.
    """
    await websocket.accept()
    
    # Get user_id from query parameters
    user_id = websocket.query_params.get("user_id", "unknown")
    logger.info(f"WebSocket connection established for user_id: {user_id}")
    
    audio_chunks = []
    is_receiving_audio = False
    last_audio_time = None
    
    try:
        # Send connection confirmation
        await websocket.send_json({
            "type": "connection",
            "status": "connected",
            "message": "WebSocket connection established"
        })
        
        while True:
            try:
                # Receive message with timeout
                message = await asyncio.wait_for(websocket.receive(), timeout=1.0)
                
                # Handle text messages (JSON)
                if "text" in message:
                    try:
                        data = json.loads(message["text"])
                        message_type = data.get("type")
                        
                        if message_type == "connection":
                            logger.info(f"Connection message from user_id: {user_id}, message: {data.get('message', '')}")
                            await websocket.send_json({
                                "type": "connection",
                                "status": "acknowledged",
                                "message": "Connection acknowledged"
                            })
                        
                        elif message_type == "audio":
                            logger.info(f"Audio message received from user_id: {user_id}")
                            is_receiving_audio = True
                            audio_chunks = []
                            last_audio_time = datetime.now()
                            await websocket.send_json({
                                "type": "audio",
                                "status": "ready",
                                "message": "Ready to receive audio chunks"
                            })
                        
                        elif message_type == "audio_end":
                            # Process accumulated audio
                            if is_receiving_audio and len(audio_chunks) > 0:
                                logger.info(f"Processing {len(audio_chunks)} audio chunks")
                                await websocket.send_json({
                                    "type": "processing",
                                    "message": "Processing audio..."
                                })
                                
                                transcription, response_text, audio_response = await process_audio_chunks(audio_chunks, user_id)
                                
                                # Send transcription
                                if transcription:
                                    await websocket.send_json({
                                        "type": "transcription",
                                        "transcription": transcription
                                    })
                                
                                # Send text response
                                if response_text:
                                    await websocket.send_json({
                                        "type": "answer",
                                        "answer": response_text
                                    })
                                
                                # Send audio response
                                if audio_response:
                                    # Send as base64 for easier handling
                                    audio_base64 = base64.b64encode(audio_response).decode('utf-8')
                                    await websocket.send_json({
                                        "type": "audio_response",
                                        "audio_base64": audio_base64,
                                        "format": "mp3"
                                    })
                                
                                # Reset for next audio
                                audio_chunks = []
                                is_receiving_audio = False
                        
                    except json.JSONDecodeError:
                        logger.warning(f"Invalid JSON received: {message.get('text', '')}")
                        continue
                
                # Handle binary messages (audio chunks)
                elif "bytes" in message:
                    if is_receiving_audio:
                        audio_chunk = message["bytes"]
                        audio_chunks.append(audio_chunk)
                        last_audio_time = datetime.now()
                        logger.debug(f"Received audio chunk of size: {len(audio_chunk)} bytes, total chunks: {len(audio_chunks)}")
                    else:
                        logger.warning("Received audio chunk but not in audio receiving mode")
                
            except asyncio.TimeoutError:
                # Check if we should process audio after timeout (e.g., 2 seconds of silence)
                if is_receiving_audio and len(audio_chunks) > 0 and last_audio_time:
                    time_since_last = (datetime.now() - last_audio_time).total_seconds()
                    if time_since_last > 2.0:  # 2 seconds of silence
                        logger.info("Processing audio after timeout")
                        await websocket.send_json({
                            "type": "processing",
                            "message": "Processing audio..."
                        })
                        
                        transcription, response_text, audio_response = await process_audio_chunks(audio_chunks, user_id)
                        
                        if transcription:
                            await websocket.send_json({
                                "type": "transcription",
                                "transcription": transcription
                            })
                        
                        if response_text:
                            await websocket.send_json({
                                "type": "answer",
                                "answer": response_text
                            })
                        
                        if audio_response:
                            audio_base64 = base64.b64encode(audio_response).decode('utf-8')
                            await websocket.send_json({
                                "type": "audio_response",
                                "audio_base64": audio_base64,
                                "format": "mp3"
                            })
                        
                        audio_chunks = []
                        is_receiving_audio = False
                        last_audio_time = None
                continue
                    
            except WebSocketDisconnect:
                logger.info(f"WebSocket disconnected for user_id: {user_id}")
                break
            except Exception as e:
                logger.error(f"Error in WebSocket handler: {str(e)}\n{traceback.format_exc()}")
                await websocket.send_json({
                    "type": "error",
                    "message": f"Error processing message: {str(e)}"
                })
    
    except Exception as e:
        logger.error(f"WebSocket error for user_id {user_id}: {str(e)}\n{traceback.format_exc()}")
    finally:
        logger.info(f"WebSocket connection closed for user_id: {user_id}")

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
        # Run synchronous OpenAI client in executor to avoid blocking
        loop = asyncio.get_event_loop()
        embedding_resp = await loop.run_in_executor(
            None,
            lambda: client.embeddings.create(
                input=query,
                model="text-embedding-3-small"
            )
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
                logger.info(f"Pinecone /context endpoint returned {len(result.get('matches', []))} matches")
                passages = []
                for i, match in enumerate(result.get("matches", []), 1):
                    score = match.get("score", 0)
                    meta = match.get("metadata", {})
                    
                    # Try multiple possible metadata field names
                    passage = (
                        meta.get("text") or 
                        meta.get("passage") or 
                        meta.get("content") or 
                        meta.get("chunk") or 
                        meta.get("document") or
                        meta.get("data") or
                        str(meta)
                    )
                    
                    logger.info(f"Match {i}: score={score:.4f}, metadata_keys={list(meta.keys())}")
                    
                    if passage and score > 0:
                        passages.append(passage)
                
                logger.info(f"Extracted {len(passages)} passages from /context query")
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

def clean_passage(passage):
    """Clean passage by removing HTML/UI boilerplate text."""
    if not passage:
        return passage
    
    # Remove common HTML/UI boilerplate patterns
    patterns_to_remove = [
        r"HELP\s+\d{4}-\d{4}\s+Undergraduate\s+Catalog.*?Add to Portfolio.*?\)",
        r"Print-Friendly Page.*?\)",
        r"Facebook this Page.*?\)",
        r"Tweet this Page.*?\)",
        r"Add to Portfolio.*?\)",
        r"\(opens a new window\)",
        r"HELP.*?Catalog",
    ]
    
    cleaned = passage
    for pattern in patterns_to_remove:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE | re.DOTALL)
    
    # Remove extra whitespace
    cleaned = re.sub(r'\s+', ' ', cleaned)
    cleaned = cleaned.strip()
    
    # If passage is too short after cleaning, return original
    if len(cleaned) < 50:
        return passage
    
    return cleaned

async def get_embedding(text):
    """Get embedding for the text using OpenAI API."""
    try:
        # Run synchronous OpenAI client in executor to avoid blocking
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: client.embeddings.create(
                input=text,
                model="text-embedding-3-small"
            )
        )
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Error generating embedding: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error generating embedding: {str(e)}")

async def query_pinecone(embedding, top_k=3):
    """Query Pinecone with the embedding."""
    try:
        headers = {
            "Api-Key": required_env_vars["PINECONE_API_KEY"],
            "Content-Type": "application/json"
        }
        payload = {
            "vector": embedding,
            "topK": top_k,  # Fixed: use the parameter instead of hardcoded 3
            "includeMetadata": True,
            "includeValues": False  # Don't need vector values in response
        }
        logger.info(f"Pinecone query: topK={top_k}, embedding dimension={len(embedding)}")
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{PINECONE_INDEX_HOST}/query", headers=headers, data=json.dumps(payload)) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    logger.error(f"Pinecone query failed: {resp.status} {error_text}")
                    return []
                result = await resp.json()
                logger.info(f"Pinecone returned {len(result.get('matches', []))} matches")
                
                passages = []
                for i, match in enumerate(result.get("matches", []), 1):
                    score = match.get("score", 0)
                    meta = match.get("metadata", {})
                    
                    # Try multiple possible metadata field names
                    passage = (
                        meta.get("text") or 
                        meta.get("passage") or 
                        meta.get("content") or 
                        meta.get("chunk") or 
                        meta.get("document") or
                        meta.get("data") or
                        meta.get("course_description") or
                        meta.get("description") or
                        str(meta)  # Fallback: use entire metadata as string
                    )
                    
                    logger.info(f"Match {i}: score={score:.4f}, metadata_keys={list(meta.keys())}, passage_length={len(passage) if passage else 0}")
                    
                    # For course catalog, accept all matches with positive scores
                    # Temporarily removing threshold to see all results for debugging
                    # Cosine similarity typically ranges from -1 to 1
                    # For debugging: accept all matches, we can filter later if needed
                    if passage:
                        # Clean the passage to remove HTML/UI boilerplate
                        cleaned_passage = clean_passage(passage)
                        if cleaned_passage and len(cleaned_passage) > 50:  # Only add if meaningful content remains
                            passages.append(cleaned_passage)
                            logger.info(f"Match {i} accepted with score {score:.4f}, original length: {len(passage)}, cleaned length: {len(cleaned_passage)}")
                            logger.debug(f"Cleaned passage preview: {cleaned_passage[:200]}...")
                        else:
                            logger.warning(f"Match {i} passage too short after cleaning, skipping")
                    else:
                        logger.warning(f"Match {i} has no extractable passage, metadata keys: {list(meta.keys())}")
                
                logger.info(f"Extracted {len(passages)} passages from Pinecone results")
                if len(passages) == 0:
                    if len(result.get("matches", [])) > 0:
                        logger.error("No passages extracted despite having matches. Full result:")
                        logger.error(json.dumps(result, indent=2))
                        logger.error("This suggests a metadata structure mismatch. Check the metadata keys above.")
                    else:
                        logger.error("Pinecone returned NO matches at all. This could mean:")
                        logger.error("1. The index is empty")
                        logger.error("2. The query embedding doesn't match any vectors")
                        logger.error("3. The index name or host is incorrect")
                        logger.error(f"Full Pinecone response: {json.dumps(result, indent=2)}")
                
                return passages
    except Exception as e:
        logger.error(f"Error querying Pinecone: {e}\n{traceback.format_exc()}")
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
