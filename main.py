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

app = FastAPI(
    title="Majors Exploration AI Assistant",
    description="AI-powered academic advisor that helps students explore majors and minors through voice conversations",
    version="2.0.0"
)

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

@app.post("/explore_majors")
async def explore_majors(file: UploadFile = File(...)):
    """
    Analyze student conversation about academic interests and recommend suitable majors/minors.
    Transcribes audio, retrieves relevant program information from Pinecone, and provides AI-powered recommendations.
    """
    temp_file_path = "temp_audio.m4a"
    wav_file_path = "temp_audio.wav"
    bucket_name = "audio-files"  # Use existing bucket for now

    try:
        # Enhanced logging for debugging
        logger.info(f"Received file upload request:")
        logger.info(f"  - Filename: {file.filename}")
        logger.info(f"  - Content-Type: {file.content_type}")
        logger.info(f"  - File size: {file.size if hasattr(file, 'size') else 'Unknown'}")
        
        # Check if file is provided
        if not file.filename:
            logger.error("No filename provided in upload")
            raise HTTPException(
                status_code=400,
                detail="No file provided. Please upload an audio file."
            )
        
        # Check file format
        if not file.filename.endswith(('.m4a', '.mp3', '.wav', '.ogg')):
            logger.error(f"Invalid file format: {file.filename}")
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

        # Upload to Supabase Storage (temporarily disabled until bucket is created)
        try:
            with open(temp_file_path, "rb") as audio_file:
                response = supabase.storage.from_(bucket_name).upload(
                    path=storage_path,
                    file=audio_file
                )
            logger.info(f"File uploaded to Supabase Storage: {file_name}")
        except Exception as e:
            # Temporarily skip storage upload if bucket doesn't exist
            logger.warning(f"Storage upload skipped (bucket may not exist): {str(e)}")
            logger.info("Continuing with analysis without storing audio file...")

        # Transcribe using Whisper
        with open(wav_file_path, "rb") as audio_file:
            logger.info("Starting audio transcription with Whisper")
            transcript = client.audio.transcriptions.create(
                file=audio_file,
                model="whisper-1"
            )
            logger.info("Transcription completed successfully")

        transcription_text = transcript.text

        # Generate embedding for transcript to find relevant programs
        logger.info("Generating embedding for transcript")
        embedding_response = client.embeddings.create(
            model="text-embedding-3-small",
            input=transcription_text
        )
        embedding = embedding_response.data[0].embedding

        # Query Pinecone for relevant academic programs
        logger.info("Querying Pinecone for relevant academic programs")
        context = await retrieve_context(transcription_text, top_k=5)
        relevant_programs = context.get("passages", [])

        # Generate academic program recommendations using GPT-4
        programs_context = "\n\n".join(relevant_programs) if relevant_programs else "No specific program information found."
        
        prompt = f"""
You are an experienced academic advisor helping a student explore potential majors and minors based on their conversation.

Please analyze the following student conversation and provide personalized academic recommendations. Focus on:
1. Understanding their interests, strengths, and goals
2. Recommending specific majors that align with their interests
3. Suggesting complementary minors that could enhance their education
4. Explaining why these programs are a good fit

Student Conversation:
\"\"\"
{transcription_text}
\"\"\"

Available Academic Programs Context:
\"\"\"
{programs_context}
\"\"\"

Format your response as:

**Recommended Major:**
<Major name and detailed explanation of why it fits>

**Recommended Minor:**
<Minor name and how it complements the major>

**Additional Considerations:**
<Any other relevant advice or alternative paths to consider>

Be encouraging, specific, and focus on helping the student find their academic passion.
"""

        logger.info("Starting GPT-4 analysis for academic recommendations")
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a knowledgeable and supportive academic advisor who helps students find their ideal academic path."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.6
        )
        logger.info("GPT-4 analysis completed successfully")

        recommendations = response.choices[0].message.content

        # Store data in Supabase database
        logger.info("Storing academic exploration data in Supabase database")
        data = {
            "transcription": transcription_text,
            "recommendations": recommendations,
            "relevant_programs": relevant_programs,
            "created_at": datetime.now().isoformat()
        }

        result = supabase.table("academic_explorations").insert(data).execute()
        logger.info("Academic exploration data stored successfully in database")

        return {
            "transcription": transcription_text,
            "recommendations": recommendations,
            "relevant_programs": relevant_programs
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

@app.get("/health")
async def health_check():
    return {
        "status": "ok", 
        "message": "Majors Exploration AI Assistant is running smoothly 🎓",
        "version": "2.0.0"
    }

@app.post("/test_upload")
async def test_upload(file: UploadFile = File(...)):
    """Test endpoint to debug file upload issues."""
    try:
        logger.info(f"Test upload received:")
        logger.info(f"  - Filename: {file.filename}")
        logger.info(f"  - Content-Type: {file.content_type}")
        logger.info(f"  - File size: {file.size if hasattr(file, 'size') else 'Unknown'}")
        
        return {
            "status": "success",
            "filename": file.filename,
            "content_type": file.content_type,
            "message": "File upload test successful"
        }
    except Exception as e:
        logger.error(f"Test upload failed: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Test upload failed: {str(e)}")

@app.get("/transcript/{exploration_id}")
async def get_transcript(exploration_id: str):
    """Fetch transcript by exploration_id."""
    try:
        response = supabase.table("academic_explorations").select("transcription").eq("id", exploration_id).execute()
        transcript = response.data[0]["transcription"] if response.data else ""
        return {"transcript": transcript}
    except Exception as e:
        logger.error(f"Error fetching transcript: {e}")
        raise HTTPException(status_code=500, detail="Error fetching transcript")

@app.post("/context")
async def retrieve_context(query: str, top_k: int = 5):
    """Retrieve top-K passages from Pinecone."""
    try:
        embedding_resp = client.embeddings.create(
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

@app.post("/generate_detailed_analysis")
async def generate_detailed_analysis(exploration_id: str):
    """Generate detailed academic analysis and career guidance."""
    try:
        # Fetch transcript (get_transcript returns {"transcript": "..."} via the endpoint)
        transcript_resp = await get_transcript(exploration_id)
        transcript = transcript_resp.get("transcript") if isinstance(transcript_resp, dict) else transcript_resp
        if not transcript:
            raise HTTPException(status_code=404, detail="Transcript not found")

        # Retrieve relevant context from Pinecone
        context = await retrieve_context(transcript, top_k=5)
        passages = context.get("passages", []) if isinstance(context, dict) else context or []

        # Generate detailed analysis and career guidance
        analysis, career_guidance = await generate_detailed_academic_analysis(transcript, passages)

        # Save results to Supabase
        await save_academic_results(exploration_id, analysis, career_guidance)

        return {"detailed_analysis": analysis, "career_guidance": career_guidance}
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error generating detailed analysis: {e}")
        raise HTTPException(status_code=500, detail="Error generating detailed analysis")


async def generate_detailed_academic_analysis(transcript_text, passages):
    """Generate detailed academic analysis and career guidance using GPT-4."""
    try:
        logger.info("Generating detailed academic analysis")
        
        passages_text = "\n\n".join(passages) if passages else "No specific program information found."
        logger.info(f"Using {len(passages)} relevant program passages for analysis")

        prompt = f"""
You are an experienced academic advisor and career counselor helping a student with detailed academic and career planning.

Based on the student's conversation below and the relevant academic program information, provide:

1. A comprehensive analysis of the student's interests, strengths, and academic goals
2. Detailed career guidance including potential career paths, job outlook, and salary expectations
3. Academic pathway recommendations including prerequisites, course sequences, and extracurricular activities
4. Alternative academic paths to consider

Student Conversation:
\"\"\"{transcript_text}\"\"\"

Relevant Academic Programs Information:
\"\"\"{passages_text}\"\"\"

Format your response as:

**Academic Profile Analysis:**
<Detailed analysis of the student's interests, strengths, and goals>

**Career Pathways:**
<Specific career options with job outlook and salary information>

**Academic Roadmap:**
<Recommended course sequences, prerequisites, and academic milestones>

**Alternative Paths:**
<Other academic programs or career directions to consider>

**Next Steps:**
<Specific actions the student should take to move forward>

Be comprehensive, encouraging, and provide specific, actionable advice.
"""

        logger.info("Starting GPT-4 analysis for detailed academic guidance")
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a knowledgeable academic advisor and career counselor with expertise in helping students find their ideal academic and career path."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.6
        )
        logger.info("GPT-4 detailed analysis completed successfully")

        result = response.choices[0].message.content

        # Parse the detailed response
        analysis_sections = {
            "academic_profile": "",
            "career_pathways": "",
            "academic_roadmap": "",
            "alternative_paths": "",
            "next_steps": ""
        }

        # Simple parsing - in a production app, you might want more robust parsing
        if "**Academic Profile Analysis:**" in result:
            analysis_sections["academic_profile"] = result.split("**Academic Profile Analysis:**")[1].split("**Career Pathways:**")[0].strip()
        if "**Career Pathways:**" in result:
            analysis_sections["career_pathways"] = result.split("**Career Pathways:**")[1].split("**Academic Roadmap:**")[0].strip()
        if "**Academic Roadmap:**" in result:
            analysis_sections["academic_roadmap"] = result.split("**Academic Roadmap:**")[1].split("**Alternative Paths:**")[0].strip()
        if "**Alternative Paths:**" in result:
            analysis_sections["alternative_paths"] = result.split("**Alternative Paths:**")[1].split("**Next Steps:**")[0].strip()
        if "**Next Steps:**" in result:
            analysis_sections["next_steps"] = result.split("**Next Steps:**")[1].strip()

        return analysis_sections, result
    except Exception as e:
        logger.error(f"Error in generate_detailed_academic_analysis: {e}")
        raise HTTPException(status_code=500, detail="Error generating detailed academic analysis")

def get_embedding(text):
    """Get embedding for the text using OpenAI API."""
    try:
        response = client.embeddings.create(
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

async def save_academic_results(exploration_id, analysis, career_guidance):
    """Save the generated academic analysis and career guidance to Supabase."""
    try:
        data = {
            "id": exploration_id,
            "detailed_analysis": analysis,
            "career_guidance": career_guidance,
            "updated_at": datetime.now().isoformat()
        }
        result = supabase.table("academic_explorations").upsert(data).execute()
        logger.info("Academic analysis results saved successfully")
    except Exception as e:
        logger.error(f"Error saving academic results to Supabase: {e}")
        raise HTTPException(status_code=500, detail="Error saving academic results")

@app.post("/search_programs")
async def search_programs(query: str, top_k: int = 10):
    """Search for academic programs based on text query."""
    try:
        logger.info(f"Searching for programs with query: {query}")
        
        # Get embedding for the search query
        embedding_response = client.embeddings.create(
            model="text-embedding-3-small",
            input=query
        )
        embedding = embedding_response.data[0].embedding
        
        # Search Pinecone for relevant programs
        context = await retrieve_context(query, top_k=top_k)
        relevant_programs = context.get("passages", [])
        
        return {
            "query": query,
            "results": relevant_programs,
            "total_found": len(relevant_programs)
        }
    except Exception as e:
        logger.error(f"Error searching programs: {e}")
        raise HTTPException(status_code=500, detail="Error searching programs")

@app.get("/exploration_history")
async def get_exploration_history(limit: int = 10):
    """Get recent academic exploration history."""
    try:
        response = supabase.table("academic_explorations").select("*").order("created_at", desc=True).limit(limit).execute()
        return {"explorations": response.data}
    except Exception as e:
        logger.error(f"Error fetching exploration history: {e}")
        raise HTTPException(status_code=500, detail="Error fetching exploration history")

@app.get("/exploration/{exploration_id}")
async def get_exploration_details(exploration_id: str):
    """Get detailed information about a specific academic exploration."""
    try:
        response = supabase.table("academic_explorations").select("*").eq("id", exploration_id).execute()
        if not response.data:
            raise HTTPException(status_code=404, detail="Exploration not found")
        return {"exploration": response.data[0]}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching exploration details: {e}")
        raise HTTPException(status_code=500, detail="Error fetching exploration details")

# Legacy endpoint for backward compatibility
@app.post("/analyze_sales_call")
async def analyze_sales_call_legacy(file: UploadFile = File(...)):
    """
    LEGACY ENDPOINT - Redirects to /explore_majors
    This endpoint is maintained for backward compatibility with existing clients.
    Please update your client to use /explore_majors instead.
    """
    logger.warning("Legacy endpoint /analyze_sales_call called. Please update client to use /explore_majors")
    
    # Redirect to the new endpoint
    return await explore_majors(file)
