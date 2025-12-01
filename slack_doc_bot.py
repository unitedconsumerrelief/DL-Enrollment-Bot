import os
import re
import time
import logging
import traceback
import openai
import faiss
import numpy as np
import pdfplumber
from PIL import Image
import pytesseract
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from slack_sdk.web import WebClient
from slack_sdk.errors import SlackApiError
from dotenv import load_dotenv
from policy_codex_full_ready import POLICY_CODEX

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()
SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
SLACK_APP_TOKEN = os.getenv("SLACK_APP_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Validate environment variables
if not SLACK_BOT_TOKEN:
    raise ValueError("❌ SLACK_BOT_TOKEN not found in .env file")
if not SLACK_APP_TOKEN:
    raise ValueError("❌ SLACK_APP_TOKEN not found in .env file")
if not OPENAI_API_KEY:
    raise ValueError("❌ OPENAI_API_KEY not found in .env file")

# Initialize OpenAI client (new API v1.x)
USE_NEW_API = False
openai_client = None
try:
    from openai import OpenAI
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    USE_NEW_API = True
    logger.info("Using OpenAI API v1.x (new client)")
except Exception as e:
    # Fallback for old API
    try:
        openai.api_key = OPENAI_API_KEY
        USE_NEW_API = False
        logger.info("Using OpenAI API v0.x (legacy)")
    except Exception as e2:
        logger.error(f"Could not initialize OpenAI: {e}, {e2}")
        raise ValueError("❌ Could not initialize OpenAI API. Please check your OPENAI_API_KEY.")

app = App(token=SLACK_BOT_TOKEN)
client = WebClient(token=SLACK_BOT_TOKEN)

index = None
chunks = []
chunk_sources = []

def extract_chunks_from_text(text, source):
    output = []
    lines = text.split("\n")
    buffer = []
    max_chunk_words = 120
    
    # Keywords that indicate important policy content even in short chunks
    policy_keywords = [
        "credit union", "secured loan", "furniture", "military", "federal", 
        "student loan", "auto loan", "mortgage", "collections", "ach",
        "minimum payment", "enrollment", "eligible", "disqualified", "restricted",
        "capped", "limit", "requirement", "condition", "waiver", "approval",
        "not allowed", "prohibited", "excluded", "restricted", "conditional",
        "must", "only if", "required", "necessary", "mandatory"
    ]
    
    # Emojis that indicate important policy status
    policy_emojis = ["❌", "✅", "⚠️", "🚫", "💳", "🏦", "💰", "📋", "🔒", "⚡"]

    def is_important_content(text):
        """Check if content contains important policy indicators"""
        text_lower = text.lower()
        has_keywords = any(keyword in text_lower for keyword in policy_keywords)
        has_emojis = any(emoji in text for emoji in policy_emojis)
        has_bullets = "-" in text or "•" in text
        has_restrictions = any(term in text_lower for term in ["not allowed", "prohibited", "excluded", "restricted"])
        has_requirements = any(term in text_lower for term in ["must", "only if", "required", "necessary"])
        return has_keywords or has_emojis or has_bullets or has_restrictions or has_requirements

    def is_policy_header(line):
        """Check if line is a policy header (all caps, short, likely creditor name)"""
        return (line.isupper() and 
                len(line.split()) <= 4 and  # Reduced from 8 to 4 for shorter headers like "OPORTUN"
                len(line) >= 2 and
                not line.startswith("-") and
                not line.startswith("•"))

    def is_bullet_or_indented(line):
        """Check if line is a bullet point or indented policy line"""
        return (line.startswith("-") or 
                line.startswith("•") or 
                line.startswith("  ") or  # Indented lines
                line.startswith("\t"))    # Tab-indented lines

    def should_merge_with_previous(line, buffer):
        """Determine if line should be merged with previous content"""
        if not buffer:
            return False
        
        # Always merge bullet points or indented lines with previous content
        if is_bullet_or_indented(line):
            return True
        
        # Merge short lines that seem related to previous content
        if len(line.split()) <= 5 and is_important_content(line):
            return True
        
        # Merge lines that continue a policy rule (containing emojis or keywords)
        if is_important_content(line) and is_important_content(" ".join(buffer)):
            return True
        
        # Merge if previous content is a policy header and current line is related
        if buffer and is_policy_header(buffer[0]) and is_important_content(line):
            return True
        
        # Merge if we have a policy header and current line is short and related
        if buffer and is_policy_header(buffer[0]) and len(line.split()) <= 8:
            return True
        
        return False

    def is_policy_block(buffer):
        """Check if buffer contains a complete policy block worth preserving"""
        if not buffer:
            return False
        
        # If it has a header and bullet points, it's definitely a policy block
        if len(buffer) >= 2 and is_policy_header(buffer[0]):
            has_bullets = any(is_bullet_or_indented(line) for line in buffer[1:])
            if has_bullets:
                return True
        
        # If it contains important policy content with emojis or restrictions, preserve it
        joined = " ".join(buffer)
        if is_important_content(joined):
            return True
        
        # If it's a short header with any related content, preserve it
        if len(buffer) >= 2 and is_policy_header(buffer[0]):
            return True
        
        return False

    def flush_buffer():
        if buffer:
            joined = " ".join(buffer).strip()
            # Always preserve policy blocks, regardless of length
            if is_policy_block(buffer) or len(joined.split()) >= 3:
                output.append((joined, source))
            buffer.clear()

    for line in lines:
        line = line.strip()
        
        if line == "":
            flush_buffer()
        elif should_merge_with_previous(line, buffer):
            # Merge with previous content instead of creating new chunk
            buffer.append(line)
        elif is_policy_header(line):
            # This is likely a policy header - flush previous and start new
            flush_buffer()
            buffer.append(line)
        else:
            buffer.append(line)
            # Check if we've exceeded max chunk size
            if len(" ".join(buffer).split()) > max_chunk_words:
                flush_buffer()

    flush_buffer()
    return output

def load_documents(folder_path="documents"):
    logger.info("📄 Loading and chunking documents...")
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"❌ Documents folder not found: {folder_path}")
    all_chunks = []
    all_sources = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf") or filename.endswith(".txt"):
            path = os.path.join(folder_path, filename)
            logger.info(f"🔍 Processing: {filename}")
            try:
                if filename.endswith(".pdf"):
                    with pdfplumber.open(path) as pdf:
                        text_blocks = []
                        for page in pdf.pages:
                            text = page.extract_text()
                            if not text:
                                # Try OCR as fallback, but handle gracefully if tesseract not available
                                try:
                                    img = page.to_image(resolution=300).original
                                    pil_image = Image.frombytes("RGB", img.size, img.tobytes())
                                    text = pytesseract.image_to_string(pil_image)
                                    if not text.strip():
                                        logger.warning(f"No text extracted from {filename} page {page.page_number} (even with OCR)")
                                except Exception as ocr_error:
                                    logger.warning(f"OCR failed for {filename} page {page.page_number}: {ocr_error}. Skipping OCR.")
                                    text = ""  # Continue without OCR text
                            text_blocks.append(text.strip())
                        combined = "\n".join(text_blocks)
                else:
                    with open(path, "r", encoding="utf-8") as f:
                        combined = f.read()
                doc_chunks = extract_chunks_from_text(combined, filename)
                for chunk, source in doc_chunks:
                    all_chunks.append(chunk)
                    all_sources.append(filename)
                logger.info(f"✅ Extracted {len(doc_chunks)} chunks from: {filename}")
            except Exception as e:
                logger.error(f"❌ ERROR processing {filename}: {str(e)}\n{traceback.format_exc()}")
    return all_chunks, all_sources

def embed_chunks(chunks, batch_size=100, max_retries=3):
    """Create embeddings with batching and retry logic for reliability"""
    print("🔢 Creating embeddings...")
    logger.info(f"Creating embeddings for {len(chunks)} chunks in batches of {batch_size}")
    all_embeddings = []
    
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        logger.info(f"Processing batch {i//batch_size + 1}/{(len(chunks) + batch_size - 1)//batch_size}")
        
        for attempt in range(max_retries):
            try:
                if USE_NEW_API and openai_client:
                    response = openai_client.embeddings.create(
                        model="text-embedding-ada-002",
                        input=batch
                    )
                    batch_embeddings = [np.array(r.embedding, dtype=np.float32) for r in response.data]
                else:
                    # Old API fallback
                    response = openai.Embedding.create(
                        model="text-embedding-ada-002",
                        input=batch
                    )
                    batch_embeddings = [np.array(r["embedding"], dtype=np.float32) for r in response["data"]]
                all_embeddings.extend(batch_embeddings)
                break  # Success, move to next batch
            except Exception as e:
                error_str = str(e).lower()
                # Check for rate limit (works for both old and new API)
                if "rate limit" in error_str or "ratelimit" in error_str:
                    wait_time = 2 * (2 ** attempt)  # Exponential backoff
                    logger.warning(f"Rate limit hit on batch {i//batch_size + 1}, waiting {wait_time}s")
                    time.sleep(wait_time)
                elif attempt < max_retries - 1:
                    wait_time = 2 * (attempt + 1)
                    logger.warning(f"Error embedding batch {i//batch_size + 1}, retrying in {wait_time}s: {e}")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Failed to embed batch {i//batch_size + 1} after {max_retries} attempts: {e}")
                    raise
    
    return all_embeddings

def create_vector_index(vectors):
    dim = len(vectors[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(vectors))
    return index

def search_codex(question):
    question_lower = question.lower()
    matched = []
    for entry in POLICY_CODEX:
        if any(k.lower() in question_lower for k in entry["keywords"]):
            matched.append(entry)
    return matched

def ask_gpt(prompt, max_retries=3, retry_delay=2):
    """Ask GPT with retry logic for reliability"""
    for attempt in range(max_retries):
        try:
            if USE_NEW_API and openai_client:
                response = openai_client.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3
                )
                return response.choices[0].message.content.strip()
            else:
                # Old API fallback
                response = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3
                )
                return response.choices[0].message["content"].strip()
        except Exception as e:
            error_str = str(e).lower()
            if "rate limit" in error_str or "ratelimit" in error_str:
                wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                logger.warning(f"Rate limit hit, waiting {wait_time}s before retry {attempt + 1}/{max_retries}")
                time.sleep(wait_time)
            elif attempt < max_retries - 1:
                wait_time = retry_delay * (attempt + 1)
                logger.warning(f"API error: {e}, retrying in {wait_time}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
            else:
                logger.error(f"API error after {max_retries} attempts: {e}")
                raise
    raise Exception("Failed to get GPT response after all retries")

def detect_language(text, max_retries=2):
    """Detect language with retry logic"""
    for attempt in range(max_retries):
        try:
            if USE_NEW_API and openai_client:
                response = openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": f"What language is this question in? Just reply with one word.\n{text}"}]
                )
                return response.choices[0].message.content.strip().lower()
            else:
                # Old API fallback
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": f"What language is this question in? Just reply with one word.\n{text}"}]
                )
                return response.choices[0].message["content"].strip().lower()
        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning(f"Language detection failed, retrying: {e}")
                time.sleep(1)
            else:
                logger.warning(f"Language detection failed after retries, defaulting to English: {e}")
                return "english"  # Default to English if detection fails

def translate_answer(answer, target_lang, max_retries=2):
    """Translate answer with retry logic"""
    if not answer or len(answer.strip()) == 0:
        return answer
    try:
        prompt = f"Translate the following text to {target_lang}:\n{answer}"
        return ask_gpt(prompt, max_retries=max_retries)
    except Exception as e:
        logger.error(f"Translation failed: {e}, returning original answer")
        return answer  # Return original if translation fails

def get_top_chunks(question, k=15, max_retries=3):
    """Get top chunks with retry logic - increased k for better coverage"""
    if index is None or len(chunks) == 0:
        raise RuntimeError("❌ Vector index not initialized. Please ensure documents are loaded first.")
    
    for attempt in range(max_retries):
        try:
            if USE_NEW_API and openai_client:
                response = openai_client.embeddings.create(
                    model="text-embedding-ada-002",
                    input=[question]
                )
                question_vec = response.data[0].embedding
            else:
                # Old API fallback
                response = openai.Embedding.create(
                    model="text-embedding-ada-002",
                    input=[question]
                )
                question_vec = response["data"][0]["embedding"]
            D, I = index.search(np.array([question_vec], dtype=np.float32), k)
            # Return chunks with distance scores
            results = []
            for j, i in enumerate(I[0]):
                if i < len(chunks):
                    results.append((chunks[i], chunk_sources[i], float(D[0][j])))
            return results
        except Exception as e:
            error_str = str(e).lower()
            if "rate limit" in error_str or "ratelimit" in error_str:
                wait_time = 2 * (2 ** attempt)
                logger.warning(f"Rate limit hit on embedding search, waiting {wait_time}s")
                time.sleep(wait_time)
            elif attempt < max_retries - 1:
                wait_time = 2 * (attempt + 1)
                logger.warning(f"Error getting chunks, retrying in {wait_time}s: {e}")
                time.sleep(wait_time)
            else:
                logger.error(f"Failed to get chunks after {max_retries} attempts: {e}")
                raise
    return []  # Return empty if all retries failed

def keyword_search_chunks(question, chunks_list, chunk_sources_list, top_n=10):
    """Fallback keyword-based search when vector search might miss relevant chunks"""
    question_lower = question.lower()
    question_words = set(question_lower.split())
    
    # Extract important keywords (remove common stop words)
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'does', 'do', 'did', 'can', 'could', 'should', 'would', 'will', 'what', 'when', 'where', 'why', 'how', 'this', 'that', 'these', 'those'}
    important_words = [w for w in question_words if w not in stop_words and len(w) > 2]
    
    if not important_words:
        return []
    
    scored_chunks = []
    for i, (chunk, source) in enumerate(zip(chunks_list, chunk_sources_list)):
        chunk_lower = chunk.lower()
        # Count keyword matches
        matches = sum(1 for word in important_words if word in chunk_lower)
        if matches > 0:
            # Score based on number of matches and word length
            score = matches / len(important_words) + (0.1 if any(word in chunk_lower for word in important_words) else 0)
            scored_chunks.append((chunk, source, score, i))
    
    # Sort by score and return top N
    scored_chunks.sort(key=lambda x: x[2], reverse=True)
    return [(chunk, source) for chunk, source, score, idx in scored_chunks[:top_n]]

def hybrid_search(question, k_vector=15, k_keyword=10):
    """Combine vector search and keyword search for better coverage"""
    vector_results = []
    keyword_results = []
    
    # Try vector search first
    try:
        vector_results = get_top_chunks(question, k=k_vector)
        # Convert to (chunk, source) format if distance included
        if vector_results and len(vector_results[0]) == 3:
            vector_results = [(chunk, source) for chunk, source, dist in vector_results]
    except Exception as e:
        logger.warning(f"Vector search failed: {e}, using keyword search only")
    
    # Always try keyword search as fallback/complement
    try:
        keyword_results = keyword_search_chunks(question, chunks, chunk_sources, top_n=k_keyword)
    except Exception as e:
        logger.warning(f"Keyword search failed: {e}")
    
    # Combine results, prioritizing vector results but including keyword matches
    combined = {}
    # Add vector results with higher priority
    for chunk, source in vector_results:
        key = (chunk[:100], source)  # Use first 100 chars as key to deduplicate
        if key not in combined:
            combined[key] = (chunk, source, 1.0)  # Vector results get priority 1.0
    
    # Add keyword results with lower priority but still include them
    for chunk, source in keyword_results:
        key = (chunk[:100], source)
        if key not in combined:
            combined[key] = (chunk, source, 0.5)  # Keyword results get priority 0.5
        else:
            # If already in vector results, boost its priority
            chunk, source, priority = combined[key]
            combined[key] = (chunk, source, 1.5)  # Boosted priority
    
    # Return sorted by priority, then by chunk length (longer chunks often have more context)
    results = list(combined.values())
    results.sort(key=lambda x: (x[2], len(x[0])), reverse=True)
    
    return [(chunk, source) for chunk, source, priority in results]

def preprocess_question(question, max_retries=2):
    """
    Preprocess and normalize questions to improve understanding.
    Fixes typos, expands abbreviations, and clarifies intent.
    """
    if not question or len(question.strip()) < 3:
        return question
    
    # Quick fixes for common issues
    question = question.strip()
    
    # Common typo fixes
    typo_fixes = {
        "eligble": "eligible",
        "eligibile": "eligible",
        "accepts": "accept",
        "acceptted": "accept",
        "elevat": "elevate",
        "clariy": "clarity",
        "mortagage": "mortgage",
    }
    
    for typo, correct in typo_fixes.items():
        question = question.replace(typo, correct)
    
    # If question is very short or unclear, try to expand it with GPT
    question_lower = question.lower()
    is_very_short = len(question.split()) <= 3
    is_incomplete = not any(q_word in question_lower for q_word in ["is", "does", "what", "can", "will", "are", "do", "how", "when", "where", "why"])
    has_creditor_mention = any(creditor in question_lower for creditor in [
        "oportun", "regional", "mortgage", "auto", "student", "credit union",
        "discover", "amex", "clarity", "elevate"
    ])
    
    # If question needs clarification, use GPT to reformat it
    if (is_very_short or is_incomplete) and has_creditor_mention:
        try:
            clarification_prompt = f"""The user asked: "{question}"

This question is about debt relief programs (Elevate and Clarity). Please reformat it into a clear, complete question that asks about eligibility or acceptance.

Examples:
- "Oportun is eligble?" -> "Is Oportun eligible for Elevate or Clarity programs?"
- "mortgage accepted?" -> "Are mortgage loans accepted in Elevate or Clarity programs?"
- "regional finance" -> "Is Regional Finance accepted in Elevate or Clarity programs?"

Just return the reformatted question, nothing else:"""
            
            if USE_NEW_API and openai_client:
                response = openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": clarification_prompt}],
                    temperature=0.3,
                    max_tokens=50
                )
                clarified = response.choices[0].message.content.strip()
            else:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": clarification_prompt}],
                    temperature=0.3,
                    max_tokens=50
                )
                clarified = response.choices[0].message["content"].strip()
            
            # Remove quotes if GPT added them
            clarified = clarified.strip('"').strip("'")
            if clarified and len(clarified) > len(question) * 0.5:  # Only use if it's reasonable
                logger.info(f"Question clarified: '{question}' -> '{clarified}'")
                return clarified
        except Exception as e:
            logger.warning(f"Question clarification failed: {e}, using original question")
    
    # If question doesn't mention programs, add context
    if has_creditor_mention and "elevate" not in question_lower and "clarity" not in question_lower:
        # Try to make it more specific
        if "?" not in question:
            question = question.rstrip(".") + "?"
        if not question_lower.startswith(("is ", "does ", "are ", "can ", "do ", "what ", "how ")):
            # Add a question starter
            if "eligible" in question_lower or "accept" in question_lower:
                question = f"Is {question}"
            elif "minimum" in question_lower or "maximum" in question_lower:
                question = f"What is the {question}"
    
    return question

def is_valid_primary_chunk(chunk, source, question_keywords=None):
    """
    Check if a chunk is valid for primary document-based answers.
    More lenient validation to avoid filtering out relevant information.
    """
    # Check word count - reduced threshold to include more chunks
    word_count = len(chunk.split())
    if word_count < 3:  # Reduced from 5 to 3
        return False
    
    # Check if source is from relevant policy documents
    source_lower = source.lower()
    
    # Primary program documents
    is_clarity = "clarity" in source_lower or "affiliate_training_packet" in source_lower
    is_elevate = "elevate" in source_lower
    
    # Policy and reference documents - expanded list
    is_policy = any(term in source_lower for term in [
        "disqualified", "unacceptable", "state", "comparison", "list", "criteria", 
        "unacceptablecreditunion", "debt", "program", "enrollment", "training"
    ])
    
    # Check if chunk contains important policy indicators (override word count)
    chunk_lower = chunk.lower()
    has_policy_indicators = any(term in chunk_lower for term in [
        "❌", "✅", "⚠️", "not allowed", "prohibited", "disqualified", "restricted", 
        "mortgage", "secured", "accepted", "eligible", "enrollment", "program"
    ])
    
    # If question keywords provided, check for relevance
    if question_keywords:
        chunk_words = set(chunk_lower.split())
        keyword_matches = sum(1 for kw in question_keywords if kw in chunk_lower)
        if keyword_matches >= 1:  # At least one keyword match
            return True
    
    # Always include chunks with important policy content
    if has_policy_indicators:
        return True
    
    # More lenient: include if from any policy document OR has minimum word count
    return (is_clarity or is_elevate or is_policy) and word_count >= 3

def get_program_sources_from_chunks(chunk_sources):
    """
    Extract program names from chunk sources, only counting Clarity and Elevate.
    """
    programs = set()
    for source in chunk_sources:
        source_lower = source.lower()
        if "clarity" in source_lower or "affiliate_training_packet" in source_lower:
            programs.add("Clarity")
        if "elevate" in source_lower:
            programs.add("Elevate")
    return sorted(list(programs))

def ask_gpt_with_system_prompt(system_prompt, user_prompt, max_retries=3):
    """
    Ask GPT with a specific system prompt and retry logic.
    """
    for attempt in range(max_retries):
        try:
            if USE_NEW_API and openai_client:
                response = openai_client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.3
                )
                return response.choices[0].message.content.strip()
            else:
                # Old API fallback
                response = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.3
                )
                return response.choices[0].message["content"].strip()
        except Exception as e:
            error_str = str(e).lower()
            if "rate limit" in error_str or "ratelimit" in error_str:
                wait_time = 2 * (2 ** attempt)
                logger.warning(f"Rate limit hit, waiting {wait_time}s before retry {attempt + 1}/{max_retries}")
                time.sleep(wait_time)
            elif attempt < max_retries - 1:
                wait_time = 2 * (attempt + 1)
                logger.warning(f"API error: {e}, retrying in {wait_time}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
            else:
                logger.error(f"API error after {max_retries} attempts: {e}")
                raise
    raise Exception("Failed to get GPT response after all retries")

def handle_question(question):
    print(f"🚀 handle_question called with: {question}")
    if not question or not question.strip():
        return "⚠️ Please provide a valid question."
    
    # Step 0: Preprocess and normalize question
    question = preprocess_question(question)
    logger.info(f"📝 Preprocessed question: {question}")
    
    # Step 1: Normalize question
    question_clean = question.lower()
    print(f"🔍 Normalized question: {question_clean}")
    
    # Step 2: Comprehensive hardcoded acceptance/rejection logic
    hard_rejections = {
        # DEBT TYPES - NOT ACCEPTED
        "mortgage": {
            "global": (
                "❌ *Elevate:* Mortgage loans are not accepted.\n"
                "❌ *Clarity:* Mortgage loans are not accepted.\n"
                "📝 *Please inform the client that mortgage loans must be resolved outside the program.*",
                "❌ *Elevate:* Los préstamos hipotecarios no se aceptan.\n"
                "❌ *Clarity:* Los préstamos hipotecarios no se aceptan.\n"
                "📝 *Por favor informe al cliente que los préstamos hipotecarios deben resolverse fuera del programa.*"
            )
        },
        "secured loan": {
            "global": (
                "❌ *Elevate:* Secured loans are not accepted.\n"
                "❌ *Clarity:* Secured loans are not accepted.\n"
                "📝 *Please inform the client that secured loans must be resolved outside the program.*",
                "❌ *Elevate:* Los préstamos con garantía no se aceptan.\n"
                "❌ *Clarity:* Los préstamos con garantía no se aceptan.\n"
                "📝 *Por favor informe al cliente que los préstamos con garantía deben resolverse fuera del programa.*"
            )
        },
        "federal student loan": {
            "global": (
                "❌ *Elevate:* Federal student loans are not accepted.\n"
                "❌ *Clarity:* Federal student loans are not accepted.\n"
                "📝 *Please inform the client that federal student loans must be resolved outside the program.*",
                "❌ *Elevate:* Los préstamos estudiantiles federales no se aceptan.\n"
                "❌ *Clarity:* Los préstamos estudiantiles federales no se aceptan.\n"
                "📝 *Por favor informe al cliente que los préstamos estudiantiles federales deben resolverse fuera del programa.*"
            )
        },
        "auto loan": {
            "global": (
                "❌ *Elevate:* Auto loans are not accepted.\n"
                "❌ *Clarity:* Auto loans are not accepted (except post-repossession deficiencies).\n"
                "📝 *Please inform the client that auto loans must be resolved outside the program.*",
                "❌ *Elevate:* Los préstamos de auto no se aceptan.\n"
                "❌ *Clarity:* Los préstamos de auto no se aceptan (excepto deficiencias post-embargo).\n"
                "📝 *Por favor informe al cliente que los préstamos de auto deben resolverse fuera del programa.*"
            )
        },
        "irs": {
            "global": (
                "❌ *Elevate:* IRS/tax debt is not accepted.\n"
                "❌ *Clarity:* IRS/tax debt is not accepted.\n"
                "📝 *Please inform the client that IRS/tax debt must be resolved outside the program.*",
                "❌ *Elevate:* La deuda del IRS/impuestos no se acepta.\n"
                "❌ *Clarity:* La deuda del IRS/impuestos no se acepta.\n"
                "📝 *Por favor informe al cliente que la deuda del IRS/impuestos debe resolverse fuera del programa.*"
            )
        },
        "judgment": {
            "global": (
                "❌ *Elevate:* Judgments are not accepted.\n"
                "❌ *Clarity:* Judgments are not accepted (unless filed 6+ months ago with no active collection).\n"
                "📝 *Please inform the client that judgments must be resolved outside the program.*",
                "❌ *Elevate:* Los juicios no se aceptan.\n"
                "❌ *Clarity:* Los juicios no se aceptan (a menos que se presentaron hace 6+ meses sin cobro activo).\n"
                "📝 *Por favor informe al cliente que los juicios deben resolverse fuera del programa.*"
            )
        },
        "alimony": {
            "global": (
                "❌ *Elevate:* Alimony/child support is not accepted.\n"
                "❌ *Clarity:* Alimony/child support is not accepted.\n"
                "📝 *Please inform the client that alimony/child support must be resolved outside the program.*",
                "❌ *Elevate:* La pensión alimenticia no se acepta.\n"
                "❌ *Clarity:* La pensión alimenticia no se acepta.\n"
                "📝 *Por favor informe al cliente que la pensión alimenticia debe resolverse fuera del programa.*"
            )
        },
        "gambling": {
            "global": (
                "❌ *Elevate:* Gambling debts are not accepted.\n"
                "❌ *Clarity:* Gambling debts are not accepted.\n"
                "📝 *Please inform the client that gambling debts must be resolved outside the program.*",
                "❌ *Elevate:* Las deudas de juego no se aceptan.\n"
                "❌ *Clarity:* Las deudas de juego no se aceptan.\n"
                "📝 *Por favor informe al cliente que las deudas de juego deben resolverse fuera del programa.*"
            )
        },
        "timeshare": {
            "global": (
                "❌ *Elevate:* Timeshares are not accepted.\n"
                "❌ *Clarity:* Timeshares are not accepted.\n"
                "📝 *Please inform the client that timeshares must be resolved outside the program.*",
                "❌ *Elevate:* Los tiempos compartidos no se aceptan.\n"
                "❌ *Clarity:* Los tiempos compartidos no se aceptan.\n"
                "📝 *Por favor informe al cliente que los tiempos compartidos deben resolverse fuera del programa.*"
            )
        },
        "property tax": {
            "global": (
                "❌ *Elevate:* Property taxes are not accepted.\n"
                "❌ *Clarity:* Property taxes are not accepted.\n"
                "📝 *Please inform the client that property taxes must be resolved outside the program.*",
                "❌ *Elevate:* Los impuestos sobre la propiedad no se aceptan.\n"
                "❌ *Clarity:* Los impuestos sobre la propiedad no se aceptan.\n"
                "📝 *Por favor informe al cliente que los impuestos sobre la propiedad deben resolverse fuera del programa.*"
            )
        },
        "bail bond": {
            "global": (
                "❌ *Elevate:* Bail bonds are not accepted.\n"
                "❌ *Clarity:* Bail bonds are not accepted.\n"
                "📝 *Please inform the client that bail bonds must be resolved outside the program.*",
                "❌ *Elevate:* Las fianzas no se aceptan.\n"
                "❌ *Clarity:* Las fianzas no se aceptan.\n"
                "📝 *Por favor informe al cliente que las fianzas deben resolverse fuera del programa.*"
            )
        },
        
        # SPECIFIC CREDITORS - NOT ACCEPTED
        "ncb": {
            "global": (
                "❌ *Elevate:* NCB Management Services is not accepted.\n"
                "❌ *Clarity:* NCB Management Services is not accepted.\n"
                "📝 *Please inform the client that NCB debts must be resolved outside the program.*",
                "❌ *Elevate:* NCB Management Services no se acepta.\n"
                "❌ *Clarity:* NCB Management Services no se acepta.\n"
                "📝 *Por favor informe al cliente que las deudas de NCB deben resolverse fuera del programa.*"
            )
        },
        "rocket loan": {
            "global": (
                "❌ *Elevate:* Rocket Loans is not accepted.\n"
                "❌ *Clarity:* Rocket Loans is not accepted.\n"
                "📝 *Please inform the client that Rocket Loans must be resolved outside the program.*",
                "❌ *Elevate:* Rocket Loans no se acepta.\n"
                "❌ *Clarity:* Rocket Loans no se acepta.\n"
                "📝 *Por favor informe al cliente que Rocket Loans debe resolverse fuera del programa.*"
            )
        },
        "goodleap": {
            "global": (
                "❌ *Elevate:* GoodLeap is not accepted.\n"
                "❌ *Clarity:* GoodLeap is not accepted.\n"
                "📝 *Please inform the client that GoodLeap must be resolved outside the program.*",
                "❌ *Elevate:* GoodLeap no se acepta.\n"
                "❌ *Clarity:* GoodLeap no se acepta.\n"
                "📝 *Por favor informe al cliente que GoodLeap debe resolverse fuera del programa.*"
            )
        },
        "military star": {
            "global": (
                "❌ *Elevate:* Military Star is not accepted.\n"
                "❌ *Clarity:* Military Star is not accepted.\n"
                "📝 *Please inform the client that Military Star must be resolved outside the program.*",
                "❌ *Elevate:* Military Star no se acepta.\n"
                "❌ *Clarity:* Military Star no se acepta.\n"
                "📝 *Por favor informe al cliente que Military Star debe resolverse fuera del programa.*"
            )
        },
        "tower loan": {
            "global": (
                "❌ *Elevate:* Tower Loan is not accepted.\n"
                "❌ *Clarity:* Tower Loan is not accepted.\n"
                "📝 *Please inform the client that Tower Loan must be resolved outside the program.*",
                "❌ *Elevate:* Tower Loan no se acepta.\n"
                "❌ *Clarity:* Tower Loan no se acepta.\n"
                "📝 *Por favor informe al cliente que Tower Loan debe resolverse fuera del programa.*"
            )
        },
        "aqua finance": {
            "global": (
                "❌ *Elevate:* Aqua Finance is not accepted.\n"
                "❌ *Clarity:* Aqua Finance is not accepted.\n"
                "📝 *Please inform the client that Aqua Finance must be resolved outside the program.*",
                "❌ *Elevate:* Aqua Finance no se acepta.\n"
                "❌ *Clarity:* Aqua Finance no se acepta.\n"
                "📝 *Por favor informe al cliente que Aqua Finance debe resolverse fuera del programa.*"
            )
        },
        "pentagon": {
            "global": (
                "❌ *Elevate:* Pentagon FCU installment loans are not accepted (credit cards only).\n"
                "❌ *Clarity:* Pentagon FCU installment loans are not accepted.\n"
                "📝 *Please inform the client that Pentagon FCU installment loans must be resolved outside the program.*",
                "❌ *Elevate:* Los préstamos a plazos de Pentagon FCU no se aceptan (solo tarjetas de crédito).\n"
                "❌ *Clarity:* Los préstamos a plazos de Pentagon FCU no se aceptan.\n"
                "📝 *Por favor informe al cliente que los préstamos a plazos de Pentagon FCU deben resolverse fuera del programa.*"
            )
        },
        "koalafi": {
            "global": (
                "❌ *Elevate:* KOALAFI is not accepted.\n"
                "❌ *Clarity:* KOALAFI is not accepted.\n"
                "📝 *Please inform the client that KOALAFI must be resolved outside the program.*",
                "❌ *Elevate:* KOALAFI no se acepta.\n"
                "❌ *Clarity:* KOALAFI no se acepta.\n"
                "📝 *Por favor informe al cliente que KOALAFI debe resolverse fuera del programa.*"
            )
        },
        "republic finance": {
            "global": (
                "❌ *Elevate:* Republic Finance is not accepted.\n"
                "❌ *Clarity:* Republic Finance is not accepted.\n"
                "📝 *Please inform the client that Republic Finance must be resolved outside the program.*",
                "❌ *Elevate:* Republic Finance no se acepta.\n"
                "❌ *Clarity:* Republic Finance no se acepta.\n"
                "📝 *Por favor informe al cliente que Republic Finance debe resolverse fuera del programa.*"
            )
        },
        "snap tools": {
            "global": (
                "❌ *Elevate:* Snap Tools is not accepted.\n"
                "❌ *Clarity:* Snap Tools is not accepted.\n"
                "📝 *Please inform the client that Snap Tools must be resolved outside the program.*",
                "❌ *Elevate:* Snap Tools no se acepta.\n"
                "❌ *Clarity:* Snap Tools no se acepta.\n"
                "📝 *Por favor informe al cliente que Snap Tools debe resolverse fuera del programa.*"
            )
        },
        "cnh": {
            "global": (
                "❌ *Elevate:* CNH Industrial is not accepted.\n"
                "❌ *Clarity:* CNH Industrial is not accepted.\n"
                "📝 *Please inform the client that CNH Industrial must be resolved outside the program.*",
                "❌ *Elevate:* CNH Industrial no se acepta.\n"
                "❌ *Clarity:* CNH Industrial no se acepta.\n"
                "📝 *Por favor informe al cliente que CNH Industrial debe resolverse fuera del programa.*"
            )
        },
        "duvera": {
            "global": (
                "❌ *Elevate:* Duvera Finance is not accepted.\n"
                "❌ *Clarity:* Duvera Finance is not accepted.\n"
                "📝 *Please inform the client that Duvera Finance must be resolved outside the program.*",
                "❌ *Elevate:* Duvera Finance no se acepta.\n"
                "❌ *Clarity:* Duvera Finance no se acepta.\n"
                "📝 *Por favor informe al cliente que Duvera Finance debe resolverse fuera del programa.*"
            )
        },
        "grt american": {
            "global": (
                "❌ *Elevate:* GRT American Financial is not accepted.\n"
                "❌ *Clarity:* GRT American Financial is not accepted.\n"
                "📝 *Please inform the client that GRT American Financial must be resolved outside the program.*",
                "❌ *Elevate:* GRT American Financial no se acepta.\n"
                "❌ *Clarity:* GRT American Financial no se acepta.\n"
                "📝 *Por favor informe al cliente que GRT American Financial debe resolverse fuera del programa.*"
            )
        },
        "service finance": {
            "global": (
                "❌ *Elevate:* Service Finance is not accepted.\n"
                "❌ *Clarity:* Service Finance is not accepted.\n"
                "📝 *Please inform the client that Service Finance must be resolved outside the program.*",
                "❌ *Elevate:* Service Finance no se acepta.\n"
                "❌ *Clarity:* Service Finance no se acepta.\n"
                "📝 *Por favor informe al cliente que Service Finance debe resolverse fuera del programa.*"
            )
        },
        "schools first": {
            "global": (
                "❌ *Elevate:* Schools First CU loans are not accepted (credit cards only).\n"
                "❌ *Clarity:* Schools First CU loans are not accepted.\n"
                "📝 *Please inform the client that Schools First CU loans must be resolved outside the program.*",
                "❌ *Elevate:* Los préstamos de Schools First CU no se aceptan (solo tarjetas de crédito).\n"
                "❌ *Clarity:* Los préstamos de Schools First CU no se aceptan.\n"
                "📝 *Por favor informe al cliente que los préstamos de Schools First CU deben resolverse fuera del programa.*"
            )
        },
        "nebraska furniture": {
            "global": (
                "❌ *Elevate:* Nebraska Furniture is not accepted.\n"
                "❌ *Clarity:* Nebraska Furniture is not accepted.\n"
                "📝 *Please inform the client that Nebraska Furniture must be resolved outside the program.*",
                "❌ *Elevate:* Nebraska Furniture no se acepta.\n"
                "❌ *Clarity:* Nebraska Furniture no se acepta.\n"
                "📝 *Por favor informe al cliente que Nebraska Furniture debe resolverse fuera del programa.*"
            )
        },
        "aaron": {
            "global": (
                "❌ *Elevate:* Aaron's Rent is not accepted.\n"
                "❌ *Clarity:* Aaron's Rent is not accepted.\n"
                "📝 *Please inform the client that Aaron's Rent must be resolved outside the program.*",
                "❌ *Elevate:* Aaron's Rent no se acepta.\n"
                "❌ *Clarity:* Aaron's Rent no se acepta.\n"
                "📝 *Por favor informe al cliente que Aaron's Rent debe resolverse fuera del programa.*"
            )
        },
        "sofi": {
            "global": (
                "❌ *Elevate:* SoFi is not accepted if federally backed.\n"
                "❌ *Clarity:* SoFi is not accepted if federally backed.\n"
                "📝 *Please inform the client that SoFi must be resolved outside the program.*",
                "❌ *Elevate:* SoFi no se acepta si está respaldado federalmente.\n"
                "❌ *Clarity:* SoFi no se acepta si está respaldado federalmente.\n"
                "📝 *Por favor informe al cliente que SoFi debe resolverse fuera del programa.*"
            )
        },
        "rc willey": {
            "global": (
                "❌ *Elevate:* RC Willey is not accepted.\n"
                "❌ *Clarity:* RC Willey is not accepted.\n"
                "📝 *Please inform the client that RC Willey must be resolved outside the program.*",
                "❌ *Elevate:* RC Willey no se acepta.\n"
                "❌ *Clarity:* RC Willey no se acepta.\n"
                "📝 *Por favor informe al cliente que RC Willey debe resolverse fuera del programa.*"
            )
        },
        "fortiva": {
            "global": (
                "❌ *Elevate:* Fortiva is not accepted.\n"
                "❌ *Clarity:* Fortiva is not accepted.\n"
                "📝 *Please inform the client that Fortiva must be resolved outside the program.*",
                "❌ *Elevate:* Fortiva no se acepta.\n"
                "❌ *Clarity:* Fortiva no se acepta.\n"
                "📝 *Por favor informe al cliente que Fortiva debe resolverse fuera del programa.*"
            )
        },
        "omni financial": {
            "global": (
                "❌ *Elevate:* OMNI Financial is not accepted.\n"
                "❌ *Clarity:* OMNI Financial is not accepted.\n"
                "📝 *Please inform the client that OMNI Financial must be resolved outside the program.*",
                "❌ *Elevate:* OMNI Financial no se acepta.\n"
                "❌ *Clarity:* OMNI Financial no se acepta.\n"
                "📝 *Por favor informe al cliente que OMNI Financial debe resolverse fuera del programa.*"
            )
        },
        "srvfinco": {
            "global": (
                "❌ *Elevate:* SRVFINCO is not accepted.\n"
                "❌ *Clarity:* SRVFINCO is not accepted.\n"
                "📝 *Please inform the client that SRVFINCO must be resolved outside the program.*",
                "❌ *Elevate:* SRVFINCO no se acepta.\n"
                "❌ *Clarity:* SRVFINCO no se acepta.\n"
                "📝 *Por favor informe al cliente que SRVFINCO debe resolverse fuera del programa.*"
            )
        },
        "bhg": {
            "global": (
                "❌ *Elevate:* BHG Bankers Healthcare Group is not accepted.\n"
                "❌ *Clarity:* BHG Bankers Healthcare Group is not accepted.\n"
                "📝 *Please inform the client that BHG must be resolved outside the program.*",
                "❌ *Elevate:* BHG Bankers Healthcare Group no se acepta.\n"
                "❌ *Clarity:* BHG Bankers Healthcare Group no se acepta.\n"
                "📝 *Por favor informe al cliente que BHG debe resolverse fuera del programa.*"
            )
        },
        "mariner finance": {
            "global": (
                "❌ *Elevate:* Mariner Finance is not accepted.\n"
                "❌ *Clarity:* Mariner Finance is not accepted.\n"
                "📝 *Please inform the client that Mariner Finance must be resolved outside the program.*",
                "❌ *Elevate:* Mariner Finance no se acepta.\n"
                "❌ *Clarity:* Mariner Finance no se acepta.\n"
                "📝 *Por favor informe al cliente que Mariner Finance debe resolverse fuera del programa.*"
            )
        },
        "security finance": {
            "global": (
                "❌ *Elevate:* Security Finance is not accepted.\n"
                "❌ *Clarity:* Security Finance is not accepted.\n"
                "📝 *Please inform the client that Security Finance must be resolved outside the program.*",
                "❌ *Elevate:* Security Finance no se acepta.\n"
                "❌ *Clarity:* Security Finance no se acepta.\n"
                "📝 *Por favor informe al cliente que Security Finance debe resolverse fuera del programa.*"
            )
        },
        "pioneer credit": {
            "global": (
                "❌ *Elevate:* Pioneer Credit is not accepted.\n"
                "❌ *Clarity:* Pioneer Credit is not accepted.\n"
                "📝 *Please inform the client that Pioneer Credit must be resolved outside the program.*",
                "❌ *Elevate:* Pioneer Credit no se acepta.\n"
                "❌ *Clarity:* Pioneer Credit no se acepta.\n"
                "📝 *Por favor informe al cliente que Pioneer Credit debe resolverse fuera del programa.*"
            )
        },
        "world finance": {
            "global": (
                "❌ *Elevate:* World Finance is not accepted.\n"
                "❌ *Clarity:* World Finance is not accepted.\n"
                "📝 *Please inform the client that World Finance must be resolved outside the program.*",
                "❌ *Elevate:* World Finance no se acepta.\n"
                "❌ *Clarity:* World Finance no se acepta.\n"
                "📝 *Por favor informe al cliente que World Finance debe resolverse fuera del programa.*"
            )
        },
        
        # CONDITIONAL ACCEPTANCE
        "oportun": {
            "california": (
                "❌ *Elevate:* Oportun is not accepted in California.\n"
                "❌ *Clarity:* Oportun is not accepted in California.\n"
                "📝 *Please inform the client that this debt must be resolved outside the program.*",
                "❌ *Elevate:* Oportun no se acepta en California.\n"
                "❌ *Clarity:* Oportun no se acepta en California.\n"
                "📝 *Por favor informe al cliente que esta deuda debe resolverse fuera del programa.*"
            ),
            "global": (
                "✅ *Elevate:* Oportun is accepted (max 25% of total debt).\n"
                "✅ *Clarity:* Oportun is accepted (no cap stated).\n"
                "📝 *Please ensure client meets all other program criteria.*",
                "✅ *Elevate:* Oportun se acepta (máx 25% de la deuda total).\n"
                "✅ *Clarity:* Oportun se acepta (sin límite establecido).\n"
                "📝 *Por favor asegúrese de que el cliente cumpla con todos los demás criterios del programa.*"
            )
        },
        "regional finance": {
            "global": (
                "❌ *Elevate:* Regional Finance is not accepted.\n"
                "✅ *Clarity:* Regional Finance is accepted if unsecured and meets standard criteria.\n"
                "📝 *Please check specific program requirements.*",
                "❌ *Elevate:* Regional Finance no se acepta.\n"
                "✅ *Clarity:* Regional Finance se acepta si es sin garantía y cumple con los criterios estándar.\n"
                "📝 *Por favor verifique los requisitos específicos del programa.*"
            )
        }
    }
    
    # Check for hardcoded rejections
    for creditor, conditions in hard_rejections.items():
        if creditor in question_clean:
            print(f"🔍 Found creditor: {creditor}")
            for condition, (eng_msg, spa_msg) in conditions.items():
                print(f"🔍 Checking condition: {condition}")
                print(f"🔍 Question contains 'california': {'california' in question_clean}")
                print(f"🔍 Question contains 'ca': {'ca' in question_clean}")
                print(f"🔍 Full question: {question_clean}")
                if condition == "global" or (condition in question_clean or "ca" in question_clean):
                    print(f"🔒 Hardcoded rejection triggered for {creditor} + {condition}")
                    return f"💬 *Answer (English):*\n{eng_msg}\n\n💬 *Respuesta (Spanish):*\n{spa_msg}"
                else:
                    print(f"❌ Condition not met: {condition} not in question and not 'ca'")
    
    # Step 3: Global disqualification check (additional creditors not in hardcoded rules)
    global_disqualified = [
        "accion usa", "diamond resorts", "cashnetusa", "advance financial", "armed forces bank",
        "army navy exchange", "ashley furniture", "avio credit", "b&f finance", "bannerbank",
        "blue green corp", "cc flow", "christianccu", "commonwealth cu", "conns credit",
        "cornwell tools", "credit america", "crest financial", "diamond resorts", "duvera finance",
        "educators cu", "enerbank", "founders fcu", "future income payments", "gecrb", "intermountain healthcare",
        "ispc", "john deere", "karrot loans", "lending usa", "lendmark", "loanmart", "loanosity",
        "mac credit", "mahindra finance", "mcservices", "monterey collections", "nasa fcu",
        "new credit america", "orange lake", "paramount", "payday loans", "qualstar cu",
        "schewels furniture", "snap tools", "spteachercu", "starwood vacation", "superior financial group",
        "teachers cu", "tempoe llc", "texans credit corp", "time investments", "tribal loans",
        "tsi trans world systems", "veridian credit union", "virginia cu", "webbank", "welk resort group",
        "wf/bobsfurniture", "wilshire commercial", "wilson b&t", "world acceptance corporation"
    ]
    
    for keyword in global_disqualified:
        if keyword in question_clean:
            eng = (
                "❌ *Elevate:* This creditor is disqualified and not eligible under any circumstances.\n"
                "❌ *Clarity:* This creditor is disqualified based on policy documents.\n"
                "📝 *Please advise the client to resolve this debt outside the program.*"
            )
            spa = translate_answer(eng, "spanish")
            return f"💬 *Answer (English):*\n{eng}\n\n💬 *Respuesta (Spanish):*\n{spa}"

    # Step 4: Hybrid search - combine vector and keyword search for better coverage
    try:
        # Extract keywords from question for relevance checking
        question_words = set(question_clean.split())
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'does', 'do', 'did', 'can', 'could', 'should', 'would', 'will', 'what', 'when', 'where', 'why', 'how', 'this', 'that', 'these', 'those', 'does', 'elevate', 'clarity', 'accept', 'accepts'}
        question_keywords = [w for w in question_words if w not in stop_words and len(w) > 2]
        
        # Use hybrid search to get more comprehensive results
        top_chunks = hybrid_search(question, k_vector=20, k_keyword=15)
        logger.info(f"Found {len(top_chunks)} chunks from hybrid search")
        
        # More lenient validation with keyword relevance
        valid_chunks = []
        for chunk, src in top_chunks:
            if is_valid_primary_chunk(chunk, src, question_keywords):
                valid_chunks.append((chunk, src))
        
        logger.info(f"After validation: {len(valid_chunks)} valid chunks")
        
        # If still no valid chunks, try even more lenient approach
        if not valid_chunks and top_chunks:
            logger.warning("No chunks passed strict validation, using top chunks anyway")
            # Use top 10 chunks even if they don't pass strict validation
            valid_chunks = top_chunks[:10]
        
    except RuntimeError as e:
        logger.error(f"Error retrieving chunks: {str(e)}")
        eng = (
            "⚠️ *Elevate:* Document search is not available.\n"
            "⚠️ *Clarity:* Document search is not available.\n"
            "📝 *Please contact support for assistance.*"
        )
        spa = translate_answer(eng, "spanish")
        return f"💬 *Answer (English):*\n{eng}\n\n💬 *Respuesta (Spanish):*\n{spa}"
    except Exception as e:
        logger.error(f"Unexpected error in chunk retrieval: {e}\n{traceback.format_exc()}")
        eng = (
            "⚠️ *Elevate:* Document search encountered an error.\n"
            "⚠️ *Clarity:* Document search encountered an error.\n"
            "📝 *Please try rephrasing your question or contact support.*"
        )
        spa = translate_answer(eng, "spanish")
        return f"💬 *Answer (English):*\n{eng}\n\n💬 *Respuesta (Spanish):*\n{spa}"
    
    # Check if we have valid context
    if not valid_chunks:
        logger.warning(f"No valid chunks found for question: {question}")
        eng = (
            "⚠️ *Elevate:* No specific information found in policy documents.\n"
            "⚠️ *Clarity:* No specific information found in policy documents.\n"
            "📝 *Please consult the latest program guidelines or contact support for assistance.*"
        )
        spa = translate_answer(eng, "spanish")
        return f"💬 *Answer (English):*\n{eng}\n\n💬 *Respuesta (Spanish):*\n{spa}"
    
    # Use top 10 chunks for context (increased from 5)
    context_chunks = valid_chunks[:10]
    context = "\n\n".join(f"[Source: {src}]\n{chunk}" for chunk, src in context_chunks)
    logger.info(f"Using {len(context_chunks)} chunks for context")

    # Step 5: Create enhanced system prompt with explicit instructions
    system_prompt = (
        "You are an expert in Elevate and Clarity debt relief programs. "
        "Your task is to answer questions based STRICTLY on the provided document chunks.\n\n"
        
        "CRITICAL INSTRUCTIONS:\n"
        "1. **ALWAYS respond in ENGLISH** - Your answer will be translated to Spanish automatically.\n"
        "2. Read ALL provided document chunks carefully - information may be in any of them.\n"
        "3. Look for EXACT matches to the question terms, synonyms, and related concepts.\n"
        "4. If information exists in the chunks, you MUST provide it - do not say 'not found' if it's actually there.\n"
        "5. Always answer for *both* Elevate and Clarity programs, even if the question mentions only one.\n"
        "6. Use ✅ for accepted/eligible, ❌ for not accepted/ineligible, ⚠️ for conditional/uncertain.\n"
        "7. Quote specific details from the chunks when possible (e.g., percentages, dollar amounts, conditions).\n\n"
        
        "ANSWER FORMAT (IN ENGLISH):\n"
        "- Start with a clear yes/no/conditional answer for each program\n"
        "- Provide specific details from the documents\n"
        "- Include any conditions, limits, or restrictions mentioned\n"
        "- Use emojis for visual clarity (✅ ❌ ⚠️)\n"
        "- Format: *Elevate:* [answer] and *Clarity:* [answer]\n\n"
        
        "SPECIAL CASES:\n"
        "- If question mentions a specific creditor, check ALL chunks for that creditor name (including variations)\n"
        "- If question mentions a state, check for state-specific rules in the chunks\n"
        "- If question asks about debt types (mortgage, auto loan, etc.), search for those exact terms\n"
        "- If information is partially available, provide what you found and note what's missing\n\n"
        
        "IMPORTANT: If the answer is clearly stated in the provided chunks, you MUST provide it. "
        "Only say 'not found' if you have thoroughly searched ALL chunks and the information is genuinely absent. "
        "Be thorough - information might be phrased differently than the question asks.\n\n"
        
        "REMEMBER: Respond ONLY in English. Do not use Spanish in your response."
    )
    # Ensure question is in English for processing (translate if needed)
    question_lang = detect_language(question)
    if question_lang == "spanish":
        logger.info("Question is in Spanish, translating to English for processing")
        question_en_for_gpt = translate_answer(question, "english", max_retries=1)
        if not question_en_for_gpt or len(question_en_for_gpt.strip()) < 3:
            question_en_for_gpt = question  # Fallback to original if translation fails
    else:
        question_en_for_gpt = question
    
    user_prompt = f"DOCUMENTS:\n{context}\n\nQUESTION (answer in ENGLISH):\n{question_en_for_gpt}"

    answer_en = ask_gpt_with_system_prompt(system_prompt, user_prompt)
    
    # Verify answer is in English (if it's in Spanish, translate it back)
    answer_lang = detect_language(answer_en)
    if answer_lang == "spanish":
        logger.warning("GPT returned answer in Spanish, this should not happen. Answer will be used as-is.")
        # The answer is already in Spanish, so use it for Spanish section and translate to English
        answer_es = answer_en
        answer_en = translate_answer(answer_en, "english", max_retries=1)
        if not answer_en or len(answer_en.strip()) < 3:
            answer_en = "⚠️ Error: Could not generate English answer. Please try again."
    else:
        # Normal flow: answer is in English, translate to Spanish
        answer_es = translate_answer(answer_en, "spanish")
    
    return f"💬 *Answer (English):*\n{answer_en}\n\n💬 *Respuesta (Spanish):*\n{answer_es}"

def respond(channel, thread_ts, user_mention, question):
    """Respond to user question with comprehensive error handling"""
    try:
        # Send processing message
        try:
            client.chat_postMessage(
                channel=channel,
                thread_ts=thread_ts,
                text=f"🔍 Processing your question, {user_mention}..."
            )
        except SlackApiError as e:
            logger.error(f"Failed to send processing message: {e}")
            # Continue anyway, try to answer the question
        
        # Process question
        try:
            lang = detect_language(question)
            if lang == "spanish":
                question_en = translate_answer(question, "english")
            else:
                question_en = question
            answer = handle_question(question_en)
        except Exception as e:
            logger.error(f"Error processing question: {e}\n{traceback.format_exc()}")
            answer = (
                "⚠️ *Sorry, I encountered an error processing your question.*\n"
                "📝 *Please try rephrasing your question or contact support if the issue persists.*"
            )
        
        # Send answer
        try:
            client.chat_postMessage(channel=channel, thread_ts=thread_ts, text=answer)
        except SlackApiError as e:
            logger.error(f"Failed to send answer: {e}")
            # Try to send error message
            try:
                client.chat_postMessage(
                    channel=channel,
                    thread_ts=thread_ts,
                    text="⚠️ *I processed your question but couldn't send the response. Please try again.*"
                )
            except:
                logger.error("Failed to send error message to Slack")
                
    except Exception as e:
        logger.error(f"Unexpected error in respond(): {e}\n{traceback.format_exc()}")
        # Last resort: try to send error message
        try:
            client.chat_postMessage(
                channel=channel,
                thread_ts=thread_ts,
                text=f"❌ *Sorry, I encountered an unexpected error. Please try again later.*"
            )
        except:
            logger.error("Complete failure: Could not send any message to Slack")

@app.event("app_mention")
def handle_app_mention_events(body, event, say):
    """Handle when bot is mentioned with @docgpt - mentions only, no DMs"""
    text = event.get("text", "")
    channel = event["channel"]
    thread_ts = event.get("ts")
    user_mention = f"<@{event.get('user')}>"
    
    # Remove bot mention from text (e.g., "<@U123456> your question" -> "your question")
    # Remove any @ mentions from the text
    import re
    text = re.sub(r'<@[A-Z0-9]+>', '', text).strip()
    
    if text:  # Only respond if there's actual text after removing mentions
        respond(channel, thread_ts, user_mention, text)
    else:
        # If just mentioned without a question, send a helpful message
        try:
            client.chat_postMessage(
                channel=channel,
                thread_ts=thread_ts,
                text=f"Hi {user_mention}! 👋 Ask me a question about Elevate or Clarity programs.\n\nExample: `@docgpt Does Elevate accept mortgage loans?`"
            )
        except Exception as e:
            logger.error(f"Failed to send help message: {e}")

if __name__ == "__main__":
    logger.info("🚀 Starting Slack DocGPT bot with codex and document fallback...")
    try:
        chunks, chunk_sources = load_documents()
        if len(chunks) == 0:
            raise ValueError("❌ No document chunks loaded. Please check documents folder.")
        logger.info(f"📚 Loaded {len(chunks)} chunks from documents.")
        vectors = embed_chunks(chunks)
        index = create_vector_index(vectors)
        logger.info("✅ Bot is ready and connected to Slack!")
        SocketModeHandler(app, SLACK_APP_TOKEN).start()
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"❌ Failed to start bot: {str(e)}\n{traceback.format_exc()}")
        raise
