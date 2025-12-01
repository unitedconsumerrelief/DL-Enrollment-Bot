#!/usr/bin/env python3
"""
Local testing script for Slack bot - tests without Slack connection
Run this to test document search improvements before deploying
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import bot functions (but not the Slack app initialization)
import openai
import faiss
import numpy as np
# Import the bot module to access its functions and globals
import slack_doc_bot
from slack_doc_bot import (
    load_documents, 
    embed_chunks, 
    create_vector_index,
    hybrid_search,
    is_valid_primary_chunk,
    handle_question,
    ask_gpt_with_system_prompt,
    translate_answer
)

# Set up OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("❌ OPENAI_API_KEY not found in .env file")
    sys.exit(1)
openai.api_key = OPENAI_API_KEY

# Global variables (same as bot)
index = None
chunks = []
chunk_sources = []

def initialize_bot():
    """Initialize the bot's document index (same as production)"""
    global index, chunks, chunk_sources
    
    print("🚀 Initializing bot for local testing...")
    print("=" * 60)
    
    try:
        # Load documents
        chunks, chunk_sources = load_documents()
        if len(chunks) == 0:
            print("❌ No document chunks loaded!")
            return False
        
        print(f"✅ Loaded {len(chunks)} chunks from documents")
        
        # Create embeddings
        print("🔢 Creating embeddings (this may take a minute)...")
        vectors = embed_chunks(chunks)
        
        # Create index
        index = create_vector_index(vectors)
        
        # Make chunks and index available globally for hybrid_search
        # Update the global variables in the slack_doc_bot module
        slack_doc_bot.chunks = chunks
        slack_doc_bot.chunk_sources = chunk_sources
        slack_doc_bot.index = index
        
        print(f"   Index created with {len(chunks)} chunks")
        
        print("✅ Bot initialized and ready for testing!")
        print("=" * 60)
        print("\n💡 Tip: If the window closes too quickly, run: test_bot_local.bat")
        return True
        
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_question(question, show_chunks=False):
    """Test a question and show results"""
    print(f"\n{'='*60}")
    print(f"❓ QUESTION: {question}")
    print(f"{'='*60}")
    
    try:
        # Show hybrid search results
        print("\n🔍 Running hybrid search...")
        top_chunks = hybrid_search(question, k_vector=20, k_keyword=15)
        print(f"   Found {len(top_chunks)} chunks from hybrid search")
        
        # Show validation
        question_words = set(question.lower().split())
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'does', 'do', 'did', 'can', 'could', 'should', 'would', 'will', 'what', 'when', 'where', 'why', 'how', 'this', 'that', 'these', 'those', 'does', 'elevate', 'clarity', 'accept', 'accepts'}
        question_keywords = [w for w in question_words if w not in stop_words and len(w) > 2]
        
        valid_chunks = []
        for chunk, src in top_chunks:
            if is_valid_primary_chunk(chunk, src, question_keywords):
                valid_chunks.append((chunk, src))
        
        print(f"   After validation: {len(valid_chunks)} valid chunks")
        
        if show_chunks and valid_chunks:
            print(f"\n📋 Top {min(5, len(valid_chunks))} chunks that will be used:")
            for i, (chunk, src) in enumerate(valid_chunks[:5], 1):
                print(f"\n   Chunk {i} (Source: {src}):")
                print(f"   {chunk[:200]}..." if len(chunk) > 200 else f"   {chunk}")
        
        # Get full answer using handle_question
        print("\n🤖 Generating answer...")
        answer = handle_question(question)
        
        print(f"\n💬 ANSWER:")
        print("-" * 60)
        print(answer)
        print("-" * 60)
        
        return answer
        
    except Exception as e:
        print(f"\n❌ Error processing question: {e}")
        import traceback
        traceback.print_exc()
        return None

def interactive_mode():
    """Interactive testing mode"""
    print("\n" + "="*60)
    print("🧪 INTERACTIVE TESTING MODE")
    print("="*60)
    print("Enter questions to test (type 'quit' or 'exit' to stop)")
    print("Type 'chunks' before your question to show retrieved chunks")
    print("="*60)
    
    show_chunks = False
    
    while True:
        try:
            question = input("\n❓ Your question: ").strip()
            
            if not question:
                continue
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Exiting test mode. Goodbye!")
                break
            
            if question.lower() == 'chunks':
                show_chunks = True
                print("✅ Chunk display enabled. Enter your question:")
                continue
            
            if question.lower() == 'nochunks':
                show_chunks = False
                print("✅ Chunk display disabled.")
                continue
            
            test_question(question, show_chunks=show_chunks)
            show_chunks = False  # Reset after each question
            
        except EOFError:
            print("\n\n⚠️ Input stream closed. Exiting test mode.")
            break
        except KeyboardInterrupt:
            print("\n\n👋 Exiting test mode. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()

def batch_test_mode():
    """Test with predefined questions"""
    test_questions = [
        "Does Elevate accept mortgage loans?",
        "Does Clarity accept mortgage loans?",
        "What is the minimum payment for Elevate?",
        "Is Oportun accepted in California?",
        "What creditors are disqualified?",
        "Does Elevate accept auto loans?",
        "What is the minimum debt amount for Clarity?",
        "Is Regional Finance accepted?",
    ]
    
    print("\n" + "="*60)
    print("🧪 BATCH TEST MODE - Testing predefined questions")
    print("="*60)
    
    results = []
    for i, question in enumerate(test_questions, 1):
        print(f"\n\n[Test {i}/{len(test_questions)}]")
        answer = test_question(question, show_chunks=False)
        results.append((question, answer))
    
    print("\n" + "="*60)
    print("📊 BATCH TEST SUMMARY")
    print("="*60)
    for i, (question, answer) in enumerate(results, 1):
        status = "✅" if answer and "not found" not in answer.lower() and "not available" not in answer.lower() else "⚠️"
        print(f"{status} Test {i}: {question[:50]}...")
    
    print("\n✅ Batch testing complete!")
    print("\nPress Enter to exit...")
    try:
        input()
    except (EOFError, KeyboardInterrupt):
        pass

if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║     Slack Bot Local Testing (No Slack Required)          ║
    ║     Tests document search improvements                    ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    try:
        # Initialize bot
        if not initialize_bot():
            print("\n❌ Failed to initialize bot. Please check your configuration.")
            print("\nPress Enter to exit...")
            try:
                input()
            except:
                pass
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n👋 Initialization cancelled. Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error during initialization: {e}")
        import traceback
        traceback.print_exc()
        print("\nPress Enter to exit...")
        try:
            input()
        except:
            pass
        sys.exit(1)
    
    # Ask user for mode
    print("\nSelect testing mode:")
    print("1. Interactive mode (enter questions one by one)")
    print("2. Batch test mode (test predefined questions)")
    print("3. Quick test (test one question and exit)")
    
    try:
        choice = input("\nEnter choice (1, 2, or 3, default=1): ").strip()
        
        if not choice:
            choice = "1"  # Default to interactive
        
        if choice == "2":
            batch_test_mode()
        elif choice == "3":
            # Quick test mode
            test_q = input("\nEnter a test question: ").strip()
            if test_q:
                test_question(test_q, show_chunks=True)
            else:
                print("No question provided. Exiting.")
        else:
            # Default to interactive mode
            interactive_mode()
            
    except EOFError:
        # Handle case where input is not available (like in some IDEs)
        print("\n⚠️ Input not available. Running batch test mode instead...")
        try:
            batch_test_mode()
        except Exception as e:
            print(f"Error in batch mode: {e}")
    except KeyboardInterrupt:
        print("\n\n👋 Exiting. Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    # Keep window open at the end
    print("\n" + "="*60)
    print("Testing complete!")
    print("="*60)
    print("\nPress Enter to exit...")
    try:
        input()
    except:
        pass

