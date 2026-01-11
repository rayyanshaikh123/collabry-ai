"""
Test RAG with a real PDF file - End-to-End CLI Test

This script:
1. Takes a PDF file path as input
2. Extracts text from the PDF
3. Ingests it into RAG system
4. Launches interactive CLI where you can ask questions about the PDF
5. Verifies responses use PDF content only

Usage:
  python test_pdf_rag.py path/to/your/file.pdf
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import argparse
import json
from core.agent import create_agent
from core.rag_retriever import RAGRetriever
from langchain_core.documents import Document
from config import CONFIG

# Test configuration
TEST_USER = "pdf_test_user"
TEST_SESSION = "pdf_test_session"

def extract_pdf_text(pdf_path):
    """Extract text from PDF file."""
    try:
        import PyPDF2
        
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text_parts = []
            
            print(f"📄 Extracting text from PDF: {pdf_path}")
            print(f"   Pages: {len(pdf_reader.pages)}")
            
            for i, page in enumerate(pdf_reader.pages, 1):
                text = page.extract_text()
                if text.strip():
                    text_parts.append(text)
                    print(f"   ✓ Extracted page {i} ({len(text)} chars)")
            
            full_text = "\n\n".join(text_parts)
            print(f"\n✓ Total text extracted: {len(full_text)} characters\n")
            
            return full_text
            
    except ImportError:
        print("❌ PyPDF2 not installed. Installing now...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "PyPDF2"])
        print("✓ PyPDF2 installed. Please run the script again.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error extracting PDF: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def ingest_pdf_to_rag(pdf_text, pdf_filename):
    """Ingest PDF content into RAG system."""
    print("="*70)
    print("INGESTING PDF INTO RAG SYSTEM")
    print("="*70)
    
    # Create RAG retriever
    rag = RAGRetriever(CONFIG, user_id=TEST_USER)
    
    # Create document
    doc = Document(
        page_content=pdf_text,
        metadata={
            "source": pdf_filename,
            "user_id": TEST_USER,
            "type": "pdf",
            "ingested_via": "test_script"
        }
    )
    
    # Ingest
    print(f"✓ Ingesting document: {pdf_filename}")
    rag.add_user_documents([doc], user_id=TEST_USER, save_index=True)
    print("✓ Document successfully ingested into RAG system\n")
    
    # Verify retrieval
    print("Verifying retrieval with sample query...")
    test_docs = rag.get_relevant_documents("summary", user_id=TEST_USER)
    print(f"✓ Can retrieve {len(test_docs)} relevant chunks\n")
    
    return rag

def run_interactive_cli(pdf_filename):
    """Run interactive CLI for testing."""
    print("="*70)
    print("INTERACTIVE PDF Q&A - CLI MODE")
    print("="*70)
    print(f"PDF File: {pdf_filename}")
    print(f"User: {TEST_USER}")
    print(f"Session: {TEST_SESSION}")
    print("="*70)
    
    # Create agent
    agent, _, _, memory = create_agent(
        user_id=TEST_USER,
        session_id=TEST_SESSION,
        config=CONFIG
    )
    
    print(f"✓ Agent initialized with RAG retriever")
    print(f"✓ Ready to answer questions about: {pdf_filename}\n")
    
    print("SUGGESTED TEST QUERIES:")
    print("  1. 'summarize this document'")
    print("  2. 'what are the main topics covered?'")
    print("  3. 'extract key points from this PDF'")
    print("  4. Ask specific questions about content\n")
    
    print("Commands: 'exit', 'quit' to stop\n")
    print("="*70 + "\n")
    
    response_buffer = ""
    
    def stream_printer(chunk: str):
        """Print streaming tokens."""
        nonlocal response_buffer
        response_buffer += chunk
        # Don't print during streaming - we'll parse and print after
    
    try:
        while True:
            user_input = input("\n🧑 You: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ('exit', 'quit', 'bye'):
                print("\n👋 Goodbye! Test complete.")
                break
            
            print(f"\n🤖 Collabry: ", end="", flush=True)
            response_buffer = ""
            
            try:
                agent.handle_user_input_stream(user_input, stream_printer)
                
                # Parse JSON and extract answer
                if response_buffer:
                    # Try to extract JSON
                    try:
                        # Find JSON object in response
                        start = response_buffer.find('{')
                        end = response_buffer.rfind('}') + 1
                        
                        if start != -1 and end > start:
                            json_str = response_buffer[start:end]
                            parsed = json.loads(json_str)
                            
                            # Extract answer field
                            if 'answer' in parsed:
                                answer = parsed['answer']
                                print(answer, flush=True)
                            else:
                                # No answer field, just print raw
                                print(response_buffer, flush=True)
                        else:
                            # No JSON found, print raw
                            print(response_buffer, flush=True)
                    except json.JSONDecodeError:
                        # JSON parsing failed, print raw response
                        print(response_buffer, flush=True)
                
                print("\n")
                
                # Analyze response
                if response_buffer:
                    response_lower = response_buffer.lower()
                    
                    # Check for hallucination indicators
                    hallucination_phrases = [
                        "i don't have access",
                        "i don't know",
                        "cannot access",
                        "don't have information",
                        "provided sources don't contain"
                    ]
                    
                    has_hallucination_warning = any(phrase in response_lower for phrase in hallucination_phrases)
                    has_source_citation = "source" in response_lower
                    
                    print("─" * 70)
                    if has_source_citation:
                        print("✓ Response cites sources - Good!")
                    elif has_hallucination_warning:
                        print("⚠ Response indicates information not in sources - This is correct behavior")
                    else:
                        print("ℹ Response generated (verify it uses PDF content)")
                    print("─" * 70)
                
            except Exception as e:
                print(f"\n❌ Error: {e}")
                import traceback
                traceback.print_exc()
    
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted. Test complete.")

def main():
    """Main test flow."""
    parser = argparse.ArgumentParser(
        description="Test RAG system with a real PDF file",
        epilog="Example: python test_pdf_rag.py myfile.pdf"
    )
    parser.add_argument(
        "pdf_path",
        help="Path to PDF file to test with"
    )
    args = parser.parse_args()
    
    pdf_path = Path(args.pdf_path)
    
    # Validate PDF exists
    if not pdf_path.exists():
        print(f"❌ Error: PDF file not found: {pdf_path}")
        sys.exit(1)
    
    if pdf_path.suffix.lower() != '.pdf':
        print(f"❌ Error: File is not a PDF: {pdf_path}")
        sys.exit(1)
    
    print("\n" + "="*70)
    print("PDF RAG SYSTEM - END-TO-END TEST")
    print("="*70)
    print(f"PDF File: {pdf_path.name}")
    print(f"Full Path: {pdf_path.absolute()}")
    print("="*70 + "\n")
    
    try:
        # Step 1: Extract text from PDF
        pdf_text = extract_pdf_text(pdf_path)
        
        if not pdf_text or len(pdf_text) < 100:
            print("⚠ Warning: Very little text extracted from PDF")
            print("   The PDF might be scanned images or empty")
            response = input("   Continue anyway? (y/n): ")
            if response.lower() != 'y':
                sys.exit(0)
        
        # Show preview
        print("TEXT PREVIEW (first 500 chars):")
        print("─" * 70)
        print(pdf_text[:500] + "..." if len(pdf_text) > 500 else pdf_text)
        print("─" * 70 + "\n")
        
        # Step 2: Ingest into RAG
        ingest_pdf_to_rag(pdf_text, pdf_path.name)
        
        # Step 3: Interactive CLI
        run_interactive_cli(pdf_path.name)
        
        print("\n" + "="*70)
        print("✅ END-TO-END TEST COMPLETE")
        print("="*70)
        print("\nIf the agent:")
        print("  ✓ Cited sources from your PDF")
        print("  ✓ Answered based on PDF content")
        print("  ✓ Didn't hallucinate external info")
        print("\nThen RAG is working correctly! 🎉")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n❌ Test failed with error:")
        print(f"{type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
