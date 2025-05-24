#!/usr/bin/env python3
"""
ChatPDF-AI - Fully Open Source Version
No external API dependencies - everything runs locally
"""

import os
from typing import Any, List, Optional

import streamlit as st
import torch
from langchain.chains import ConversationalRetrievalChain
from langchain.llms.base import LLM
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from PyPDF2 import PdfReader
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from htmlTemplates import bot_template, css, user_template


class LocalLLM(LLM):
    """Local LLM using Transformers library - completely open source"""

    model_name: str = "microsoft/DialoGPT-small"
    pipeline: Any = None
    tokenizer: Any = None
    model: Any = None

    class Config:
        """Pydantic config to allow arbitrary types"""

        arbitrary_types_allowed = True

    def __init__(self, model_name: str = "microsoft/DialoGPT-small", **kwargs):
        super().__init__(model_name=model_name, **kwargs)
        self._load_model()

    def _load_model(self):
        """Load the model locally"""
        try:
            st.info(f"Loading local model: {self.model_name}")

            # Check if CUDA is available
            device = 0 if torch.cuda.is_available() else -1
            device_name = "GPU" if device == 0 else "CPU"
            st.info(f"Using device: {device_name}")

            # Load model with pipeline for text generation
            pipeline_obj = pipeline(
                "text-generation",
                model=self.model_name,
                tokenizer=self.model_name,
                device=device,
                torch_dtype=torch.float16
                if torch.cuda.is_available()
                else torch.float32,
                trust_remote_code=True,
            )

            # Set the pipeline using object's __dict__ to bypass Pydantic validation
            object.__setattr__(self, "pipeline", pipeline_obj)

            st.success(f"✅ Successfully loaded {self.model_name} on {device_name}")

        except Exception as e:
            st.error(f"Error loading model {self.model_name}: {e}")
            # Fallback to a smaller model
            try:
                st.info("Trying fallback model: distilgpt2")
                # Update model_name using object's __dict__
                object.__setattr__(self, "model_name", "distilgpt2")

                pipeline_obj = pipeline(
                    "text-generation",
                    model="distilgpt2",
                    device=device,
                )
                object.__setattr__(self, "pipeline", pipeline_obj)
                st.success("✅ Successfully loaded fallback model: distilgpt2")
            except Exception as e2:
                st.error(f"Failed to load fallback model: {e2}")
                raise e2

    @property
    def _llm_type(self) -> str:
        return "local_transformers"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """Generate response using local model"""
        try:
            # Truncate prompt if too long
            max_prompt_length = 400
            if len(prompt) > max_prompt_length:
                prompt = prompt[:max_prompt_length] + "..."

            # Generate response
            result = self.pipeline(
                prompt,
                max_new_tokens=150,
                num_return_sequences=1,
                do_sample=True,
                temperature=0.7,
                pad_token_id=50256,
                return_full_text=False,
            )

            if result and len(result) > 0:
                response = result[0]["generated_text"].strip()

                # Clean up the response
                if stop:
                    for stop_word in stop:
                        if stop_word in response:
                            response = response.split(stop_word)[0]

                return (
                    response
                    if response
                    else "I need more context to provide a helpful answer."
                )
            else:
                return "I apologize, but I couldn't generate a response. Please try rephrasing your question."

        except Exception as e:
            st.error(f"Error generating response: {e}")
            return f"Error: {str(e)}"


def get_pdf_text(pdf_docs):
    """Enhanced PDF text extraction with better formatting and structure preservation"""
    all_text = ""
    
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            pdf_text = f"\n\n=== Document: {pdf.name} ===\n\n"
            
            for page_num, page in enumerate(pdf_reader.pages, 1):
                try:
                    page_text = page.extract_text()
                    
                    if page_text.strip():  # Only add non-empty pages
                        # Add page separator for better chunking
                        pdf_text += f"\n--- Page {page_num} ---\n"
                        
                        # Clean up text formatting
                        page_text = page_text.replace('\x00', '')  # Remove null characters
                        
                        # Normalize whitespace but preserve structure
                        lines = page_text.split('\n')
                        cleaned_lines = []
                        
                        for line in lines:
                            line = line.strip()
                            if line:  # Skip empty lines
                                cleaned_lines.append(line)
                        
                        # Join with single newlines for better sentence detection
                        clean_page_text = '\n'.join(cleaned_lines)
                        pdf_text += clean_page_text + "\n\n"
                        
                except Exception as page_error:
                    st.warning(f"Error reading page {page_num} of {pdf.name}: {page_error}")
                    continue
            
            all_text += pdf_text
            
        except Exception as e:
            st.error(f"Error reading PDF {pdf.name}: {e}")
            continue
    
    return all_text


def get_text_chunks(text):
    """Split text into smaller, more granular chunks for better processing"""
    
    # Create multiple chunking strategies for more comprehensive coverage
    chunks_all = []
    
    # Strategy 1: Small chunks for precise matching (primary)
    small_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " "],
        chunk_size=300,      # Smaller chunks for better precision
        chunk_overlap=50,    # Overlap to maintain context
        length_function=len,
        keep_separator=True
    )
    small_chunks = small_splitter.split_text(text)
    chunks_all.extend(small_chunks)
    
    # Strategy 2: Medium chunks for context (secondary)
    medium_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". "],
        chunk_size=600,      # Medium chunks for context
        chunk_overlap=100,   # More overlap for context preservation
        length_function=len,
        keep_separator=True
    )
    medium_chunks = medium_splitter.split_text(text)
    chunks_all.extend(medium_chunks)
    
    # Strategy 3: Sentence-based chunks for semantic coherence
    sentence_splitter = RecursiveCharacterTextSplitter(
        separators=[". ", "! ", "? ", "\n"],
        chunk_size=200,      # Very small for sentence-level granularity
        chunk_overlap=20,    # Minimal overlap for sentences
        length_function=len,
        keep_separator=True
    )
    sentence_chunks = sentence_splitter.split_text(text)
    chunks_all.extend(sentence_chunks)
    
    # Remove duplicate chunks and very small chunks
    unique_chunks = []
    seen_chunks = set()
    
    for chunk in chunks_all:
        chunk_clean = chunk.strip()
        # Skip very small chunks (less than 50 characters)
        if len(chunk_clean) < 50:
            continue
        # Skip duplicates
        if chunk_clean not in seen_chunks:
            unique_chunks.append(chunk_clean)
            seen_chunks.add(chunk_clean)
    
    # Sort by length to have more relevant chunks first
    unique_chunks.sort(key=len, reverse=True)
    
    return unique_chunks


def get_vectorstore(text_chunks):
    """Create vector store using open-source embeddings"""
    try:
        st.info("Loading embedding model...")

        # Use completely local embeddings - no API required
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},  # Use CPU for embeddings (more stable)
            encode_kwargs={"normalize_embeddings": True},
        )

        st.info("Creating vector store...")
        vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
        st.success("✅ Vector store created successfully!")

        return vectorstore
    except Exception as e:
        st.error(f"Error creating vector store: {e}")
        return None


def get_conversation_chain(vectorstore):
    """Create conversation chain with local LLM"""

    # List of local models to try (from smallest to largest)
    models_to_try = [
        "distilgpt2",  # Smallest, most likely to work
        "microsoft/DialoGPT-small",  # Small conversational model
        "microsoft/DialoGPT-medium",  # Medium model (if resources allow)
    ]

    llm = None
    for model_name in models_to_try:
        try:
            st.info(f"Attempting to load: {model_name}")
            llm = LocalLLM(model_name=model_name)

            # Test the model
            try:
                test_response = llm("Hello, how are you?")
                if test_response and not test_response.startswith("Error:"):
                    st.success(f"✅ Model {model_name} loaded and tested successfully!")
                    break
                else:
                    st.warning(f"Model {model_name} loaded but test failed")
                    continue
            except Exception as test_e:
                st.warning(f"Model {model_name} test failed: {test_e}")
                continue

        except Exception as e:
            st.warning(f"Failed to load {model_name}: {e}")
            continue

    if llm is None:
        st.error("❌ Could not load any local language model!")
        st.info("💡 Try installing a smaller model or check your system resources.")
        return None

    try:
        # Create conversation chain
        memory = ConversationBufferMemory(
            memory_key="chat_history", 
            return_messages=True,
            output_key="answer"  # Specify the output key for the chain
        )

        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(
                search_type="mmr",  # Use Maximum Marginal Relevance for diverse results
                search_kwargs={
                    "k": 8,           # Retrieve more chunks (increased from 3)
                    "fetch_k": 20,    # Fetch more candidates before MMR filtering
                    "lambda_mult": 0.7  # Balance between relevance and diversity
                }
            ),
            memory=memory,
            return_source_documents=True,
        )

        return conversation_chain
    except Exception as e:
        st.error(f"Error creating conversation chain: {e}")
        return None


def handle_userinput(user_question):
    """Handle user input and generate response"""
    if st.session_state.conversation is None:
        st.error("Please upload and process PDF documents first!")
        return

    try:
        with st.spinner("🤔 Thinking... (this may take a moment with local models)"):
            response = st.session_state.conversation({"question": user_question})
            
            # Handle the response structure properly
            if "chat_history" in response:
                st.session_state.chat_history = response["chat_history"]
            else:
                # For ConversationalRetrievalChain, we need to get the answer and update chat history manually
                answer = response.get("answer", "Sorry, I couldn't generate a response.")
                
                # Initialize chat history if it doesn't exist
                if "chat_history" not in st.session_state or st.session_state.chat_history is None:
                    st.session_state.chat_history = []
                
                # Add user question and bot answer to chat history
                # Create simple message objects for display
                class SimpleMessage:
                    def __init__(self, content):
                        self.content = content
                
                st.session_state.chat_history.append(SimpleMessage(user_question))
                st.session_state.chat_history.append(SimpleMessage(answer))

        # Display conversation history
        if st.session_state.chat_history:
            for i, message in enumerate(st.session_state.chat_history):
                if i % 2 == 0:
                    st.write(
                        user_template.replace("{{MSG}}", message.content),
                        unsafe_allow_html=True,
                    )
                else:
                    st.write(
                        bot_template.replace("{{MSG}}", message.content),
                        unsafe_allow_html=True,
                    )

        # Show source documents if available
        if "source_documents" in response and response["source_documents"]:
            with st.expander(f"📄 Source Documents ({len(response['source_documents'])} found)"):
                for i, doc in enumerate(response["source_documents"][:6]):  # Show top 6 (increased from 3)
                    st.write(f"**Source {i + 1}:**")
                    # Show more text from each document
                    content = doc.page_content
                    if len(content) > 500:
                        st.write(content[:500] + "...")
                    else:
                        st.write(content)
                    
                    # Add separator between sources
                    if i < len(response["source_documents"][:6]) - 1:
                        st.markdown("---")

    except Exception as e:
        st.error(f"Error processing your question: {e}")
        st.info(
            "💡 Try asking a simpler question or check if your PDFs were processed correctly."
        )


def main():
    """Main application function"""
    st.set_page_config(
        page_title="ChatPDF-AI (Open Source)", page_icon="📚", layout="wide"
    )
    st.write(css, unsafe_allow_html=True)

    # Initialize session state
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    # Main header
    st.header("📚 ChatPDF-AI - Open Source Edition")
    st.markdown("*Completely local processing - no external APIs required!*")

    # System info
    with st.expander("ℹ️ System Information"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "PyTorch Available",
                "✅" if torch.cuda.is_available() else "❌ CPU Only",
            )
        with col2:
            st.metric("CUDA Available", "✅" if torch.cuda.is_available() else "❌")
        with col3:
            device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
            st.metric("GPU Devices", device_count)

    # Main interface
    user_question = st.text_input("💬 Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    # Sidebar for document upload
    with st.sidebar:
        st.subheader("📄 Your Documents")

        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click 'Process'",
            accept_multiple_files=True,
            type=["pdf"],
        )

        if pdf_docs:
            st.success(f"📁 Uploaded {len(pdf_docs)} PDF(s)")

            # Show file details
            for pdf in pdf_docs:
                st.write(f"• {pdf.name} ({pdf.size} bytes)")

        if st.button("🚀 Process Documents", type="primary"):
            if not pdf_docs:
                st.error("Please upload at least one PDF file!")
                return

            with st.spinner("⚙️ Processing documents..."):
                # Extract text from PDFs
                raw_text = get_pdf_text(pdf_docs)

                if not raw_text.strip():
                    st.error("❌ No text could be extracted from the PDF(s).")
                    st.info(
                        "Make sure your PDFs contain readable text (not just images)."
                    )
                    return

                st.info(f"📝 Extracted {len(raw_text):,} characters from PDF(s)")

                # Create text chunks with enhanced strategy
                text_chunks = get_text_chunks(raw_text)
                st.success(f"✂️ Created {len(text_chunks)} text chunks using multi-strategy approach")
                
                # Show chunking details
                if text_chunks:
                    avg_chunk_size = sum(len(chunk) for chunk in text_chunks) / len(text_chunks)
                    st.info(f"📊 Average chunk size: {avg_chunk_size:.0f} characters, Total chunks: {len(text_chunks)}")
                    
                    # Show chunk size distribution
                    small_chunks = sum(1 for chunk in text_chunks if len(chunk) < 300)
                    medium_chunks = sum(1 for chunk in text_chunks if 300 <= len(chunk) < 600)
                    large_chunks = sum(1 for chunk in text_chunks if len(chunk) >= 600)
                    
                    st.info(f"📈 Chunk distribution: {small_chunks} small, {medium_chunks} medium, {large_chunks} large")

                # Create vector store
                vectorstore = get_vectorstore(text_chunks)
                if vectorstore is None:
                    return

                # Create conversation chain
                conversation = get_conversation_chain(vectorstore)
                if conversation is not None:
                    st.session_state.conversation = conversation
                    st.success("✅ Processing completed! You can now ask questions.")
                    st.balloons()
                else:
                    st.error("❌ Failed to create conversation chain.")

        # Info section
        st.markdown("---")
        st.markdown("### 🔧 Open Source Components")
        st.markdown("""
        - **LLM**: Local Transformers models
        - **Embeddings**: HuggingFace Embeddings
        - **Vector DB**: FAISS (local)
        - **UI**: Streamlit
        - **No API keys required!**
        """)

        st.markdown("### 💡 Tips")
        st.markdown("""
        - First run will download models (~500MB)
        - GPU recommended for better performance
        - Smaller documents process faster
        - Be patient with local models
        """)


if __name__ == "__main__":
    main()
