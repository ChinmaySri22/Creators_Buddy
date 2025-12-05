"""
RAG (Retrieval Augmented Generation) System for Script Generation
Uses ChromaDB to store and retrieve similar scripts for context injection
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Optional

# Try to import chromadb, handle missing dependencies gracefully
try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except (ImportError, ValueError) as e:
    CHROMADB_AVAILABLE = False
    print(f"[WARN] ChromaDB not available: {e}")
    print("[WARN] Install required packages: pip install chromadb onnxruntime")

try:
    from transcript_processor import ProcessedTranscript
except ImportError:
    # Fallback for when transcript_processor is not available
    ProcessedTranscript = None

class RAGSystem:
    """RAG system for retrieving similar scripts based on topic/content"""
    
    def __init__(self, persist_directory: str = "chroma_db"):
        """Initialize the RAG system with ChromaDB"""
        if not CHROMADB_AVAILABLE:
            raise ImportError("ChromaDB is not available. Install with: pip install chromadb onnxruntime")
        
        self.persist_directory = persist_directory
        os.makedirs(persist_directory, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name="youtube_scripts",
            metadata={"hnsw:space": "cosine"}
        )
        
        self._initialized = False
    
    def initialize(self, transcripts: List[ProcessedTranscript]):
        """Initialize the vector database with transcripts"""
        if self._initialized:
            # Check if we need to update
            existing_count = self.collection.count()
            if existing_count >= len(transcripts):
                print(f"[RAG] Database already initialized with {existing_count} scripts")
                return
        
        print(f"[RAG] Initializing vector database with {len(transcripts)} transcripts...")
        
        # Clear existing data if reinitializing
        if self.collection.count() > 0:
            # Delete and recreate collection
            self.client.delete_collection("youtube_scripts")
            self.collection = self.client.create_collection(
                name="youtube_scripts",
                metadata={"hnsw:space": "cosine"}
            )
        
        # Process transcripts in batches
        batch_size = 100
        for i in range(0, len(transcripts), batch_size):
            batch = transcripts[i:i + batch_size]
            self._add_transcripts_batch(batch, i)
        
        self._initialized = True
        print(f"[RAG] Vector database initialized with {self.collection.count()} scripts")
    
    def _add_transcripts_batch(self, transcripts: List[ProcessedTranscript], start_idx: int):
        """Add a batch of transcripts to the vector database"""
        ids = []
        documents = []
        metadatas = []
        
        for idx, transcript in enumerate(transcripts):
            # Create document text from transcript
            doc_text = self._create_document_text(transcript)
            
            # Create unique ID
            doc_id = f"script_{start_idx + idx}_{transcript.metadata.video_id}"
            
            # Create metadata
            metadata = {
                "video_id": transcript.metadata.video_id,
                "title": transcript.metadata.title,
                "uploader": transcript.metadata.uploader,
                "duration": transcript.metadata.duration,
                "word_count": transcript.metadata.word_count,
                "genre": getattr(transcript.metadata, 'genre', 'unknown'),
                "view_count": getattr(transcript.metadata, 'view_count', 0),
            }
            
            ids.append(doc_id)
            documents.append(doc_text)
            metadatas.append(metadata)
        
        # Add to collection
        self.collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )
    
    def _create_document_text(self, transcript: ProcessedTranscript) -> str:
        """Create a searchable text document from transcript"""
        # Combine title, transcript text, and metadata
        parts = [
            f"Title: {transcript.metadata.title}",
            f"Creator: {transcript.metadata.uploader}",
        ]
        
        # Add transcript text (first 2000 chars to avoid too long documents)
        transcript_text = " ".join([seg.get('text', '') for seg in transcript.transcript_segments[:50]])
        if len(transcript_text) > 2000:
            transcript_text = transcript_text[:2000] + "..."
        parts.append(f"Content: {transcript_text}")
        
        # Add style markers if available
        if hasattr(transcript, 'style_markers') and transcript.style_markers:
            parts.append(f"Style: {', '.join(transcript.style_markers[:5])}")
        
        return "\n".join(parts)
    
    def retrieve_similar(self, query: str, n_results: int = 5, filters: Optional[Dict] = None) -> List[Dict]:
        """Retrieve similar scripts based on query"""
        if not self._initialized or self.collection.count() == 0:
            return []
        
        try:
            # Build query filters if provided
            where = None
            if filters:
                where = {}
                if 'uploader' in filters:
                    where['uploader'] = filters['uploader']
                if 'genre' in filters:
                    where['genre'] = filters['genre']
            
            # Query the collection
            results = self.collection.query(
                query_texts=[query],
                n_results=min(n_results, self.collection.count()),
                where=where
            )
            
            # Format results
            similar_scripts = []
            if results['ids'] and len(results['ids'][0]) > 0:
                for i in range(len(results['ids'][0])):
                    script_data = {
                        'id': results['ids'][0][i],
                        'document': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'distance': results['distances'][0][i] if 'distances' in results else None
                    }
                    similar_scripts.append(script_data)
            
            return similar_scripts
            
        except Exception as e:
            print(f"[RAG] Error retrieving similar scripts: {e}")
            return []
    
    def get_best_performing_scripts(self, n: int = 5) -> List[Dict]:
        """Retrieve best performing scripts based on view count"""
        if not self._initialized or self.collection.count() == 0:
            return []
        
        try:
            # Get all scripts
            all_results = self.collection.get()
            
            # Sort by view count
            scripts_with_views = []
            for i, metadata in enumerate(all_results['metadatas']):
                view_count = metadata.get('view_count', 0)
                if view_count > 0:
                    scripts_with_views.append({
                        'id': all_results['ids'][i],
                        'document': all_results['documents'][i],
                        'metadata': metadata,
                        'view_count': view_count
                    })
            
            # Sort by view count (descending) and return top n
            scripts_with_views.sort(key=lambda x: x['view_count'], reverse=True)
            return scripts_with_views[:n]
            
        except Exception as e:
            print(f"[RAG] Error retrieving best performing scripts: {e}")
            return []
    
    def get_scripts_by_creator(self, creator_name: str, n: int = 5) -> List[Dict]:
        """Retrieve scripts by a specific creator"""
        return self.retrieve_similar(
            query=f"Creator: {creator_name}",
            n_results=n,
            filters={'uploader': creator_name}
        )

