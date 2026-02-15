"""
Verified Knowledge Ingestion Service

Unified service for ingesting content from multiple source types.
"""

import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .base import get_verified_knowledge_store
from .extractors import URLExtractor, TextProcessor, PDFExtractor
from .validators import ContentValidator, AuthorityClassifier

logger = logging.getLogger(__name__)


@dataclass
class IngestionResult:
    """Result of content ingestion."""
    success: bool
    documents_added: int = 0
    authority_level: str = 'low'
    errors: List[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.metadata is None:
            self.metadata = {}


class VerifiedKnowledgeIngestionService:
    """Service for ingesting content into verified knowledge base."""
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        """
        Initialize ingestion service.
        
        Args:
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        self.vkb = get_verified_knowledge_store()
        self.url_extractor = URLExtractor()
        self.text_processor = TextProcessor()
        try:
            self.pdf_extractor = PDFExtractor()
        except ImportError:
            logger.warning("PDF extraction not available (PyPDF2 not installed)")
            self.pdf_extractor = None
        
        self.validator = ContentValidator()
        self.authority_classifier = AuthorityClassifier()
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
    
    async def ingest_from_url(
        self,
        url: str,
        user_authority: Optional[str] = None,
        **extra_metadata
    ) -> IngestionResult:
        """
        Ingest content from URL.
        
        Args:
            url: Web URL to ingest
            user_authority: User-specified authority level
            **extra_metadata: Additional metadata
            
        Returns:
            IngestionResult
        """
        logger.info(f"Ingesting from URL: {url}")
        
        try:
            # Extract content
            content = self.url_extractor.extract(url)
            
            # Validate
            validation = self.validator.validate(content.text, 'url')
            if not validation.is_valid:
                logger.warning(f"Validation failed for {url}: {validation.warnings}")
                return IngestionResult(
                    success=False,
                    errors=validation.warnings
                )
            
            # Classify authority
            authority = self.authority_classifier.classify(
                'url',
                {'url': url, **content.metadata},
                user_authority
            )
            
            # Chunk content
            chunks = self._chunk_content(content.text)
            
            # Create documents
            documents = []
            for i, chunk in enumerate(chunks):
                doc = Document(
                    page_content=chunk,
                    metadata={
                        'source_type': 'url',
                        'source_name': content.title or url,
                        'url': url,
                        'domain': content.domain,
                        'chunk_index': i,
                        'total_chunks': len(chunks),
                        'authority_level': authority,
                        **content.metadata,
                        **extra_metadata
                    }
                )
                documents.append(doc)
            
            # Add to knowledge base
            success = self.vkb.add_documents(
                documents,
                authority_level=authority,
                source_type='url',
                source_name=content.title or url
            )
            
            if success:
                logger.info(f"✅ Ingested {len(documents)} chunks from {url}")
                return IngestionResult(
                    success=True,
                    documents_added=len(documents),
                    authority_level=authority,
                    metadata={'url': url, 'title': content.title}
                )
            else:
                return IngestionResult(
                    success=False,
                    errors=["Failed to add documents to knowledge base"]
                )
        
        except Exception as e:
            logger.error(f"Failed to ingest URL {url}: {e}")
            return IngestionResult(
                success=False,
                errors=[str(e)]
            )
    
    def ingest_from_text(
        self,
        text: str,
        title: str,
        authority_level: str = 'low',
        **extra_metadata
    ) -> IngestionResult:
        """
        Ingest plain text content.
        
        Args:
            text: Text content
            title: Content title
            authority_level: Authority level ('high' | 'medium' | 'low')
            **extra_metadata: Additional metadata
            
        Returns:
            IngestionResult
        """
        logger.info(f"Ingesting text: {title}")
        
        try:
            # Process text
            processed = self.text_processor.process(
                text,
                title,
                authority_level,
                **extra_metadata
            )
            
            # Validate
            validation = self.validator.validate(processed.text, 'text')
            if not validation.is_valid:
                logger.warning(f"Validation failed for text '{title}': {validation.warnings}")
                return IngestionResult(
                    success=False,
                    errors=validation.warnings
                )
            
            # Chunk content
            chunks = self._chunk_content(processed.text)
            
            # Create documents
            documents = []
            for i, chunk in enumerate(chunks):
                doc = Document(
                    page_content=chunk,
                    metadata={
                        'source_type': 'text',
                        'source_name': title,
                        'chunk_index': i,
                        'total_chunks': len(chunks),
                        'authority_level': authority_level,
                        **processed.metadata,
                        **extra_metadata
                    }
                )
                documents.append(doc)
            
            # Add to knowledge base
            success = self.vkb.add_documents(
                documents,
                authority_level=authority_level,
                source_type='text',
                source_name=title
            )
            
            if success:
                logger.info(f"✅ Ingested {len(documents)} chunks from text '{title}'")
                return IngestionResult(
                    success=True,
                    documents_added=len(documents),
                    authority_level=authority_level,
                    metadata={'title': title, 'word_count': processed.word_count}
                )
            else:
                return IngestionResult(
                    success=False,
                    errors=["Failed to add documents to knowledge base"]
                )
        
        except Exception as e:
            logger.error(f"Failed to ingest text '{title}': {e}")
            return IngestionResult(
                success=False,
                errors=[str(e)]
            )
    
    async def ingest_from_pdf(
        self,
        pdf_path: str,
        user_authority: Optional[str] = None,
        **extra_metadata
    ) -> IngestionResult:
        """
        Ingest content from PDF file.
        
        Args:
            pdf_path: Path to PDF file
            user_authority: User-specified authority level
            **extra_metadata: Additional metadata
            
        Returns:
            IngestionResult
        """
        if not self.pdf_extractor:
            return IngestionResult(
                success=False,
                errors=["PDF extraction not available (PyPDF2 not installed)"]
            )
        
        logger.info(f"Ingesting from PDF: {pdf_path}")
        
        try:
            # Extract content
            content = self.pdf_extractor.extract(pdf_path)
            
            # Validate
            validation = self.validator.validate(content.text, 'pdf')
            if not validation.is_valid:
                logger.warning(f"Validation failed for PDF {pdf_path}: {validation.warnings}")
                return IngestionResult(
                    success=False,
                    errors=validation.warnings
                )
            
            # Classify authority
            authority = self.authority_classifier.classify(
                'pdf',
                content.metadata,
                user_authority
            )
            
            # Chunk content
            chunks = self._chunk_content(content.text)
            
            # Create documents
            documents = []
            for i, chunk in enumerate(chunks):
                doc = Document(
                    page_content=chunk,
                    metadata={
                        'source_type': 'pdf',
                        'source_name': content.title or pdf_path,
                        'pdf_path': pdf_path,
                        'chunk_index': i,
                        'total_chunks': len(chunks),
                        'authority_level': authority,
                        **content.metadata,
                        **extra_metadata
                    }
                )
                documents.append(doc)
            
            # Add to knowledge base
            success = self.vkb.add_documents(
                documents,
                authority_level=authority,
                source_type='pdf',
                source_name=content.title or pdf_path
            )
            
            if success:
                logger.info(f"✅ Ingested {len(documents)} chunks from PDF {pdf_path}")
                return IngestionResult(
                    success=True,
                    documents_added=len(documents),
                    authority_level=authority,
                    metadata={
                        'title': content.title,
                        'page_count': content.page_count,
                        'publisher': content.publisher
                    }
                )
            else:
                return IngestionResult(
                    success=False,
                    errors=["Failed to add documents to knowledge base"]
                )
        
        except Exception as e:
            logger.error(f"Failed to ingest PDF {pdf_path}: {e}")
            return IngestionResult(
                success=False,
                errors=[str(e)]
            )
    
    def _chunk_content(self, text: str) -> List[str]:
        """Chunk text content into smaller pieces."""
        return self.text_splitter.split_text(text)
