"""
PDF Content Extractor

Extracts text content from PDF files with publisher-based authority classification.
"""

import logging
import re
from typing import Optional, Dict, Any
from dataclasses import dataclass
from pathlib import Path

try:
    import PyPDF2
    HAS_PYPDF2 = True
except ImportError:
    HAS_PYPDF2 = False
    logger = logging.getLogger(__name__)
    logger.warning("PyPDF2 not installed. PDF extraction will not work.")


logger = logging.getLogger(__name__)


# Trusted publishers for authority classification
TRUSTED_PUBLISHERS = {
    'high': [
        'NCERT',
        'CBSE',
        'ICSE',
        'Pearson Education',
        'McGraw-Hill Education',
        'Oxford University Press',
        'Cambridge University Press',
        'Wiley',
        'Springer',
        'Elsevier',
    ],
    'medium': [
        'Cengage Learning',
        'Houghton Mifflin',
        'Macmillan',
        'Scholastic',
    ],
    'low': []
}


@dataclass
class ExtractedPDF:
    """Container for extracted PDF content."""
    text: str
    title: Optional[str] = None
    author: Optional[str] = None
    publisher: Optional[str] = None
    publication_year: Optional[int] = None
    isbn: Optional[str] = None
    doi: Optional[str] = None
    page_count: int = 0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class PDFExtractor:
    """Extract text content from PDF files."""
    
    def __init__(self):
        """Initialize PDF extractor."""
        if not HAS_PYPDF2:
            raise ImportError("PyPDF2 is required for PDF extraction. Install with: pip install PyPDF2")
    
    def extract(self, pdf_path: str) -> ExtractedPDF:
        """
        Extract content from PDF file.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            ExtractedPDF object
            
        Raises:
            Exception: If extraction fails
        """
        logger.info(f"Extracting content from PDF: {pdf_path}")
        
        if not Path(pdf_path).exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                # Extract metadata
                metadata = self._extract_metadata(pdf_reader)
                
                # Extract text from all pages
                text_parts = []
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text = page.extract_text()
                    if text:
                        text_parts.append(text)
                
                full_text = '\n\n'.join(text_parts)
                
                # Clean text
                cleaned_text = self._clean_text(full_text)
                
                return ExtractedPDF(
                    text=cleaned_text,
                    title=metadata.get('title'),
                    author=metadata.get('author'),
                    publisher=metadata.get('publisher'),
                    publication_year=metadata.get('publication_year'),
                    isbn=metadata.get('isbn'),
                    doi=metadata.get('doi'),
                    page_count=len(pdf_reader.pages),
                    metadata=metadata
                )
        
        except Exception as e:
            logger.error(f"Failed to extract PDF {pdf_path}: {e}")
            raise Exception(f"PDF extraction failed: {e}")
    
    def _extract_metadata(self, pdf_reader: 'PyPDF2.PdfReader') -> Dict[str, Any]:
        """Extract metadata from PDF."""
        metadata = {}
        
        if pdf_reader.metadata:
            pdf_meta = pdf_reader.metadata
            
            # Standard metadata fields
            if pdf_meta.get('/Title'):
                metadata['title'] = str(pdf_meta.get('/Title'))
            if pdf_meta.get('/Author'):
                metadata['author'] = str(pdf_meta.get('/Author'))
            if pdf_meta.get('/Subject'):
                metadata['subject'] = str(pdf_meta.get('/Subject'))
            if pdf_meta.get('/Creator'):
                metadata['creator'] = str(pdf_meta.get('/Creator'))
            if pdf_meta.get('/Producer'):
                metadata['producer'] = str(pdf_meta.get('/Producer'))
            
            # Try to extract publisher from creator/producer
            creator = metadata.get('creator', '')
            producer = metadata.get('producer', '')
            for publisher in TRUSTED_PUBLISHERS['high'] + TRUSTED_PUBLISHERS['medium']:
                if publisher.lower() in creator.lower() or publisher.lower() in producer.lower():
                    metadata['publisher'] = publisher
                    break
        
        metadata['page_count'] = len(pdf_reader.pages)
        
        return metadata
    
    def _clean_text(self, text: str) -> str:
        """Clean extracted PDF text."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page numbers (common pattern)
        text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
        
        # Remove headers/footers (heuristic)
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            # Skip very short lines that might be headers/footers
            if len(line.strip()) > 10:
                cleaned_lines.append(line)
        
        text = '\n'.join(cleaned_lines)
        
        # Normalize line breaks
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text.strip()
    
    def classify_publisher_authority(self, publisher: Optional[str]) -> str:
        """
        Classify authority level based on publisher.
        
        Args:
            publisher: Publisher name
            
        Returns:
            'high' | 'medium' | 'low'
        """
        if not publisher:
            return 'low'
        
        publisher_lower = publisher.lower()
        
        # Check high authority publishers
        for trusted_pub in TRUSTED_PUBLISHERS['high']:
            if trusted_pub.lower() in publisher_lower:
                return 'high'
        
        # Check medium authority publishers
        for trusted_pub in TRUSTED_PUBLISHERS['medium']:
            if trusted_pub.lower() in publisher_lower:
                return 'medium'
        
        return 'low'
    
    def extract_isbn(self, text: str) -> Optional[str]:
        """Extract ISBN from text."""
        # ISBN-13 pattern
        isbn13_pattern = r'ISBN[-\s]?(?:13)?:?\s*(\d{3}[-\s]?\d{1,5}[-\s]?\d{1,7}[-\s]?\d{1,7}[-\s]?\d)'
        match = re.search(isbn13_pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).replace('-', '').replace(' ', '')
        
        # ISBN-10 pattern
        isbn10_pattern = r'ISBN[-\s]?(?:10)?:?\s*(\d{1,5}[-\s]?\d{1,7}[-\s]?\d{1,7}[-\s]?[\dX])'
        match = re.search(isbn10_pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).replace('-', '').replace(' ', '')
        
        return None
    
    def extract_doi(self, text: str) -> Optional[str]:
        """Extract DOI from text."""
        doi_pattern = r'10\.\d{4,}/[^\s]+'
        match = re.search(doi_pattern, text)
        if match:
            return match.group(0)
        return None
