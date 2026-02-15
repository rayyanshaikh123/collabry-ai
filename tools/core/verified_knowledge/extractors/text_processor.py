"""
Text Content Processor

Processes plain text content with validation and metadata extraction.
"""

import logging
import re
from typing import Optional, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ProcessedText:
    """Container for processed text content."""
    text: str
    title: str
    metadata: Dict[str, Any]
    word_count: int = 0
    
    def __post_init__(self):
        if not self.word_count:
            self.word_count = len(self.text.split())


class TextProcessor:
    """Process plain text content for verified knowledge base."""
    
    def __init__(self):
        """Initialize text processor."""
        pass
    
    def process(
        self,
        text: str,
        title: str,
        authority_level: str = 'low',
        **metadata_kwargs
    ) -> ProcessedText:
        """
        Process text content.
        
        Args:
            text: Raw text content
            title: Content title
            authority_level: 'high' | 'medium' | 'low'
            **metadata_kwargs: Additional metadata
            
        Returns:
            ProcessedText object
        """
        logger.info(f"Processing text: {title}")
        
        # Clean text
        cleaned_text = self._clean_text(text)
        
        # Extract metadata from content
        extracted_metadata = self._extract_metadata_from_text(cleaned_text)
        
        # Merge metadata
        metadata = {
            'title': title,
            'authority_level': authority_level,
            'source_type': 'text',
            **extracted_metadata,
            **metadata_kwargs
        }
        
        return ProcessedText(
            text=cleaned_text,
            title=title,
            metadata=metadata
        )
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters (keep basic punctuation)
        text = re.sub(r'[^\w\s.,;:!?()\-\'"]+', '', text)
        
        # Normalize line breaks
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        # Remove multiple consecutive newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text.strip()
    
    def _extract_metadata_from_text(self, text: str) -> Dict[str, Any]:
        """Extract metadata from text content."""
        metadata = {}
        
        # Try to detect subject from content
        subject_keywords = {
            'physics': ['force', 'energy', 'motion', 'velocity', 'acceleration'],
            'chemistry': ['molecule', 'atom', 'reaction', 'element', 'compound'],
            'biology': ['cell', 'organism', 'photosynthesis', 'DNA', 'evolution'],
            'mathematics': ['equation', 'theorem', 'proof', 'integral', 'derivative'],
        }
        
        text_lower = text.lower()
        for subject, keywords in subject_keywords.items():
            if sum(1 for kw in keywords if kw in text_lower) >= 2:
                metadata['detected_subject'] = subject
                break
        
        # Word count
        metadata['word_count'] = len(text.split())
        
        # Character count
        metadata['char_count'] = len(text)
        
        return metadata
    
    def validate_text(self, text: str) -> tuple[bool, Optional[str]]:
        """
        Validate text content.
        
        Args:
            text: Text to validate
            
        Returns:
            (is_valid, error_message)
        """
        # Minimum length check
        if len(text.strip()) < 100:
            return False, "Text too short (minimum 100 characters)"
        
        # Maximum length check (prevent abuse)
        if len(text) > 1_000_000:  # 1MB
            return False, "Text too long (maximum 1MB)"
        
        # Check for excessive special characters (spam detection)
        special_char_ratio = len(re.findall(r'[^\w\s]', text)) / len(text)
        if special_char_ratio > 0.3:
            return False, "Too many special characters (possible spam)"
        
        # Check for minimum word count
        word_count = len(text.split())
        if word_count < 20:
            return False, "Too few words (minimum 20 words)"
        
        return True, None
