"""
Content Validator

Validates content quality and suitability for verified knowledge base.
"""

import logging
import re
from typing import Optional, Dict, List
from dataclasses import dataclass

try:
    from langdetect import detect, LangDetectException
    HAS_LANGDETECT = True
except ImportError:
    HAS_LANGDETECT = False

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of content validation."""
    is_valid: bool
    checks: Dict[str, bool]
    warnings: List[str]
    quality_score: float = 0.0


class ContentValidator:
    """Validate content for verified knowledge base."""
    
    def __init__(
        self,
        min_length: int = 100,
        max_length: int = 1_000_000,
        min_words: int = 20,
        required_language: str = 'en'
    ):
        """
        Initialize content validator.
        
        Args:
            min_length: Minimum character count
            max_length: Maximum character count
            min_words: Minimum word count
            required_language: Required language code
        """
        self.min_length = min_length
        self.max_length = max_length
        self.min_words = min_words
        self.required_language = required_language
    
    def validate(self, content: str, source_type: str) -> ValidationResult:
        """
        Validate content.
        
        Args:
            content: Content to validate
            source_type: 'url' | 'text' | 'pdf'
            
        Returns:
            ValidationResult object
        """
        checks = {}
        warnings = []
        
        # Length validation
        content_length = len(content.strip())
        checks['min_length'] = content_length >= self.min_length
        if not checks['min_length']:
            warnings.append(f"Content too short ({content_length} chars, minimum {self.min_length})")
        
        checks['max_length'] = content_length <= self.max_length
        if not checks['max_length']:
            warnings.append(f"Content too long ({content_length} chars, maximum {self.max_length})")
        
        # Word count validation
        word_count = len(content.split())
        checks['min_words'] = word_count >= self.min_words
        if not checks['min_words']:
            warnings.append(f"Too few words ({word_count} words, minimum {self.min_words})")
        
        # Language detection
        if HAS_LANGDETECT:
            try:
                detected_lang = detect(content[:1000])  # Check first 1000 chars
                checks['language'] = detected_lang == self.required_language
                if not checks['language']:
                    warnings.append(f"Wrong language (detected: {detected_lang}, required: {self.required_language})")
            except LangDetectException:
                checks['language'] = True  # Assume OK if detection fails
                warnings.append("Could not detect language")
        else:
            checks['language'] = True
        
        # Quality scoring
        quality_score = self._calculate_quality(content)
        checks['quality_score'] = quality_score >= 0.5
        if not checks['quality_score']:
            warnings.append(f"Low quality score ({quality_score:.2f}, minimum 0.5)")
        
        # Spam detection
        checks['no_spam'] = not self._is_spam(content)
        if not checks['no_spam']:
            warnings.append("Content appears to be spam")
        
        # Duplicate detection (placeholder - would need database access)
        checks['no_duplicate'] = True
        
        is_valid = all(checks.values())
        
        return ValidationResult(
            is_valid=is_valid,
            checks=checks,
            warnings=warnings,
            quality_score=quality_score
        )
    
    def _calculate_quality(self, content: str) -> float:
        """
        Calculate content quality score (0-1).
        
        Factors:
        - Readability (sentence structure)
        - Information density (unique words)
        - Coherence (paragraph structure)
        """
        score = 0.0
        
        # Sentence structure (0-0.3)
        sentences = re.split(r'[.!?]+', content)
        valid_sentences = [s for s in sentences if len(s.split()) >= 3]
        if sentences:
            sentence_score = min(len(valid_sentences) / len(sentences), 1.0) * 0.3
            score += sentence_score
        
        # Information density (0-0.4)
        words = content.lower().split()
        if words:
            unique_ratio = len(set(words)) / len(words)
            density_score = min(unique_ratio * 2, 1.0) * 0.4  # Scale up
            score += density_score
        
        # Paragraph structure (0-0.3)
        paragraphs = [p for p in content.split('\n\n') if p.strip()]
        if paragraphs:
            avg_para_length = sum(len(p.split()) for p in paragraphs) / len(paragraphs)
            # Good paragraphs: 50-200 words
            if 50 <= avg_para_length <= 200:
                para_score = 0.3
            elif 20 <= avg_para_length <= 300:
                para_score = 0.2
            else:
                para_score = 0.1
            score += para_score
        
        return min(score, 1.0)
    
    def _is_spam(self, content: str) -> bool:
        """Detect spam content."""
        # Check for excessive special characters
        special_char_ratio = len(re.findall(r'[^\w\s]', content)) / max(len(content), 1)
        if special_char_ratio > 0.3:
            return True
        
        # Check for excessive capitalization
        upper_ratio = sum(1 for c in content if c.isupper()) / max(len(content), 1)
        if upper_ratio > 0.5:
            return True
        
        # Check for repeated patterns (spam often has repetition)
        words = content.lower().split()
        if words:
            word_freq = {}
            for word in words:
                word_freq[word] = word_freq.get(word, 0) + 1
            max_freq = max(word_freq.values())
            if max_freq / len(words) > 0.2:  # Single word > 20% of content
                return True
        
        # Check for common spam phrases
        spam_phrases = [
            'click here', 'buy now', 'limited offer', 'act now',
            'free money', 'guaranteed', '100% free'
        ]
        content_lower = content.lower()
        spam_count = sum(1 for phrase in spam_phrases if phrase in content_lower)
        if spam_count >= 3:
            return True
        
        return False
