"""
Authority Classifier

Classifies authority level based on source type and metadata.
"""

import logging
from typing import Optional, Dict, Any
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


# Trusted domains for URL sources
TRUSTED_DOMAINS = {
    'high': [
        'ncert.nic.in', 'khanacademy.org', 'britannica.com',
        'wikipedia.org', '.edu', '.ac.uk', 'cbse.gov.in',
        'icse.org', 'mit.edu', 'stanford.edu', 'ox.ac.uk', 'cam.ac.uk',
    ],
    'medium': [
        'medium.com', 'towardsdatascience.com', 'scientificamerican.com',
        'nature.com', 'sciencedirect.com', 'springer.com', 'arxiv.org',
    ],
}

# Trusted publishers for PDF sources
TRUSTED_PUBLISHERS = {
    'high': [
        'NCERT', 'CBSE', 'ICSE', 'Pearson Education',
        'McGraw-Hill Education', 'Oxford University Press',
        'Cambridge University Press', 'Wiley', 'Springer', 'Elsevier',
    ],
    'medium': [
        'Cengage Learning', 'Houghton Mifflin', 'Macmillan', 'Scholastic',
    ],
}


class AuthorityClassifier:
    """Classify authority level of content sources."""
    
    def __init__(self):
        """Initialize authority classifier."""
        pass
    
    def classify(
        self,
        source_type: str,
        metadata: Dict[str, Any],
        user_override: Optional[str] = None
    ) -> str:
        """
        Classify authority level.
        
        Args:
            source_type: 'url' | 'text' | 'pdf'
            metadata: Source metadata
            user_override: User-specified authority level (takes precedence)
            
        Returns:
            'high' | 'medium' | 'low'
        """
        # User override takes precedence
        if user_override and user_override in ['high', 'medium', 'low']:
            logger.info(f"Using user-specified authority: {user_override}")
            return user_override
        
        # Classify based on source type
        if source_type == 'url':
            return self._classify_url(metadata)
        elif source_type == 'pdf':
            return self._classify_pdf(metadata)
        elif source_type == 'text':
            return self._classify_text(metadata)
        else:
            logger.warning(f"Unknown source type: {source_type}, defaulting to 'low'")
            return 'low'
    
    def _classify_url(self, metadata: Dict[str, Any]) -> str:
        """Classify URL source authority."""
        url = metadata.get('url', '')
        if not url:
            return 'low'
        
        domain = urlparse(url).netloc.lower()
        
        # Check high authority domains
        for trusted_domain in TRUSTED_DOMAINS['high']:
            if trusted_domain in domain:
                logger.info(f"High authority domain detected: {domain}")
                return 'high'
        
        # Check medium authority domains
        for trusted_domain in TRUSTED_DOMAINS['medium']:
            if trusted_domain in domain:
                logger.info(f"Medium authority domain detected: {domain}")
                return 'medium'
        
        # Default to low
        logger.info(f"Unknown domain, defaulting to low authority: {domain}")
        return 'low'
    
    def _classify_pdf(self, metadata: Dict[str, Any]) -> str:
        """Classify PDF source authority."""
        publisher = metadata.get('publisher', '')
        
        # Check for DOI (indicates peer-reviewed)
        if metadata.get('doi'):
            logger.info(f"DOI found, classifying as medium authority")
            return 'medium'
        
        # Check publisher
        if publisher:
            publisher_lower = publisher.lower()
            
            # Check high authority publishers
            for trusted_pub in TRUSTED_PUBLISHERS['high']:
                if trusted_pub.lower() in publisher_lower:
                    logger.info(f"High authority publisher detected: {publisher}")
                    return 'high'
            
            # Check medium authority publishers
            for trusted_pub in TRUSTED_PUBLISHERS['medium']:
                if trusted_pub.lower() in publisher_lower:
                    logger.info(f"Medium authority publisher detected: {publisher}")
                    return 'medium'
        
        # Check for ISBN (indicates published book)
        if metadata.get('isbn'):
            logger.info(f"ISBN found, classifying as medium authority")
            return 'medium'
        
        # Default to low
        logger.info(f"No trusted publisher/DOI/ISBN, defaulting to low authority")
        return 'low'
    
    def _classify_text(self, metadata: Dict[str, Any]) -> str:
        """Classify text source authority."""
        # For text sources, default to user-specified or low
        authority = metadata.get('authority_level', 'low')
        
        if authority not in ['high', 'medium', 'low']:
            logger.warning(f"Invalid authority level: {authority}, defaulting to 'low'")
            return 'low'
        
        logger.info(f"Text source authority: {authority}")
        return authority
    
    def get_authority_weight(self, authority_level: str) -> float:
        """
        Get numeric weight for authority level.
        
        Args:
            authority_level: 'high' | 'medium' | 'low'
            
        Returns:
            Weight (0.0-1.0)
        """
        weights = {
            'high': 1.0,
            'medium': 0.6,
            'low': 0.3
        }
        return weights.get(authority_level, 0.3)
