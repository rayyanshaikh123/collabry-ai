"""
URL Content Extractor

Extracts clean content from web URLs with domain-based authority classification.
"""

import logging
import re
from typing import Optional, Dict, Any
from urllib.parse import urlparse
from dataclasses import dataclass

import requests
from bs4 import BeautifulSoup
import trafilatura

logger = logging.getLogger(__name__)


# Trusted domain classifications
TRUSTED_DOMAINS = {
    'high': [
        'ncert.nic.in',
        'khanacademy.org',
        'britannica.com',
        'wikipedia.org',
        '.edu',  # All .edu domains
        '.ac.uk',  # UK academic institutions
        'cbse.gov.in',
        'icse.org',
        'mit.edu',
        'stanford.edu',
        'ox.ac.uk',
        'cam.ac.uk',
    ],
    'medium': [
        'medium.com',
        'towardsdatascience.com',
        'scientificamerican.com',
        'nature.com',
        'sciencedirect.com',
        'springer.com',
        'arxiv.org',
    ],
    'low': []  # Default for unknown domains
}


@dataclass
class ExtractedContent:
    """Container for extracted web content."""
    text: str
    title: Optional[str] = None
    author: Optional[str] = None
    publication_date: Optional[str] = None
    url: str = ""
    domain: str = ""
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class URLExtractor:
    """Extract clean content from web URLs."""
    
    def __init__(self, timeout: int = 30, user_agent: Optional[str] = None):
        """
        Initialize URL extractor.
        
        Args:
            timeout: Request timeout in seconds
            user_agent: Custom user agent string
        """
        self.timeout = timeout
        self.user_agent = user_agent or (
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
            '(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        )
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': self.user_agent})
    
    def extract(self, url: str) -> ExtractedContent:
        """
        Extract content from URL.
        
        Args:
            url: Web URL to extract from
            
        Returns:
            ExtractedContent object
            
        Raises:
            Exception: If extraction fails
        """
        logger.info(f"Extracting content from: {url}")
        
        # Parse domain
        parsed = urlparse(url)
        domain = parsed.netloc
        
        try:
            # Fetch content
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            html = response.text
            
            # Try trafilatura first (cleaner extraction)
            extracted_text = trafilatura.extract(
                html,
                include_comments=False,
                include_tables=True,
                no_fallback=False
            )
            
            # Fallback to BeautifulSoup if trafilatura fails
            if not extracted_text or len(extracted_text) < 100:
                logger.warning(f"Trafilatura extraction insufficient, using BeautifulSoup")
                extracted_text = self._extract_with_beautifulsoup(html)
            
            # Extract metadata
            metadata = self._extract_metadata(html, url)
            
            return ExtractedContent(
                text=extracted_text or "",
                title=metadata.get('title'),
                author=metadata.get('author'),
                publication_date=metadata.get('date'),
                url=url,
                domain=domain,
                metadata=metadata
            )
        
        except requests.RequestException as e:
            logger.error(f"Failed to fetch URL {url}: {e}")
            raise Exception(f"Failed to fetch URL: {e}")
        except Exception as e:
            logger.error(f"Failed to extract content from {url}: {e}")
            raise Exception(f"Content extraction failed: {e}")
    
    def _extract_with_beautifulsoup(self, html: str) -> str:
        """Fallback extraction using BeautifulSoup."""
        soup = BeautifulSoup(html, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()
        
        # Get text from main content areas
        main_content = soup.find('main') or soup.find('article') or soup.find('body')
        
        if main_content:
            text = main_content.get_text(separator='\n', strip=True)
        else:
            text = soup.get_text(separator='\n', strip=True)
        
        # Clean up whitespace
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return '\n'.join(lines)
    
    def _extract_metadata(self, html: str, url: str) -> Dict[str, Any]:
        """Extract metadata from HTML."""
        soup = BeautifulSoup(html, 'html.parser')
        metadata = {'url': url}
        
        # Title
        title_tag = soup.find('title')
        if title_tag:
            metadata['title'] = title_tag.get_text().strip()
        
        # Meta tags
        meta_tags = {
            'author': ['author', 'article:author'],
            'date': ['article:published_time', 'datePublished', 'date'],
            'description': ['description', 'og:description'],
        }
        
        for key, names in meta_tags.items():
            for name in names:
                tag = soup.find('meta', attrs={'name': name}) or \
                      soup.find('meta', attrs={'property': name})
                if tag and tag.get('content'):
                    metadata[key] = tag.get('content').strip()
                    break
        
        return metadata
    
    def classify_domain_authority(self, url: str) -> str:
        """
        Classify authority level based on domain.
        
        Args:
            url: Web URL
            
        Returns:
            'high' | 'medium' | 'low'
        """
        domain = urlparse(url).netloc.lower()
        
        # Check high authority domains
        for trusted_domain in TRUSTED_DOMAINS['high']:
            if trusted_domain in domain:
                return 'high'
        
        # Check medium authority domains
        for trusted_domain in TRUSTED_DOMAINS['medium']:
            if trusted_domain in domain:
                return 'medium'
        
        # Default to low
        return 'low'
    
    def is_trusted_domain(self, url: str) -> bool:
        """Check if URL is from a trusted domain."""
        authority = self.classify_domain_authority(url)
        return authority in ['high', 'medium']
