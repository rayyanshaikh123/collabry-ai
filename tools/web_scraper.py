# tools/web_scrape.py
import requests
from bs4 import BeautifulSoup
import re

def web_scrape(url: str, llm=None):
    """
    Scrape a webpage and return both summary and full text.
    The full text is stored for follow-up queries without re-scraping.
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, timeout=10, headers=headers)
        response.raise_for_status()
        html = response.text
    except Exception as e:
        return {"error": f"Failed to fetch URL: {e}"}

    soup = BeautifulSoup(html, "html.parser")

    # Remove only script and style elements (keep nav, footer, header for links)
    for script in soup(["script", "style"]):
        script.decompose()

    # Extract structured data
    text_parts = []
    links_data = []
    
    # Get title
    title = soup.find('title')
    if title:
        text_parts.append(f"Title: {title.get_text().strip()}")
        text_parts.append("="*50)
    
    # Extract all links (social, navigation, etc.)
    all_links = soup.find_all('a', href=True)
    for link in all_links:
        href = link.get('href', '').strip()
        text = link.get_text().strip()
        # Filter out empty or javascript links
        if href and not href.startswith('javascript:') and not href.startswith('#'):
            # Make relative URLs absolute
            if href.startswith('/'):
                from urllib.parse import urljoin
                href = urljoin(url, href)
            links_data.append(f"{text}: {href}" if text else href)
    
    if links_data:
        text_parts.append("\n--- Links Found ---")
        for link in links_data[:50]:  # Limit to 50 links
            text_parts.append(f"• {link}")
        text_parts.append("")
    
    # Get main content (prioritize main, article, or body)
    main_content = soup.find('main') or soup.find('article') or soup.find('body')
    
    if main_content:
        text_parts.append("--- Main Content ---")
        
        # Extract sections with specific patterns (projects, timeline, achievements, etc.)
        for section in main_content.find_all(['section', 'div'], class_=True):
            section_classes = ' '.join(section.get('class', []))
            section_id = section.get('id', '')
            
            # Detect common section patterns
            if any(keyword in section_classes.lower() + section_id.lower() 
                   for keyword in ['project', 'timeline', 'achievement', 'experience', 'education', 'skill']):
                section_name = section_id or section_classes.split()[0]
                text_parts.append(f"\n### Section: {section_name} ###")
            
            # Extract headings and content
            for elem in section.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'p', 'li', 'span']):
                text = elem.get_text().strip()
                if text and len(text) > 1:  # Skip single chars
                    if elem.name in ['h1', 'h2', 'h3', 'h4', 'h5']:
                        text_parts.append(f"\n{text}\n{'-'*min(len(text), 50)}")
                    else:
                        text_parts.append(text)
        
        # If no sections found, extract all text elements
        if len(text_parts) <= 2:  # Only title added
            for elem in main_content.find_all(['h1', 'h2', 'h3', 'p', 'li', 'span']):
                text = elem.get_text().strip()
                if text:
                    if elem.name in ['h1', 'h2', 'h3']:
                        text_parts.append(f"\n{text}\n{'-'*min(len(text), 50)}")
                    else:
                        text_parts.append(text)
    
    # Extract footer (social links, contact info)
    footer = soup.find('footer')
    if footer:
        text_parts.append("\n--- Footer ---")
        footer_links = footer.find_all('a', href=True)
        for link in footer_links:
            href = link.get('href', '').strip()
            text = link.get_text().strip() or link.get('aria-label', '') or 'Link'
            if href:
                if href.startswith('/'):
                    from urllib.parse import urljoin
                    href = urljoin(url, href)
                text_parts.append(f"• {text}: {href}")
        
        # Get any additional footer text
        footer_text = footer.get_text(separator=' ', strip=True)
        if footer_text and len(footer_text) < 500:
            text_parts.append(f"Footer text: {footer_text}")
    
    # Fallback to all text
    if len(text_parts) <= 2:
        text_parts = [t.strip() for t in soup.stripped_strings if t.strip()]
    
    full_text = "\n".join(text_parts)
    
    # Clean up excessive whitespace
    full_text = re.sub(r'\n{3,}', '\n\n', full_text)

    if not llm:
        return {"url": url, "text": full_text, "full_text": full_text}

    # Summarize via LLM (use first 10000 chars for summary)
    summary_text = full_text[:10000] if len(full_text) > 10000 else full_text
    
    prompt = f"""
Summarize the following webpage content. Include key sections, main topics, and important details:

{summary_text}

Provide a concise but informative summary.
"""

    # Use invoke (LangChain LLM base) for a single prompt string
    try:
        summary = llm.invoke(prompt)
    except Exception:
        # Fallback to _call if invoke fails
        try:
            summary = llm._call(prompt)
        except Exception as e:
            return {"error": f"LLM summarization failed: {e}"}

    return {
        "url": url,
        "summary": summary,
        "full_text": full_text,
        "text_length": len(full_text)
    }


TOOL = {
    "name": "web_scrape",
    "func": web_scrape,
    "description": "Scrape and extract full content from a specific webpage URL. Returns summary and full text for follow-up questions."
}
