import requests
from bs4 import BeautifulSoup
from typing import List, Dict
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BritannicaScraper:
    """Scrapes and processes Britannica France Land content"""

    def __init__(self, delay_seconds: float = 1.0):
        self.delay_seconds = delay_seconds
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) '
                          'Chrome/91.0.4472.124 Safari/537.36'
        })

        # Define the URLs and their section mappings
        self.urls = {
            'Land': 'https://www.britannica.com/place/France/Land',
            'The Hercynian massifs': 'https://www.britannica.com/place/France/The-Hercynian-massifs',
            'The great lowlands': 'https://www.britannica.com/place/France/The-great-lowlands',
            'The younger mountains and adjacent plains': 'https://www.britannica.com/place/France/The-younger-mountains-and-adjacent-plains',
            'Drainage': 'https://www.britannica.com/place/France/Drainage',
            'Soils': 'https://www.britannica.com/place/France/Soils',
            'Climate': 'https://www.britannica.com/place/France/Climate',
            'Plant and animal life': 'https://www.britannica.com/place/France/Plant-and-animal-life'
        }

    def scrape_page(self, url: str, section_name: str) -> Dict:
        """Scrape a single Britannica page"""
        try:
            logger.info(f"Scraping: {section_name} - {url}")
            response = self.session.get(url, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')

            # Extract main content
            content = self._extract_content(soup)

            # Extract subsections
            subsections = self._extract_subsections(soup)

            # Extract metadata
            metadata = self._extract_metadata(soup, url, section_name)

            time.sleep(self.delay_seconds)

            return {
                'url': url,
                'section': section_name,
                'content': content,
                'subsections': subsections,
                'metadata': metadata
            }

        except Exception as e:
            logger.error(f"Error scraping {url}: {str(e)}")
            return None

    def _extract_content(self, soup: BeautifulSoup) -> str:
        """Extract main content from the page"""
        # Try different content selectors
        content_selectors = [
            'div.article-body',
            'div.content',
            'div.article-content',
            'main',
            'article'
        ]

        content = ""
        for selector in content_selectors:
            content_div = soup.select_one(selector)
            if content_div:
                content = content_div.get_text(separator=' ', strip=True)
                break

        # If no specific content div found, try to get all paragraphs
        if not content:
            paragraphs = soup.find_all('p')
            content = ' '.join([p.get_text(strip=True) for p in paragraphs])

        return content

    def _extract_subsections(self, soup: BeautifulSoup) -> List[Dict]:
        """Extract subsections from the page"""
        subsections = []

        # Look for headings that indicate subsections
        headings = soup.find_all(['h2', 'h3', 'h4'])

        for heading in headings:
            if heading.get_text(strip=True):
                # Get content following this heading
                content_parts = []
                next_sibling = heading.next_sibling

                while next_sibling:
                    if next_sibling.name and next_sibling.name in ['h2', 'h3', 'h4']:
                        break
                    if next_sibling.name == 'p':
                        content_parts.append(next_sibling.get_text(strip=True))
                    next_sibling = next_sibling.next_sibling

                if content_parts:
                    subsections.append({
                        'title': heading.get_text(strip=True),
                        'content': ' '.join(content_parts)
                    })

        return subsections

    def _extract_metadata(self, soup: BeautifulSoup, url: str, section_name: str) -> Dict:
        """Extract metadata from the page"""
        metadata = {
            'url': url,
            'section': section_name,
            'domain': 'britannica.com',
            'topic': 'France Geography',
            'language': 'en'
        }

        # Try to extract title
        title_tag = soup.find('title')
        if title_tag:
            metadata['title'] = title_tag.get_text(strip=True)

        # Try to extract description
        description_tag = soup.find('meta', attrs={'name': 'description'})
        if description_tag:
            metadata['description'] = description_tag.get('content', '')

        # Try to extract keywords
        keywords_tag = soup.find('meta', attrs={'name': 'keywords'})
        if keywords_tag:
            metadata['keywords'] = keywords_tag.get('content', '')

        return metadata

    def scrape_all_pages(self) -> List[Dict]:
        """Scrape all defined URLs"""
        all_pages = []

        for section_name, url in self.urls.items():
            page_data = self.scrape_page(url, section_name)
            if page_data:
                all_pages.append(page_data)

        return all_pages
