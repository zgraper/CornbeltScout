"""
selenium_provider.py
--------------------
Fallback DuckDuckGo search provider using Selenium WebDriver.
"""

import time
import urllib.parse
from typing import Dict, List

from agriindex.utils.logging_utils import get_logger
from agriindex.search.duckduckgo_search import normalize_search_result

logger = get_logger(__name__)

def fetch_selenium_results(query: str, limit: int) -> List[Dict]:
    """
    Fetch normalized results from DuckDuckGo via Selenium.
    
    Returns an empty list on failure.
    """
    try:
        from selenium import webdriver
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
        from selenium.webdriver.chrome.options import Options
        from selenium.common.exceptions import TimeoutException, WebDriverException
    except ImportError:
        logger.error("Selenium package is not installed. Cannot use Selenium fallback. Install with: pip install selenium")
        return []

    logger.info("Initializing Selenium WebDriver for DuckDuckGo search fallback.")
    
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1920,1080")
    # Add a common user agent to avoid basic blocks
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")

    driver = None
    raw_results = []
    
    try:
        driver = webdriver.Chrome(options=options)
        driver.set_page_load_timeout(30)
        
        # Build search URL directly instead of going to homepage to save time and reduce flakiness
        encoded_query = urllib.parse.quote_plus(query)
        search_url = f"https://duckduckgo.com/html/?q={encoded_query}"
        
        logger.debug("Selenium fetching URL: %s", search_url)
        driver.get(search_url)
        
        # The HTML version of DDG has results with class 'result' or 'result__body'
        # Wait for at least one result
        try:
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, ".result__body"))
            )
        except TimeoutException:
            logger.warning("Selenium timeout waiting for search results on HTML page.")
            return []
            
        result_elements = driver.find_elements(By.CSS_SELECTOR, ".result__body")
        
        for elem in result_elements:
            if len(raw_results) >= limit:
                break
                
            try:
                title_elem = elem.find_element(By.CSS_SELECTOR, ".result__title a")
                title = title_elem.text.strip()
                url = title_elem.get_attribute("href")
                
                snippet = ""
                try:
                    snippet_elem = elem.find_element(By.CSS_SELECTOR, ".result__snippet")
                    snippet = snippet_elem.text.strip()
                except Exception:
                    pass
                    
                if title and url:
                    # Sometimes the HTML DDG redirects URLs, we try to parse the actual URL if it's a redirect
                    if "duckduckgo.com/l/?" in url:
                        parsed = urllib.parse.urlparse(url)
                        query_params = urllib.parse.parse_qs(parsed.query)
                        if "uddg" in query_params:
                            url = query_params["uddg"][0]
                            
                    raw_results.append({
                        "title": title,
                        "href": url,
                        "body": snippet
                    })
            except Exception as e:
                logger.debug("Failed to parse a Selenium search result element: %s", e)
                continue
                
    except WebDriverException as e:
        logger.error("Selenium WebDriver error: %s", e)
    except Exception as e:
        logger.error("Unexpected error in Selenium provider: %s", e)
    finally:
        if driver:
            try:
                driver.quit()
            except Exception as e:
                logger.debug("Error closing Selenium driver: %s", e)

    results: List[Dict] = []
    seen_urls = set()
    rank = 1

    for raw_index, raw in enumerate(raw_results, start=1):
        record = normalize_search_result(raw, query=query, rank=rank)
        if not record:
            continue
            
        url = record["url"]
        if url in seen_urls:
            continue

        seen_urls.add(url)
        results.append(record)
        rank += 1

    return results
