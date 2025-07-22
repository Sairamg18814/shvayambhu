"""
Always-On Web Connectivity Monitor

Implements continuous web monitoring with news aggregation, 
real-time updates, and intelligent content filtering.
"""

import asyncio
import aiohttp
import feedparser
import json
import logging
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import hashlib
from pathlib import Path
import re
from urllib.parse import urljoin, urlparse
import time
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


@dataclass
class WebMonitorConfig:
    """Configuration for web monitoring"""
    # Update frequencies (in seconds)
    news_update_frequency: int = 300     # 5 minutes
    social_update_frequency: int = 600   # 10 minutes
    general_update_frequency: int = 1800 # 30 minutes
    
    # Connection settings
    max_concurrent_requests: int = 10
    request_timeout: int = 30
    retry_attempts: int = 3
    backoff_factor: float = 2.0
    
    # Content filtering
    min_content_length: int = 100
    max_content_length: int = 50000
    language_filter: List[str] = field(default_factory=lambda: ['en'])
    
    # Rate limiting
    requests_per_minute: int = 60
    rate_limit_window: int = 60
    
    # Storage
    cache_dir: str = "data/web_cache"
    max_cache_age: int = 86400  # 24 hours
    max_cache_size: int = 1000  # Max cached items per source


@dataclass
class WebContent:
    """Represents web content item"""
    id: str
    url: str
    title: str
    content: str
    source: str
    published_at: datetime
    retrieved_at: datetime
    content_type: str  # news, social, article, etc.
    relevance_score: float
    keywords: List[str]
    summary: str
    language: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class RateLimiter:
    """Rate limiting for web requests"""
    
    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.requests = deque()
    
    async def acquire(self):
        """Wait if necessary to respect rate limit"""
        now = time.time()
        
        # Remove old requests outside the window
        while self.requests and now - self.requests[0] >= 60:
            self.requests.popleft()
        
        # Wait if we're at the limit
        if len(self.requests) >= self.requests_per_minute:
            sleep_time = 60 - (now - self.requests[0])
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
        
        self.requests.append(now)


class ContentFilter:
    """Filters and scores web content for relevance"""
    
    def __init__(self):
        # Keywords indicating high-value content
        self.high_value_keywords = {
            'ai': 2.0,
            'artificial intelligence': 2.5,
            'machine learning': 2.0,
            'consciousness': 3.0,
            'neural networks': 1.8,
            'deep learning': 1.8,
            'robotics': 1.5,
            'technology': 1.0,
            'science': 1.2,
            'research': 1.3,
            'breakthrough': 2.0,
            'innovation': 1.5,
            'discovery': 1.8
        }
        
        # Keywords indicating low-value content
        self.low_value_keywords = {
            'advertisement': -2.0,
            'sponsored': -1.5,
            'click here': -2.0,
            'buy now': -2.0,
            'sale': -1.0,
            'discount': -1.0,
            'celebrity': -1.5,
            'gossip': -2.0
        }
        
        # Spam detection patterns
        self.spam_patterns = [
            r'win \$\d+',
            r'free money',
            r'click here now',
            r'limited time offer',
            r'act now'
        ]
    
    def score_content(self, content: str, title: str = "", url: str = "") -> float:
        """Score content for relevance (0.0 to 1.0)"""
        
        score = 0.5  # Base score
        text_to_analyze = (title + " " + content).lower()
        
        # Keyword scoring
        for keyword, weight in self.high_value_keywords.items():
            if keyword in text_to_analyze:
                score += weight * 0.1
        
        for keyword, weight in self.low_value_keywords.items():
            if keyword in text_to_analyze:
                score += weight * 0.1
        
        # Spam detection
        for pattern in self.spam_patterns:
            if re.search(pattern, text_to_analyze):
                score -= 0.3
        
        # URL quality indicators
        if url:
            parsed_url = urlparse(url)
            domain = parsed_url.netloc.lower()
            
            # Boost trusted news sources
            trusted_domains = [
                'arxiv.org', 'nature.com', 'science.org', 'ieee.org',
                'acm.org', 'mit.edu', 'stanford.edu', 'openai.com'
            ]
            
            if any(trusted in domain for trusted in trusted_domains):
                score += 0.3
        
        # Content quality indicators
        word_count = len(content.split())
        if word_count < 50:
            score -= 0.2  # Too short
        elif word_count > 5000:
            score -= 0.1  # Might be too verbose
        
        # Ensure score is in valid range
        return max(0.0, min(1.0, score))
    
    def extract_keywords(self, content: str, max_keywords: int = 10) -> List[str]:
        """Extract relevant keywords from content"""
        
        # Simple keyword extraction (could be enhanced with NLP)
        words = re.findall(r'\b[a-zA-Z]{3,}\b', content.lower())
        
        # Count word frequencies
        word_freq = defaultdict(int)
        for word in words:
            word_freq[word] += 1
        
        # Filter common words
        stop_words = {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'this', 'that', 'are', 'was', 'were',
            'been', 'have', 'has', 'had', 'will', 'would', 'could',
            'should', 'may', 'might', 'can', 'must', 'shall'
        }
        
        # Get top keywords excluding stop words
        keywords = [
            word for word, freq in sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
            if word not in stop_words and len(word) > 3
        ]
        
        return keywords[:max_keywords]
    
    def generate_summary(self, content: str, max_length: int = 200) -> str:
        """Generate simple extractive summary"""
        
        sentences = re.split(r'[.!?]+', content)
        if len(sentences) <= 2:
            return content[:max_length]
        
        # Score sentences by keyword presence
        sentence_scores = []
        for sentence in sentences:
            score = 0
            sentence_lower = sentence.lower()
            
            for keyword in self.high_value_keywords:
                if keyword in sentence_lower:
                    score += self.high_value_keywords[keyword]
            
            sentence_scores.append((score, sentence.strip()))
        
        # Get top sentences
        sentence_scores.sort(reverse=True, key=lambda x: x[0])
        top_sentences = [sent for score, sent in sentence_scores[:3] if sent]
        
        summary = '. '.join(top_sentences)
        
        # Truncate if too long
        if len(summary) > max_length:
            summary = summary[:max_length].rsplit(' ', 1)[0] + '...'
        
        return summary


class NewsAggregator:
    """Aggregates news from RSS feeds and news APIs"""
    
    def __init__(self, config: WebMonitorConfig):
        self.config = config
        self.rate_limiter = RateLimiter(config.requests_per_minute)
        self.content_filter = ContentFilter()
        
        # News sources
        self.rss_feeds = [
            'http://feeds.bbci.co.uk/news/technology/rss.xml',
            'https://feeds.reuters.com/reuters/technologyNews',
            'https://rss.cnn.com/rss/edition.rss',
            'https://feeds.washingtonpost.com/rss/business/technology',
            'https://www.wired.com/feed/rss',
            'https://techcrunch.com/feed/',
            'https://feeds.arstechnica.com/arstechnica/index'
        ]
    
    async def fetch_news(self, session: aiohttp.ClientSession) -> List[WebContent]:
        """Fetch news from all configured sources"""
        
        all_content = []
        
        # Fetch RSS feeds
        rss_tasks = [self._fetch_rss_feed(session, feed_url) for feed_url in self.rss_feeds]
        rss_results = await asyncio.gather(*rss_tasks, return_exceptions=True)
        
        for result in rss_results:
            if isinstance(result, list):
                all_content.extend(result)
            elif isinstance(result, Exception):
                logger.error(f"Error fetching RSS: {result}")
        
        logger.info(f"Fetched {len(all_content)} news items")
        return all_content
    
    async def _fetch_rss_feed(self, session: aiohttp.ClientSession, feed_url: str) -> List[WebContent]:
        """Fetch and parse RSS feed"""
        
        await self.rate_limiter.acquire()
        
        try:
            async with session.get(feed_url, timeout=self.config.request_timeout) as response:
                if response.status != 200:
                    logger.warning(f"RSS feed returned {response.status}: {feed_url}")
                    return []
                
                xml_content = await response.text()
                
                # Parse RSS feed
                feed = feedparser.parse(xml_content)
                content_items = []
                
                for entry in feed.entries:
                    try:
                        # Extract content
                        content_text = ""
                        if hasattr(entry, 'content'):
                            content_text = entry.content[0].value if entry.content else ""
                        elif hasattr(entry, 'summary'):
                            content_text = entry.summary
                        elif hasattr(entry, 'description'):
                            content_text = entry.description
                        
                        # Clean HTML tags
                        content_text = re.sub(r'<[^>]+>', '', content_text)
                        
                        # Skip if content too short
                        if len(content_text) < self.config.min_content_length:
                            continue
                        
                        # Create WebContent item
                        published_date = datetime.now()
                        if hasattr(entry, 'published_parsed') and entry.published_parsed:
                            published_date = datetime(*entry.published_parsed[:6])
                        
                        url = entry.link if hasattr(entry, 'link') else feed_url
                        title = entry.title if hasattr(entry, 'title') else "No title"
                        
                        # Score and filter content
                        relevance_score = self.content_filter.score_content(content_text, title, url)
                        
                        if relevance_score < 0.3:  # Skip low-relevance content
                            continue
                        
                        content_item = WebContent(
                            id=hashlib.md5(f"{url}{title}".encode()).hexdigest(),
                            url=url,
                            title=title,
                            content=content_text[:self.config.max_content_length],
                            source=urlparse(feed_url).netloc,
                            published_at=published_date,
                            retrieved_at=datetime.now(),
                            content_type="news",
                            relevance_score=relevance_score,
                            keywords=self.content_filter.extract_keywords(content_text),
                            summary=self.content_filter.generate_summary(content_text),
                            language="en",  # Could be detected
                            metadata={'feed_url': feed_url}
                        )
                        
                        content_items.append(content_item)
                        
                    except Exception as e:
                        logger.error(f"Error processing RSS entry: {e}")
                        continue
                
                return content_items
                
        except Exception as e:
            logger.error(f"Error fetching RSS feed {feed_url}: {e}")
            return []


class WebCrawler:
    """Crawls web pages for relevant content"""
    
    def __init__(self, config: WebMonitorConfig):
        self.config = config
        self.rate_limiter = RateLimiter(config.requests_per_minute)
        self.content_filter = ContentFilter()
        self.visited_urls = set()
        
        # Target domains for crawling
        self.target_domains = [
            'arxiv.org',
            'openai.com',
            'deepmind.com',
            'ai.google',
            'research.facebook.com',
            'blog.openai.com'
        ]
    
    async def crawl_sites(self, session: aiohttp.ClientSession, urls: List[str]) -> List[WebContent]:
        """Crawl specific URLs for content"""
        
        crawl_tasks = [self._crawl_url(session, url) for url in urls if url not in self.visited_urls]
        results = await asyncio.gather(*crawl_tasks, return_exceptions=True)
        
        all_content = []
        for result in results:
            if isinstance(result, WebContent):
                all_content.append(result)
            elif isinstance(result, Exception):
                logger.error(f"Crawling error: {result}")
        
        logger.info(f"Crawled {len(all_content)} pages")
        return all_content
    
    async def _crawl_url(self, session: aiohttp.ClientSession, url: str) -> Optional[WebContent]:
        """Crawl a single URL"""
        
        await self.rate_limiter.acquire()
        self.visited_urls.add(url)
        
        try:
            headers = {
                'User-Agent': 'Shvayambhu-AI/1.0 (Educational Research Bot)',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
            }
            
            async with session.get(url, headers=headers, timeout=self.config.request_timeout) as response:
                if response.status != 200:
                    return None
                
                html_content = await response.text()
                
                # Extract text content (basic HTML parsing)
                # Remove script and style elements
                html_content = re.sub(r'<script.*?</script>', '', html_content, flags=re.DOTALL)
                html_content = re.sub(r'<style.*?</style>', '', html_content, flags=re.DOTALL)
                
                # Extract title
                title_match = re.search(r'<title>(.*?)</title>', html_content, re.IGNORECASE)
                title = title_match.group(1) if title_match else "No title"
                
                # Extract main content (simple approach)
                text_content = re.sub(r'<[^>]+>', '', html_content)
                text_content = re.sub(r'\s+', ' ', text_content).strip()
                
                # Skip if content too short or too long
                if (len(text_content) < self.config.min_content_length or 
                    len(text_content) > self.config.max_content_length):
                    return None
                
                # Score content
                relevance_score = self.content_filter.score_content(text_content, title, url)
                
                if relevance_score < 0.4:  # Higher threshold for crawled content
                    return None
                
                content_item = WebContent(
                    id=hashlib.md5(url.encode()).hexdigest(),
                    url=url,
                    title=title,
                    content=text_content,
                    source=urlparse(url).netloc,
                    published_at=datetime.now(),  # Unknown, use current time
                    retrieved_at=datetime.now(),
                    content_type="article",
                    relevance_score=relevance_score,
                    keywords=self.content_filter.extract_keywords(text_content),
                    summary=self.content_filter.generate_summary(text_content),
                    language="en",
                    metadata={'crawled': True}
                )
                
                return content_item
                
        except Exception as e:
            logger.error(f"Error crawling {url}: {e}")
            return None


class ContentCache:
    """Caches web content to avoid redundant fetches"""
    
    def __init__(self, config: WebMonitorConfig):
        self.config = config
        self.cache_dir = Path(config.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # In-memory cache for fast access
        self.memory_cache: Dict[str, WebContent] = {}
        self.cache_hits = 0
        self.cache_misses = 0
    
    def get_cache_key(self, url: str) -> str:
        """Generate cache key for URL"""
        return hashlib.md5(url.encode()).hexdigest()
    
    def is_cached(self, url: str, max_age: Optional[int] = None) -> bool:
        """Check if content is cached and still valid"""
        
        cache_key = self.get_cache_key(url)
        
        # Check memory cache first
        if cache_key in self.memory_cache:
            content = self.memory_cache[cache_key]
            age = (datetime.now() - content.retrieved_at).total_seconds()
            
            max_age = max_age or self.config.max_cache_age
            if age < max_age:
                return True
            else:
                # Remove expired content
                del self.memory_cache[cache_key]
        
        # Check disk cache
        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    cached_data = json.load(f)
                
                retrieved_at = datetime.fromisoformat(cached_data['retrieved_at'])
                age = (datetime.now() - retrieved_at).total_seconds()
                
                max_age = max_age or self.config.max_cache_age
                if age < max_age:
                    # Load into memory cache
                    content = self._deserialize_content(cached_data)
                    self.memory_cache[cache_key] = content
                    return True
                else:
                    # Remove expired cache file
                    cache_file.unlink()
            except Exception as e:
                logger.error(f"Error reading cache file {cache_file}: {e}")
        
        return False
    
    def get_cached_content(self, url: str) -> Optional[WebContent]:
        """Retrieve cached content"""
        
        cache_key = self.get_cache_key(url)
        
        if cache_key in self.memory_cache:
            self.cache_hits += 1
            return self.memory_cache[cache_key]
        
        self.cache_misses += 1
        return None
    
    def cache_content(self, content: WebContent):
        """Cache web content"""
        
        cache_key = self.get_cache_key(content.url)
        
        # Store in memory
        self.memory_cache[cache_key] = content
        
        # Store on disk
        cache_file = self.cache_dir / f"{cache_key}.json"
        try:
            with open(cache_file, 'w') as f:
                json.dump(self._serialize_content(content), f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error writing cache file {cache_file}: {e}")
        
        # Cleanup old cache if needed
        self._cleanup_cache()
    
    def _serialize_content(self, content: WebContent) -> Dict[str, Any]:
        """Serialize WebContent to dictionary"""
        
        return {
            'id': content.id,
            'url': content.url,
            'title': content.title,
            'content': content.content,
            'source': content.source,
            'published_at': content.published_at.isoformat(),
            'retrieved_at': content.retrieved_at.isoformat(),
            'content_type': content.content_type,
            'relevance_score': content.relevance_score,
            'keywords': content.keywords,
            'summary': content.summary,
            'language': content.language,
            'metadata': content.metadata
        }
    
    def _deserialize_content(self, data: Dict[str, Any]) -> WebContent:
        """Deserialize dictionary to WebContent"""
        
        return WebContent(
            id=data['id'],
            url=data['url'],
            title=data['title'],
            content=data['content'],
            source=data['source'],
            published_at=datetime.fromisoformat(data['published_at']),
            retrieved_at=datetime.fromisoformat(data['retrieved_at']),
            content_type=data['content_type'],
            relevance_score=data['relevance_score'],
            keywords=data['keywords'],
            summary=data['summary'],
            language=data['language'],
            metadata=data['metadata']
        )
    
    def _cleanup_cache(self):
        """Remove old cache files to maintain size limits"""
        
        # Get all cache files
        cache_files = list(self.cache_dir.glob("*.json"))
        
        if len(cache_files) > self.config.max_cache_size:
            # Sort by modification time and remove oldest
            cache_files.sort(key=lambda x: x.stat().st_mtime)
            files_to_remove = cache_files[:-self.config.max_cache_size]
            
            for file_path in files_to_remove:
                try:
                    file_path.unlink()
                except Exception as e:
                    logger.error(f"Error removing cache file {file_path}: {e}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0
        
        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': hit_rate,
            'memory_cache_size': len(self.memory_cache),
            'disk_cache_files': len(list(self.cache_dir.glob("*.json")))
        }


class WebMonitor:
    """Main web monitoring orchestrator"""
    
    def __init__(self, config: WebMonitorConfig):
        self.config = config
        self.news_aggregator = NewsAggregator(config)
        self.web_crawler = WebCrawler(config)
        self.content_cache = ContentCache(config)
        
        # Monitoring state
        self.is_running = False
        self.last_update_times = {}
        self.content_buffer: List[WebContent] = []
        self.update_callbacks = []
        
        # Statistics
        self.stats = {
            'total_content_fetched': 0,
            'content_by_source': defaultdict(int),
            'average_relevance_score': 0.0,
            'last_update_time': None,
            'errors_count': 0,
            'uptime_start': datetime.now()
        }
    
    def add_update_callback(self, callback):
        """Add callback for when new content is available"""
        self.update_callbacks.append(callback)
    
    async def start_monitoring(self):
        """Start continuous web monitoring"""
        
        self.is_running = True
        self.stats['uptime_start'] = datetime.now()
        
        logger.info("Starting continuous web monitoring")
        
        # Create session with connection pooling
        connector = aiohttp.TCPConnector(limit=self.config.max_concurrent_requests)
        session = aiohttp.ClientSession(connector=connector)
        
        try:
            # Start monitoring tasks
            tasks = [
                self._news_monitoring_loop(session),
                self._web_crawling_loop(session),
                self._content_processing_loop()
            ]
            
            await asyncio.gather(*tasks, return_exceptions=True)
            
        except Exception as e:
            logger.error(f"Error in web monitoring: {e}")
            self.stats['errors_count'] += 1
        finally:
            await session.close()
    
    async def stop_monitoring(self):
        """Stop web monitoring"""
        
        self.is_running = False
        logger.info("Web monitoring stopped")
    
    async def _news_monitoring_loop(self, session: aiohttp.ClientSession):
        """Continuous news monitoring loop"""
        
        while self.is_running:
            try:
                # Check if it's time to update
                last_update = self.last_update_times.get('news', datetime.min)
                if (datetime.now() - last_update).total_seconds() >= self.config.news_update_frequency:
                    
                    logger.info("Fetching news updates")
                    news_content = await self.news_aggregator.fetch_news(session)
                    
                    # Filter out cached content
                    new_content = []
                    for content in news_content:
                        if not self.content_cache.is_cached(content.url):
                            new_content.append(content)
                            self.content_cache.cache_content(content)
                    
                    # Add to buffer
                    self.content_buffer.extend(new_content)
                    
                    # Update statistics
                    self.stats['total_content_fetched'] += len(new_content)
                    for content in new_content:
                        self.stats['content_by_source'][content.source] += 1
                    
                    self.last_update_times['news'] = datetime.now()
                    
                    if new_content:
                        logger.info(f"Added {len(new_content)} new news items")
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in news monitoring loop: {e}")
                self.stats['errors_count'] += 1
                await asyncio.sleep(300)  # Wait 5 minutes on error
    
    async def _web_crawling_loop(self, session: aiohttp.ClientSession):
        """Continuous web crawling loop"""
        
        # URLs to crawl periodically
        crawl_urls = [
            'https://openai.com/blog',
            'https://deepmind.com/blog',
            'https://arxiv.org/list/cs.AI/recent',
            'https://ai.googleblog.com/'
        ]
        
        while self.is_running:
            try:
                last_update = self.last_update_times.get('crawling', datetime.min)
                if (datetime.now() - last_update).total_seconds() >= self.config.general_update_frequency:
                    
                    logger.info("Starting web crawling")
                    crawled_content = await self.web_crawler.crawl_sites(session, crawl_urls)
                    
                    # Filter new content
                    new_content = []
                    for content in crawled_content:
                        if not self.content_cache.is_cached(content.url):
                            new_content.append(content)
                            self.content_cache.cache_content(content)
                    
                    self.content_buffer.extend(new_content)
                    
                    # Update statistics
                    self.stats['total_content_fetched'] += len(new_content)
                    for content in new_content:
                        self.stats['content_by_source'][content.source] += 1
                    
                    self.last_update_times['crawling'] = datetime.now()
                    
                    if new_content:
                        logger.info(f"Added {len(new_content)} new crawled items")
                
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                logger.error(f"Error in crawling loop: {e}")
                self.stats['errors_count'] += 1
                await asyncio.sleep(600)  # Wait 10 minutes on error
    
    async def _content_processing_loop(self):
        """Process content buffer and notify callbacks"""
        
        while self.is_running:
            try:
                if self.content_buffer:
                    # Process content in batches
                    batch_size = 10
                    batch = self.content_buffer[:batch_size]
                    self.content_buffer = self.content_buffer[batch_size:]
                    
                    # Update average relevance score
                    if batch:
                        avg_score = sum(content.relevance_score for content in batch) / len(batch)
                        self.stats['average_relevance_score'] = avg_score
                        self.stats['last_update_time'] = datetime.now()
                        
                        # Notify callbacks
                        for callback in self.update_callbacks:
                            try:
                                await callback(batch)
                            except Exception as e:
                                logger.error(f"Error in update callback: {e}")
                
                await asyncio.sleep(30)  # Process every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in content processing loop: {e}")
                await asyncio.sleep(60)
    
    def get_recent_content(self, hours: int = 1, min_relevance: float = 0.5) -> List[WebContent]:
        """Get recent content above relevance threshold"""
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        # Check memory cache and content buffer
        recent_content = [
            content for content in list(self.content_cache.memory_cache.values()) + self.content_buffer
            if (content.retrieved_at >= cutoff_time and 
                content.relevance_score >= min_relevance)
        ]
        
        # Sort by relevance score descending
        recent_content.sort(key=lambda x: x.relevance_score, reverse=True)
        
        return recent_content
    
    def get_monitoring_stats(self) -> Dict[str, Any]:
        """Get comprehensive monitoring statistics"""
        
        uptime = (datetime.now() - self.stats['uptime_start']).total_seconds()
        cache_stats = self.content_cache.get_cache_stats()
        
        return {
            'is_running': self.is_running,
            'uptime_seconds': uptime,
            'total_content_fetched': self.stats['total_content_fetched'],
            'content_by_source': dict(self.stats['content_by_source']),
            'average_relevance_score': self.stats['average_relevance_score'],
            'last_update_time': self.stats['last_update_time'].isoformat() if self.stats['last_update_time'] else None,
            'errors_count': self.stats['errors_count'],
            'content_buffer_size': len(self.content_buffer),
            'cache_stats': cache_stats,
            'next_update_times': {
                source: (last_update + timedelta(seconds=freq)).isoformat()
                for source, last_update in self.last_update_times.items()
                for freq in [self.config.news_update_frequency if source == 'news' else self.config.general_update_frequency]
            }
        }


# Testing and example usage
async def test_web_monitor():
    """Test the web monitoring system"""
    
    config = WebMonitorConfig(
        news_update_frequency=60,  # 1 minute for testing
        general_update_frequency=120,  # 2 minutes for testing
        max_concurrent_requests=5
    )
    
    monitor = WebMonitor(config)
    
    # Add callback to print new content
    async def content_callback(content_batch: List[WebContent]):
        print(f"Received {len(content_batch)} new content items:")
        for content in content_batch[:2]:  # Show first 2
            print(f"  - {content.title[:60]}... (score: {content.relevance_score:.2f})")
    
    monitor.add_update_callback(content_callback)
    
    # Start monitoring for a short time
    monitoring_task = asyncio.create_task(monitor.start_monitoring())
    
    # Let it run for 5 minutes
    await asyncio.sleep(300)
    
    await monitor.stop_monitoring()
    
    # Print final stats
    stats = monitor.get_monitoring_stats()
    print("\nFinal monitoring statistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    asyncio.run(test_web_monitor())