"""
Privacy-Preserving Web Access Manager

Implements comprehensive privacy protection for web access including
anonymization, encryption, secure caching, and privacy-aware content filtering.
"""

import asyncio
import aiohttp
import hashlib
import json
import logging
import random
import time
from typing import Dict, List, Optional, Any, Tuple, Set, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import urllib.parse
from collections import defaultdict, deque

from web.connectivity.web_monitor import WebContent

logger = logging.getLogger(__name__)


@dataclass
class PrivacyConfig:
    """Configuration for privacy-preserving web access"""
    # Anonymization settings
    enable_anonymization: bool = True
    rotate_user_agents: bool = True
    randomize_request_timing: bool = True
    min_request_delay: float = 1.0
    max_request_delay: float = 5.0
    
    # Proxy and VPN settings
    proxy_servers: List[str] = field(default_factory=list)
    tor_proxy: Optional[str] = None  # e.g., "socks5://127.0.0.1:9050"
    rotate_proxies: bool = True
    proxy_rotation_interval: int = 10  # requests
    
    # Encryption settings
    encrypt_stored_data: bool = True
    encrypt_cache: bool = True
    encryption_key_rotation: bool = True
    key_rotation_interval: timedelta = field(default_factory=lambda: timedelta(hours=24))
    
    # Privacy filtering
    remove_tracking_parameters: bool = True
    block_analytics: bool = True
    strip_referrer: bool = True
    block_fingerprinting: bool = True
    
    # Data retention
    max_log_retention: timedelta = field(default_factory=lambda: timedelta(hours=1))
    clear_cache_interval: timedelta = field(default_factory=lambda: timedelta(hours=6))
    anonymize_stored_urls: bool = True
    
    # Request headers
    custom_headers: Dict[str, str] = field(default_factory=dict)
    remove_identifying_headers: bool = True
    
    # Content filtering
    filter_personal_info: bool = True
    redact_sensitive_data: bool = True
    content_sanitization: bool = True


@dataclass
class PrivacyMetrics:
    """Privacy protection metrics"""
    requests_anonymized: int = 0
    proxies_used: Set[str] = field(default_factory=set)
    user_agents_rotated: int = 0
    tracking_params_removed: int = 0
    content_sanitized: int = 0
    cache_encryptions: int = 0
    key_rotations: int = 0
    privacy_violations_detected: int = 0
    last_updated: datetime = field(default_factory=datetime.now)


class EncryptionManager:
    """Manages encryption for privacy protection"""
    
    def __init__(self, config: PrivacyConfig):
        self.config = config
        self.current_key: Optional[Fernet] = None
        self.key_history: List[Tuple[datetime, Fernet]] = []
        self.last_rotation = datetime.now()
        
        # Initialize encryption
        if config.encrypt_stored_data or config.encrypt_cache:
            self._generate_new_key()
    
    def _generate_new_key(self):
        """Generate new encryption key"""
        
        # Use a combination of timestamp and random data for key generation
        key_material = f"{time.time()}{random.random()}".encode()
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b'shvayambhu_privacy_salt',
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(key_material))
        
        # Store old key in history before rotating
        if self.current_key is not None:
            self.key_history.append((self.last_rotation, self.current_key))
            
            # Keep only recent keys for decryption compatibility
            cutoff = datetime.now() - timedelta(days=7)
            self.key_history = [
                (ts, k) for ts, k in self.key_history if ts > cutoff
            ]
        
        self.current_key = Fernet(key)
        self.last_rotation = datetime.now()
        
        logger.info("Generated new encryption key")
    
    def encrypt(self, data: Union[str, bytes]) -> bytes:
        """Encrypt data"""
        
        if not self.current_key:
            raise ValueError("Encryption not initialized")
        
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        return self.current_key.encrypt(data)
    
    def decrypt(self, encrypted_data: bytes) -> bytes:
        """Decrypt data, trying current and historical keys"""
        
        # Try current key first
        if self.current_key:
            try:
                return self.current_key.decrypt(encrypted_data)
            except Exception:
                pass
        
        # Try historical keys
        for _, historical_key in reversed(self.key_history):
            try:
                return historical_key.decrypt(encrypted_data)
            except Exception:
                continue
        
        raise ValueError("Unable to decrypt data with any available key")
    
    def should_rotate_key(self) -> bool:
        """Check if key should be rotated"""
        
        if not self.config.encryption_key_rotation:
            return False
        
        return datetime.now() - self.last_rotation > self.config.key_rotation_interval
    
    def rotate_key_if_needed(self):
        """Rotate key if rotation interval has passed"""
        
        if self.should_rotate_key():
            self._generate_new_key()
            return True
        return False


class UserAgentRotator:
    """Rotates user agents for anonymization"""
    
    def __init__(self):
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:89.0) Gecko/20100101 Firefox/89.0',
            'Mozilla/5.0 (X11; Linux x86_64; rv:89.0) Gecko/20100101 Firefox/89.0',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/91.0.864.59'
        ]
        self.current_index = 0
    
    def get_random_user_agent(self) -> str:
        """Get a random user agent"""
        return random.choice(self.user_agents)
    
    def get_next_user_agent(self) -> str:
        """Get next user agent in rotation"""
        user_agent = self.user_agents[self.current_index]
        self.current_index = (self.current_index + 1) % len(self.user_agents)
        return user_agent


class ContentSanitizer:
    """Sanitizes content to remove privacy-sensitive information"""
    
    def __init__(self, config: PrivacyConfig):
        self.config = config
        
        # Patterns for sensitive data detection
        self.sensitive_patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'\b\d{3}-\d{3}-\d{4}\b|\b\(\d{3}\)\s*\d{3}-\d{4}\b',
            'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
            'credit_card': r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b',
            'ip_address': r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'
        }
        
        # Tracking and analytics domains
        self.tracking_domains = {
            'google-analytics.com', 'googletagmanager.com', 'doubleclick.net',
            'facebook.com', 'connect.facebook.net', 'analytics.twitter.com',
            'scorecardresearch.com', 'quantserve.com', 'outbrain.com'
        }
    
    def sanitize_content(self, content: str, url: str) -> Tuple[str, int]:
        """Sanitize content to remove sensitive information"""
        
        if not self.config.content_sanitization:
            return content, 0
        
        sanitized = content
        replacements = 0
        
        if self.config.redact_sensitive_data:
            # Replace sensitive patterns
            import re
            for pattern_name, pattern in self.sensitive_patterns.items():
                matches = re.findall(pattern, sanitized)
                if matches:
                    sanitized = re.sub(pattern, f'[REDACTED_{pattern_name.upper()}]', sanitized)
                    replacements += len(matches)
        
        if self.config.filter_personal_info:
            # Remove potential personal information markers
            personal_keywords = [
                'my name is', 'i am', 'personal', 'private', 'confidential'
            ]
            
            for keyword in personal_keywords:
                if keyword.lower() in sanitized.lower():
                    # Replace the sentence containing the keyword
                    sentences = sanitized.split('.')
                    sanitized_sentences = []
                    
                    for sentence in sentences:
                        if keyword.lower() not in sentence.lower():
                            sanitized_sentences.append(sentence)
                        else:
                            sanitized_sentences.append('[PERSONAL_INFO_REDACTED]')
                            replacements += 1
                    
                    sanitized = '.'.join(sanitized_sentences)
        
        return sanitized, replacements
    
    def remove_tracking_parameters(self, url: str) -> str:
        """Remove tracking parameters from URL"""
        
        if not self.config.remove_tracking_parameters:
            return url
        
        # Common tracking parameters
        tracking_params = {
            'utm_source', 'utm_medium', 'utm_campaign', 'utm_term', 'utm_content',
            'gclid', 'fbclid', 'msclkid', 'dclid', '_ga', 'mc_cid', 'mc_eid'
        }
        
        parsed = urllib.parse.urlparse(url)
        query_params = urllib.parse.parse_qsl(parsed.query)
        
        # Filter out tracking parameters
        clean_params = [
            (key, value) for key, value in query_params
            if key not in tracking_params
        ]
        
        # Rebuild URL
        clean_query = urllib.parse.urlencode(clean_params)
        clean_url = urllib.parse.urlunparse(
            (parsed.scheme, parsed.netloc, parsed.path, parsed.params, clean_query, parsed.fragment)
        )
        
        return clean_url


class ProxyManager:
    """Manages proxy rotation for anonymization"""
    
    def __init__(self, config: PrivacyConfig):
        self.config = config
        self.proxy_list = config.proxy_servers.copy()
        self.current_proxy_index = 0
        self.request_count = 0
        self.proxy_performance: Dict[str, Dict[str, Any]] = {}
        
        # Add Tor proxy if configured
        if config.tor_proxy:
            self.proxy_list.append(config.tor_proxy)
    
    def get_current_proxy(self) -> Optional[str]:
        """Get current proxy"""
        
        if not self.proxy_list or not self.config.rotate_proxies:
            return None
        
        return self.proxy_list[self.current_proxy_index] if self.proxy_list else None
    
    def rotate_proxy(self) -> Optional[str]:
        """Rotate to next proxy"""
        
        if not self.proxy_list:
            return None
        
        self.current_proxy_index = (self.current_proxy_index + 1) % len(self.proxy_list)
        return self.proxy_list[self.current_proxy_index]
    
    def should_rotate(self) -> bool:
        """Check if proxy should be rotated"""
        
        return (self.config.rotate_proxies and 
                self.request_count % self.config.proxy_rotation_interval == 0)
    
    def record_request(self, proxy: Optional[str], success: bool, response_time: float):
        """Record proxy performance"""
        
        if proxy is None:
            proxy = 'direct'
        
        if proxy not in self.proxy_performance:
            self.proxy_performance[proxy] = {
                'requests': 0,
                'successes': 0,
                'total_response_time': 0.0,
                'last_used': datetime.now()
            }
        
        stats = self.proxy_performance[proxy]
        stats['requests'] += 1
        stats['total_response_time'] += response_time
        stats['last_used'] = datetime.now()
        
        if success:
            stats['successes'] += 1
        
        self.request_count += 1
    
    def get_best_proxies(self, top_n: int = 3) -> List[str]:
        """Get best performing proxies"""
        
        proxy_scores = []
        
        for proxy, stats in self.proxy_performance.items():
            if stats['requests'] > 0:
                success_rate = stats['successes'] / stats['requests']
                avg_response_time = stats['total_response_time'] / stats['requests']
                
                # Score based on success rate and response time
                score = success_rate / (1 + avg_response_time)
                proxy_scores.append((proxy, score))
        
        proxy_scores.sort(key=lambda x: x[1], reverse=True)
        return [proxy for proxy, _ in proxy_scores[:top_n]]


class PrivacyAwareSession:
    """Privacy-aware HTTP session"""
    
    def __init__(self, config: PrivacyConfig):
        self.config = config
        self.user_agent_rotator = UserAgentRotator()
        self.proxy_manager = ProxyManager(config)
        self.content_sanitizer = ContentSanitizer(config)
        
        self.session: Optional[aiohttp.ClientSession] = None
        self.request_times: deque = deque(maxlen=100)
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.start_session()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close_session()
    
    async def start_session(self):
        """Start HTTP session with privacy settings"""
        
        # Configure headers
        headers = {
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
        
        # Add custom headers
        headers.update(self.config.custom_headers)
        
        # Remove identifying headers if configured
        if self.config.remove_identifying_headers:
            headers.pop('User-Agent', None)  # Will be set per request
        
        # Configure proxy
        proxy = self.proxy_manager.get_current_proxy()
        
        # Configure connector
        connector = aiohttp.TCPConnector(
            limit=10,
            limit_per_host=5,
            ttl_dns_cache=300,
            use_dns_cache=True
        )
        
        self.session = aiohttp.ClientSession(
            headers=headers,
            connector=connector,
            timeout=aiohttp.ClientTimeout(total=30)
        )
    
    async def close_session(self):
        """Close HTTP session"""
        
        if self.session:
            await self.session.close()
            self.session = None
    
    async def private_request(
        self,
        method: str,
        url: str,
        **kwargs
    ) -> aiohttp.ClientResponse:
        """Make privacy-aware HTTP request"""
        
        if not self.session:
            raise ValueError("Session not started")
        
        start_time = time.time()
        
        # Clean URL
        clean_url = self.content_sanitizer.remove_tracking_parameters(url)
        
        # Configure request headers
        request_headers = kwargs.get('headers', {})
        
        if self.config.rotate_user_agents:
            request_headers['User-Agent'] = self.user_agent_rotator.get_next_user_agent()
        
        if self.config.strip_referrer:
            request_headers.pop('Referer', None)
            request_headers.pop('Referrer', None)
        
        kwargs['headers'] = request_headers
        
        # Configure proxy
        proxy = self.proxy_manager.get_current_proxy()
        if proxy:
            kwargs['proxy'] = proxy
        
        # Add request delay for anonymization
        if self.config.randomize_request_timing:
            delay = random.uniform(
                self.config.min_request_delay,
                self.config.max_request_delay
            )
            await asyncio.sleep(delay)
        
        try:
            # Make request
            response = await self.session.request(method, clean_url, **kwargs)
            
            # Record performance
            response_time = time.time() - start_time
            self.proxy_manager.record_request(proxy, True, response_time)
            self.request_times.append(response_time)
            
            # Rotate proxy if needed
            if self.proxy_manager.should_rotate():
                self.proxy_manager.rotate_proxy()
            
            return response
            
        except Exception as e:
            # Record failure
            response_time = time.time() - start_time
            self.proxy_manager.record_request(proxy, False, response_time)
            
            logger.error(f"Private request failed: {e}")
            raise


class PrivacyPreservingWebManager:
    """Main privacy-preserving web access manager"""
    
    def __init__(self, config: PrivacyConfig):
        self.config = config
        self.encryption_manager = EncryptionManager(config)
        self.content_sanitizer = ContentSanitizer(config)
        self.metrics = PrivacyMetrics()
        
        # Secure cache
        self.secure_cache: Dict[str, Tuple[bytes, datetime]] = {}
        self.cache_access_log: List[Tuple[str, datetime]] = []
        
        # Privacy logs (temporary, limited retention)
        self.privacy_logs: deque = deque(maxlen=1000)
        
        # Setup cache cleanup
        self.last_cache_cleanup = datetime.now()
        self.last_log_cleanup = datetime.now()
    
    async def fetch_content_privately(
        self,
        url: str,
        method: str = 'GET',
        **kwargs
    ) -> Tuple[WebContent, PrivacyMetrics]:
        """Fetch web content with privacy protection"""
        
        # Check secure cache first
        cache_key = self._get_cache_key(url)
        cached_content = self._get_from_secure_cache(cache_key)
        
        if cached_content:
            logger.debug(f"Retrieved from secure cache: {url}")
            return cached_content, self.metrics
        
        # Make private request
        async with PrivacyAwareSession(self.config) as session:
            try:
                response = await session.private_request(method, url, **kwargs)
                content_text = await response.text()
                
                # Sanitize content
                sanitized_content, sanitization_count = self.content_sanitizer.sanitize_content(
                    content_text, url
                )
                
                # Create WebContent object
                web_content = WebContent(
                    id=hashlib.md5(f"{url}{time.time()}".encode()).hexdigest(),
                    url=self._anonymize_url(url) if self.config.anonymize_stored_urls else url,
                    title=self._extract_title(sanitized_content),
                    content=sanitized_content,
                    source=self._extract_domain(url),
                    published_at=datetime.now(),
                    retrieved_at=datetime.now(),
                    content_type='unknown',
                    relevance_score=1.0,
                    keywords=[],
                    summary='',
                    language='en',
                    metadata={'privacy_protected': True, 'sanitized': sanitization_count > 0}
                )
                
                # Store in secure cache
                self._store_in_secure_cache(cache_key, web_content)
                
                # Update metrics
                self.metrics.requests_anonymized += 1
                if session.proxy_manager.get_current_proxy():
                    self.metrics.proxies_used.add(session.proxy_manager.get_current_proxy())
                self.metrics.user_agents_rotated += 1
                self.metrics.content_sanitized += sanitization_count
                
                # Log request (will be cleaned up automatically)
                self._log_privacy_request(url, sanitization_count > 0)
                
                # Cleanup if needed
                await self._cleanup_if_needed()
                
                return web_content, self.metrics
                
            except Exception as e:
                logger.error(f"Failed to fetch content privately: {e}")
                self.metrics.privacy_violations_detected += 1
                raise
    
    def _get_cache_key(self, url: str) -> str:
        """Generate cache key for URL"""
        
        # Use hash of URL for privacy
        return hashlib.sha256(url.encode()).hexdigest()
    
    def _get_from_secure_cache(self, cache_key: str) -> Optional[WebContent]:
        """Retrieve content from secure cache"""
        
        if not self.config.encrypt_cache:
            return None
        
        if cache_key in self.secure_cache:
            encrypted_data, cache_time = self.secure_cache[cache_key]
            
            # Check if cache is still valid
            if datetime.now() - cache_time < self.config.clear_cache_interval:
                try:
                    # Decrypt and deserialize
                    decrypted_data = self.encryption_manager.decrypt(encrypted_data)
                    content_data = json.loads(decrypted_data.decode('utf-8'))
                    
                    # Reconstruct WebContent
                    web_content = WebContent(**content_data)
                    
                    # Log cache access
                    self.cache_access_log.append((cache_key, datetime.now()))
                    
                    return web_content
                    
                except Exception as e:
                    logger.error(f"Failed to decrypt cached content: {e}")
                    # Remove corrupted cache entry
                    del self.secure_cache[cache_key]
        
        return None
    
    def _store_in_secure_cache(self, cache_key: str, content: WebContent):
        """Store content in secure cache"""
        
        if not self.config.encrypt_cache:
            return
        
        try:
            # Serialize content
            content_dict = {
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
            
            content_json = json.dumps(content_dict)
            
            # Encrypt
            encrypted_data = self.encryption_manager.encrypt(content_json)
            
            # Store with timestamp
            self.secure_cache[cache_key] = (encrypted_data, datetime.now())
            self.metrics.cache_encryptions += 1
            
        except Exception as e:
            logger.error(f"Failed to store content in secure cache: {e}")
    
    def _anonymize_url(self, url: str) -> str:
        """Anonymize URL for storage"""
        
        # Replace domain with hash
        parsed = urllib.parse.urlparse(url)
        domain_hash = hashlib.md5(parsed.netloc.encode()).hexdigest()[:16]
        
        anonymized_url = urllib.parse.urlunparse((
            'https',  # Always use https for anonymized URLs
            f'domain_{domain_hash}.anon',
            parsed.path,
            parsed.params,
            parsed.query,
            parsed.fragment
        ))
        
        return anonymized_url
    
    def _extract_title(self, content: str) -> str:
        """Extract title from content"""
        
        import re
        title_match = re.search(r'<title>(.*?)</title>', content, re.IGNORECASE)
        return title_match.group(1) if title_match else 'No title'
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL"""
        
        parsed = urllib.parse.urlparse(url)
        return parsed.netloc
    
    def _log_privacy_request(self, url: str, was_sanitized: bool):
        """Log privacy request (with limited retention)"""
        
        # Only store minimal, non-identifying information
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'domain': self._extract_domain(url),
            'was_sanitized': was_sanitized,
            'url_hash': hashlib.md5(url.encode()).hexdigest()[:16]
        }
        
        self.privacy_logs.append(log_entry)
    
    async def _cleanup_if_needed(self):
        """Perform cleanup operations if needed"""
        
        now = datetime.now()
        
        # Cache cleanup
        if now - self.last_cache_cleanup > self.config.clear_cache_interval:
            await self._cleanup_cache()
            self.last_cache_cleanup = now
        
        # Log cleanup
        if now - self.last_log_cleanup > self.config.max_log_retention:
            self._cleanup_logs()
            self.last_log_cleanup = now
        
        # Key rotation
        if self.encryption_manager.rotate_key_if_needed():
            self.metrics.key_rotations += 1
    
    async def _cleanup_cache(self):
        """Clean up expired cache entries"""
        
        cutoff_time = datetime.now() - self.config.clear_cache_interval
        expired_keys = [
            key for key, (_, cache_time) in self.secure_cache.items()
            if cache_time < cutoff_time
        ]
        
        for key in expired_keys:
            del self.secure_cache[key]
        
        # Clean up cache access log
        self.cache_access_log = [
            (key, access_time) for key, access_time in self.cache_access_log
            if access_time > cutoff_time
        ]
        
        logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")
    
    def _cleanup_logs(self):
        """Clean up old privacy logs"""
        
        cutoff_time = datetime.now() - self.config.max_log_retention
        
        # Privacy logs are automatically limited by deque maxlen
        # Just clear if we want immediate cleanup
        if self.config.max_log_retention.total_seconds() < 3600:  # Less than 1 hour
            self.privacy_logs.clear()
            logger.info("Cleared privacy logs")
    
    def get_privacy_statistics(self) -> Dict[str, Any]:
        """Get comprehensive privacy statistics"""
        
        return {
            'requests_anonymized': self.metrics.requests_anonymized,
            'unique_proxies_used': len(self.metrics.proxies_used),
            'user_agents_rotated': self.metrics.user_agents_rotated,
            'tracking_params_removed': self.metrics.tracking_params_removed,
            'content_items_sanitized': self.metrics.content_sanitized,
            'cache_encryptions': self.metrics.cache_encryptions,
            'key_rotations': self.metrics.key_rotations,
            'privacy_violations_detected': self.metrics.privacy_violations_detected,
            'secure_cache_size': len(self.secure_cache),
            'privacy_logs_count': len(self.privacy_logs),
            'encryption_enabled': self.config.encrypt_stored_data,
            'anonymization_enabled': self.config.enable_anonymization,
            'last_metrics_update': self.metrics.last_updated.isoformat()
        }
    
    def export_privacy_report(self) -> Dict[str, Any]:
        """Export comprehensive privacy protection report"""
        
        stats = self.get_privacy_statistics()
        
        # Aggregate privacy logs (without exposing sensitive data)
        domain_stats = defaultdict(int)
        sanitization_rate = 0
        
        for log_entry in self.privacy_logs:
            if isinstance(log_entry, dict):
                domain_stats[log_entry.get('domain', 'unknown')] += 1
                if log_entry.get('was_sanitized', False):
                    sanitization_rate += 1
        
        if len(self.privacy_logs) > 0:
            sanitization_rate = sanitization_rate / len(self.privacy_logs)
        
        report = {
            'report_generated': datetime.now().isoformat(),
            'privacy_configuration': {
                'anonymization_enabled': self.config.enable_anonymization,
                'proxy_rotation': self.config.rotate_proxies,
                'encryption_enabled': self.config.encrypt_stored_data,
                'content_sanitization': self.config.content_sanitization,
                'tracking_removal': self.config.remove_tracking_parameters
            },
            'statistics': stats,
            'privacy_effectiveness': {
                'content_sanitization_rate': sanitization_rate,
                'cache_encryption_rate': 1.0 if self.config.encrypt_cache else 0.0,
                'anonymization_coverage': 1.0 if self.config.enable_anonymization else 0.0
            },
            'domain_access_summary': dict(domain_stats),
            'recommendations': self._generate_privacy_recommendations()
        }
        
        return report
    
    def _generate_privacy_recommendations(self) -> List[str]:
        """Generate privacy improvement recommendations"""
        
        recommendations = []
        
        if not self.config.enable_anonymization:
            recommendations.append("Enable anonymization for better privacy protection")
        
        if not self.config.encrypt_stored_data:
            recommendations.append("Enable data encryption for stored content")
        
        if not self.config.rotate_proxies and self.config.proxy_servers:
            recommendations.append("Enable proxy rotation for improved anonymization")
        
        if not self.config.content_sanitization:
            recommendations.append("Enable content sanitization to remove sensitive data")
        
        if self.metrics.privacy_violations_detected > 0:
            recommendations.append("Review privacy violations and strengthen protection measures")
        
        if len(self.config.proxy_servers) < 3:
            recommendations.append("Add more proxy servers for better anonymization")
        
        return recommendations


# Testing and example usage
async def test_privacy_manager():
    """Test privacy-preserving web access"""
    
    # Create configuration
    config = PrivacyConfig(
        enable_anonymization=True,
        encrypt_stored_data=True,
        encrypt_cache=True,
        content_sanitization=True,
        remove_tracking_parameters=True
    )
    
    # Create privacy manager
    manager = PrivacyPreservingWebManager(config)
    
    # Test URLs
    test_urls = [
        "https://example.com/article?utm_source=test&utm_medium=email",
        "https://news.example.com/privacy-test"
    ]
    
    print("Testing privacy-preserving web access...")
    
    for url in test_urls:
        try:
            content, metrics = await manager.fetch_content_privately(url)
            
            print(f"\nFetched: {url}")
            print(f"Retrieved URL: {content.url}")
            print(f"Title: {content.title}")
            print(f"Content length: {len(content.content)} chars")
            print(f"Privacy protected: {content.metadata.get('privacy_protected', False)}")
            print(f"Content sanitized: {content.metadata.get('sanitized', False)}")
            
        except Exception as e:
            print(f"Failed to fetch {url}: {e}")
    
    # Print privacy statistics
    stats = manager.get_privacy_statistics()
    print("\nPrivacy Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Export privacy report
    report = manager.export_privacy_report()
    print("\nPrivacy Report:")
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    asyncio.run(test_privacy_manager())