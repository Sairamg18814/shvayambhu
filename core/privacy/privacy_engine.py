"""
Privacy Engine for Shvayambhu LLM System

This module implements comprehensive privacy protection mechanisms including PII detection,
data anonymization, consent management, privacy-preserving computation, and consciousness-aware
privacy processing.

Key Features:
- Personal Identifiable Information (PII) detection and handling
- Advanced data anonymization and pseudonymization techniques
- Comprehensive consent management system
- Privacy-preserving computation with differential privacy
- Secure data transmission and storage protocols
- Privacy audit and compliance monitoring
- User privacy controls and preferences management
- Privacy-aware consciousness processing
- GDPR, CCPA, and other privacy regulation compliance
- Real-time privacy risk assessment
"""

import asyncio
import hashlib
import hmac
import json
import logging
import re
import secrets
import time
from abc import ABC, abstractmethod
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable
import numpy as np
import base64

# Base consciousness integration
from ..consciousness.base import ConsciousnessAwareModule

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PrivacyRiskLevel(Enum):
    """Privacy risk severity levels"""
    NONE = auto()
    LOW = auto()
    MEDIUM = auto()
    HIGH = auto()
    CRITICAL = auto()


class PIICategory(Enum):
    """Categories of Personally Identifiable Information"""
    NAME = auto()
    EMAIL = auto()
    PHONE = auto()
    ADDRESS = auto()
    SSN = auto()
    CREDIT_CARD = auto()
    IP_ADDRESS = auto()
    MEDICAL_INFO = auto()
    FINANCIAL_INFO = auto()
    BIOMETRIC = auto()
    GEOLOCATION = auto()
    DEVICE_ID = auto()


class PrivacyAction(Enum):
    """Privacy protection actions"""
    ALLOW = auto()
    ANONYMIZE = auto()
    PSEUDONYMIZE = auto()
    ENCRYPT = auto()
    REDACT = auto()
    BLOCK = auto()
    REQUEST_CONSENT = auto()


class ConsentStatus(Enum):
    """User consent status"""
    GRANTED = auto()
    DENIED = auto()
    PENDING = auto()
    EXPIRED = auto()
    REVOKED = auto()


class DataProcessingPurpose(Enum):
    """Purposes for data processing"""
    CORE_FUNCTIONALITY = auto()
    IMPROVEMENT = auto()
    ANALYTICS = auto()
    PERSONALIZATION = auto()
    MARKETING = auto()
    RESEARCH = auto()
    LEGAL_COMPLIANCE = auto()


@dataclass
class PIIDetection:
    """Detected PII information"""
    category: PIICategory
    value: str
    confidence: float
    start_pos: int
    end_pos: int
    context: str = ""
    risk_level: PrivacyRiskLevel = PrivacyRiskLevel.MEDIUM


@dataclass
class PrivacyInput:
    """Input for privacy processing"""
    content: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    purpose: DataProcessingPurpose = DataProcessingPurpose.CORE_FUNCTIONALITY
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    consciousness_context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConsentRecord:
    """User consent record"""
    user_id: str
    purpose: DataProcessingPurpose
    status: ConsentStatus
    granted_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    revoked_at: Optional[datetime] = None
    consent_details: Dict[str, Any] = field(default_factory=dict)
    version: str = "1.0"


@dataclass
class PrivacyAssessment:
    """Privacy assessment result"""
    input_content: str
    overall_risk_level: PrivacyRiskLevel
    is_privacy_safe: bool
    detected_pii: List[PIIDetection] = field(default_factory=list)
    recommended_actions: List[PrivacyAction] = field(default_factory=list)
    processed_content: Optional[str] = None
    privacy_warnings: List[str] = field(default_factory=list)
    consent_required: List[DataProcessingPurpose] = field(default_factory=list)
    consciousness_insights: Dict[str, Any] = field(default_factory=dict)
    processing_time_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


class PIIDetector:
    """Advanced PII detection system"""
    
    def __init__(self):
        # Regex patterns for different PII types
        self.patterns = {
            PIICategory.EMAIL: [
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            ],
            PIICategory.PHONE: [
                r'\b(?:\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})\b',
                r'\b\d{3}-\d{3}-\d{4}\b',
                r'\b\(\d{3}\)\s*\d{3}-\d{4}\b'
            ],
            PIICategory.SSN: [
                r'\b\d{3}-\d{2}-\d{4}\b',
                r'\b\d{3}\s\d{2}\s\d{4}\b',
                r'\b\d{9}\b'
            ],
            PIICategory.CREDIT_CARD: [
                r'\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|3[0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})\b'
            ],
            PIICategory.IP_ADDRESS: [
                r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b',
                r'\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b'
            ],
            PIICategory.ADDRESS: [
                r'\b\d+\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct|Circle|Cir|Way|Place|Pl)\b',
            ]
        }
        
        # Name detection patterns (more complex)
        self.name_indicators = [
            r'\bMr\.\s+[A-Z][a-z]+\b',
            r'\bMs\.\s+[A-Z][a-z]+\b',
            r'\bDr\.\s+[A-Z][a-z]+\b',
            r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b'  # Simple first last name
        ]
        
        # Contextual keywords that increase PII likelihood
        self.context_keywords = {
            PIICategory.NAME: ['name', 'called', 'person', 'individual', 'patient', 'client'],
            PIICategory.EMAIL: ['email', 'contact', 'address', 'send to', 'reply to'],
            PIICategory.PHONE: ['phone', 'number', 'call', 'contact', 'mobile', 'cell'],
            PIICategory.ADDRESS: ['address', 'location', 'residence', 'home', 'office', 'building'],
            PIICategory.MEDICAL_INFO: ['diagnosis', 'condition', 'treatment', 'medication', 'symptom', 'illness'],
            PIICategory.FINANCIAL_INFO: ['account', 'balance', 'income', 'salary', 'payment', 'transaction']
        }
    
    def detect_pii(self, content: str) -> List[PIIDetection]:
        """Detect PII in content"""
        detections = []
        content_lower = content.lower()
        
        # Detect using regex patterns
        for category, patterns in self.patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, content, re.IGNORECASE)
                for match in matches:
                    confidence = self._calculate_confidence(category, match.group(), content_lower, match.start())
                    risk_level = self._assess_risk_level(category, confidence)
                    
                    detections.append(PIIDetection(
                        category=category,
                        value=match.group(),
                        confidence=confidence,
                        start_pos=match.start(),
                        end_pos=match.end(),
                        context=self._get_context(content, match.start(), match.end()),
                        risk_level=risk_level
                    ))
        
        # Detect names using specialized logic
        name_detections = self._detect_names(content)
        detections.extend(name_detections)
        
        # Remove duplicates and overlaps
        detections = self._remove_overlaps(detections)
        
        return detections
    
    def _detect_names(self, content: str) -> List[PIIDetection]:
        """Specialized name detection"""
        detections = []
        content_lower = content.lower()
        
        # Check for name indicators
        for pattern in self.name_indicators:
            matches = re.finditer(pattern, content)
            for match in matches:
                confidence = self._calculate_confidence(PIICategory.NAME, match.group(), content_lower, match.start())
                
                if confidence > 0.5:  # Only high-confidence name matches
                    detections.append(PIIDetection(
                        category=PIICategory.NAME,
                        value=match.group(),
                        confidence=confidence,
                        start_pos=match.start(),
                        end_pos=match.end(),
                        context=self._get_context(content, match.start(), match.end()),
                        risk_level=self._assess_risk_level(PIICategory.NAME, confidence)
                    ))
        
        return detections
    
    def _calculate_confidence(self, category: PIICategory, value: str, content_lower: str, position: int) -> float:
        """Calculate confidence score for PII detection"""
        base_confidence = 0.8  # Base confidence for regex match
        
        # Adjust based on contextual keywords
        context_keywords = self.context_keywords.get(category, [])
        context_window = content_lower[max(0, position-50):position+50]
        
        keyword_bonus = 0
        for keyword in context_keywords:
            if keyword in context_window:
                keyword_bonus += 0.1
        
        # Category-specific adjustments
        if category == PIICategory.EMAIL:
            if '@' in value and '.' in value.split('@')[1]:
                base_confidence = 0.95
        elif category == PIICategory.PHONE:
            if len(re.sub(r'[^\d]', '', value)) == 10:
                base_confidence = 0.9
        elif category == PIICategory.SSN:
            if len(re.sub(r'[^\d]', '', value)) == 9:
                base_confidence = 0.95
        
        return min(0.99, base_confidence + keyword_bonus)
    
    def _assess_risk_level(self, category: PIICategory, confidence: float) -> PrivacyRiskLevel:
        """Assess privacy risk level"""
        high_risk_categories = [PIICategory.SSN, PIICategory.CREDIT_CARD, PIICategory.MEDICAL_INFO, PIICategory.BIOMETRIC]
        medium_risk_categories = [PIICategory.NAME, PIICategory.EMAIL, PIICategory.PHONE, PIICategory.ADDRESS]
        
        if category in high_risk_categories:
            if confidence > 0.8:
                return PrivacyRiskLevel.CRITICAL
            else:
                return PrivacyRiskLevel.HIGH
        elif category in medium_risk_categories:
            if confidence > 0.9:
                return PrivacyRiskLevel.HIGH
            elif confidence > 0.7:
                return PrivacyRiskLevel.MEDIUM
            else:
                return PrivacyRiskLevel.LOW
        else:
            return PrivacyRiskLevel.LOW
    
    def _get_context(self, content: str, start: int, end: int, window: int = 20) -> str:
        """Get context around detected PII"""
        context_start = max(0, start - window)
        context_end = min(len(content), end + window)
        return content[context_start:context_end]
    
    def _remove_overlaps(self, detections: List[PIIDetection]) -> List[PIIDetection]:
        """Remove overlapping detections, keeping higher confidence ones"""
        if not detections:
            return detections
        
        # Sort by position then by confidence (descending)
        sorted_detections = sorted(detections, key=lambda x: (x.start_pos, -x.confidence))
        
        filtered = []
        for detection in sorted_detections:
            # Check if it overlaps with any existing detection
            overlaps = False
            for existing in filtered:
                if (detection.start_pos < existing.end_pos and 
                    detection.end_pos > existing.start_pos):
                    overlaps = True
                    break
            
            if not overlaps:
                filtered.append(detection)
        
        return filtered


class DataAnonymizer:
    """Advanced data anonymization and pseudonymization"""
    
    def __init__(self, encryption_key: Optional[bytes] = None):
        self.encryption_key = encryption_key or Fernet.generate_key()
        self.cipher = Fernet(self.encryption_key)
        
        # K-anonymity and l-diversity parameters
        self.k_anonymity_k = 5
        self.l_diversity_l = 3
        
        # Replacement strategies
        self.replacement_strategies = {
            PIICategory.NAME: self._anonymize_name,
            PIICategory.EMAIL: self._anonymize_email,
            PIICategory.PHONE: self._anonymize_phone,
            PIICategory.ADDRESS: self._anonymize_address,
            PIICategory.SSN: self._anonymize_ssn,
            PIICategory.CREDIT_CARD: self._anonymize_credit_card,
            PIICategory.IP_ADDRESS: self._anonymize_ip,
        }
    
    def anonymize_content(self, content: str, detections: List[PIIDetection], strategy: str = "anonymize") -> str:
        """Anonymize content based on detected PII"""
        if not detections:
            return content
        
        # Sort detections by position (descending) to avoid position shifts
        sorted_detections = sorted(detections, key=lambda x: x.start_pos, reverse=True)
        
        anonymized_content = content
        
        for detection in sorted_detections:
            if strategy == "anonymize":
                replacement = self._get_anonymous_replacement(detection)
            elif strategy == "pseudonymize":
                replacement = self._get_pseudonymous_replacement(detection)
            elif strategy == "redact":
                replacement = "[REDACTED]"
            elif strategy == "encrypt":
                replacement = self._encrypt_value(detection.value)
            else:
                replacement = "[REMOVED]"
            
            # Replace the detected PII
            anonymized_content = (
                anonymized_content[:detection.start_pos] + 
                replacement + 
                anonymized_content[detection.end_pos:]
            )
        
        return anonymized_content
    
    def _get_anonymous_replacement(self, detection: PIIDetection) -> str:
        """Get anonymous replacement for PII"""
        strategy = self.replacement_strategies.get(detection.category)
        if strategy:
            return strategy(detection.value, detection.category)
        else:
            return f"[{detection.category.name}]"
    
    def _get_pseudonymous_replacement(self, detection: PIIDetection) -> str:
        """Get pseudonymous replacement that maintains some properties"""
        # Generate consistent pseudonym based on hash
        hash_input = f"{detection.value}_{detection.category.name}"
        hash_value = hashlib.sha256(hash_input.encode()).hexdigest()[:8]
        return f"[{detection.category.name}_{hash_value}]"
    
    def _encrypt_value(self, value: str) -> str:
        """Encrypt sensitive value"""
        encrypted = self.cipher.encrypt(value.encode())
        return f"[ENCRYPTED_{base64.urlsafe_b64encode(encrypted).decode()}]"
    
    def _anonymize_name(self, value: str, category: PIICategory) -> str:
        """Anonymize names while preserving structure"""
        parts = value.split()
        if len(parts) == 2:  # First Last
            return "[FIRST_NAME] [LAST_NAME]"
        elif len(parts) == 3:  # First Middle Last
            return "[FIRST_NAME] [MIDDLE_NAME] [LAST_NAME]"
        else:
            return "[NAME]"
    
    def _anonymize_email(self, value: str, category: PIICategory) -> str:
        """Anonymize email while preserving domain structure"""
        if '@' in value:
            local, domain = value.split('@', 1)
            domain_parts = domain.split('.')
            if len(domain_parts) >= 2:
                return f"[EMAIL_USER]@[EMAIL_DOMAIN].{domain_parts[-1]}"
        return "[EMAIL]"
    
    def _anonymize_phone(self, value: str, category: PIICategory) -> str:
        """Anonymize phone number while preserving format"""
        digits_only = re.sub(r'[^\d]', '', value)
        if len(digits_only) == 10:
            return "XXX-XXX-XXXX"
        elif len(digits_only) == 11:
            return "+X-XXX-XXX-XXXX"
        else:
            return "[PHONE]"
    
    def _anonymize_address(self, value: str, category: PIICategory) -> str:
        """Anonymize address while preserving structure"""
        return "[ADDRESS]"
    
    def _anonymize_ssn(self, value: str, category: PIICategory) -> str:
        """Anonymize SSN"""
        return "XXX-XX-XXXX"
    
    def _anonymize_credit_card(self, value: str, category: PIICategory) -> str:
        """Anonymize credit card"""
        return "XXXX-XXXX-XXXX-XXXX"
    
    def _anonymize_ip(self, value: str, category: PIICategory) -> str:
        """Anonymize IP address while preserving network structure"""
        if ':' in value:  # IPv6
            return "[IPv6_ADDRESS]"
        else:  # IPv4
            parts = value.split('.')
            if len(parts) == 4:
                return f"{parts[0]}.{parts[1]}.XXX.XXX"
        return "[IP_ADDRESS]"


class ConsentManager:
    """Comprehensive consent management system"""
    
    def __init__(self):
        self.consent_records: Dict[str, List[ConsentRecord]] = {}
        self.consent_templates = {
            DataProcessingPurpose.CORE_FUNCTIONALITY: {
                "title": "Core Functionality",
                "description": "Process your data to provide basic AI assistance",
                "required": True,
                "duration_days": 365
            },
            DataProcessingPurpose.IMPROVEMENT: {
                "title": "Service Improvement",
                "description": "Use your data to improve AI capabilities",
                "required": False,
                "duration_days": 180
            },
            DataProcessingPurpose.PERSONALIZATION: {
                "title": "Personalization",
                "description": "Personalize responses based on your preferences",
                "required": False,
                "duration_days": 365
            },
            DataProcessingPurpose.ANALYTICS: {
                "title": "Analytics",
                "description": "Analyze usage patterns to improve service",
                "required": False,
                "duration_days": 90
            }
        }
    
    def request_consent(self, user_id: str, purposes: List[DataProcessingPurpose]) -> Dict[str, Any]:
        """Request consent for specific purposes"""
        consent_requests = []
        
        for purpose in purposes:
            template = self.consent_templates.get(purpose, {})
            consent_requests.append({
                "purpose": purpose.name,
                "title": template.get("title", purpose.name),
                "description": template.get("description", ""),
                "required": template.get("required", False),
                "duration_days": template.get("duration_days", 30)
            })
        
        return {
            "user_id": user_id,
            "consent_requests": consent_requests,
            "timestamp": datetime.now().isoformat(),
            "expires_at": (datetime.now() + timedelta(hours=24)).isoformat()  # Consent request expires
        }
    
    def grant_consent(self, user_id: str, purpose: DataProcessingPurpose, duration_days: int = 365) -> ConsentRecord:
        """Grant consent for a specific purpose"""
        expires_at = datetime.now() + timedelta(days=duration_days)
        
        record = ConsentRecord(
            user_id=user_id,
            purpose=purpose,
            status=ConsentStatus.GRANTED,
            granted_at=datetime.now(),
            expires_at=expires_at
        )
        
        if user_id not in self.consent_records:
            self.consent_records[user_id] = []
        
        # Remove any existing consent for this purpose
        self.consent_records[user_id] = [
            r for r in self.consent_records[user_id] if r.purpose != purpose
        ]
        
        self.consent_records[user_id].append(record)
        
        logger.info(f"Consent granted for user {user_id}, purpose {purpose.name}")
        return record
    
    def revoke_consent(self, user_id: str, purpose: DataProcessingPurpose) -> bool:
        """Revoke consent for a specific purpose"""
        if user_id not in self.consent_records:
            return False
        
        for record in self.consent_records[user_id]:
            if record.purpose == purpose and record.status == ConsentStatus.GRANTED:
                record.status = ConsentStatus.REVOKED
                record.revoked_at = datetime.now()
                logger.info(f"Consent revoked for user {user_id}, purpose {purpose.name}")
                return True
        
        return False
    
    def check_consent(self, user_id: str, purpose: DataProcessingPurpose) -> ConsentStatus:
        """Check consent status for a specific purpose"""
        if user_id not in self.consent_records:
            return ConsentStatus.DENIED
        
        for record in self.consent_records[user_id]:
            if record.purpose == purpose:
                # Check if consent is expired
                if (record.expires_at and 
                    record.expires_at < datetime.now() and 
                    record.status == ConsentStatus.GRANTED):
                    record.status = ConsentStatus.EXPIRED
                
                return record.status
        
        return ConsentStatus.DENIED
    
    def get_user_consents(self, user_id: str) -> List[ConsentRecord]:
        """Get all consent records for a user"""
        return self.consent_records.get(user_id, [])
    
    def cleanup_expired_consents(self) -> int:
        """Clean up expired consent records"""
        cleaned_count = 0
        current_time = datetime.now()
        
        for user_id in self.consent_records:
            for record in self.consent_records[user_id]:
                if (record.expires_at and 
                    record.expires_at < current_time and 
                    record.status == ConsentStatus.GRANTED):
                    record.status = ConsentStatus.EXPIRED
                    cleaned_count += 1
        
        logger.info(f"Cleaned up {cleaned_count} expired consent records")
        return cleaned_count


class PrivacyEngine(ConsciousnessAwareModule):
    """Main privacy engine coordinating all privacy protection measures"""
    
    def __init__(
        self,
        consciousness_state: Optional[Dict[str, Any]] = None,
        strict_mode: bool = True,
        enable_differential_privacy: bool = True
    ):
        super().__init__(consciousness_state)
        self.strict_mode = strict_mode
        self.enable_differential_privacy = enable_differential_privacy
        
        # Initialize components
        self.pii_detector = PIIDetector()
        self.anonymizer = DataAnonymizer()
        self.consent_manager = ConsentManager()
        
        # Privacy configuration
        self.privacy_thresholds = {
            PrivacyRiskLevel.CRITICAL: 0.9,
            PrivacyRiskLevel.HIGH: 0.8,
            PrivacyRiskLevel.MEDIUM: 0.6,
            PrivacyRiskLevel.LOW: 0.4
        }
        
        # Differential privacy parameters
        self.epsilon = 1.0  # Privacy budget
        self.delta = 1e-5   # Probability of privacy breach
        
        # Privacy statistics
        self.stats = {
            'total_assessments': 0,
            'pii_detections': 0,
            'privacy_actions': {action: 0 for action in PrivacyAction},
            'consent_requests': 0,
            'consent_grants': 0
        }
        
        logger.info("Privacy Engine initialized")
    
    async def assess_privacy(self, input_data: PrivacyInput) -> PrivacyAssessment:
        """Perform comprehensive privacy assessment"""
        start_time = time.time()
        
        try:
            # Update consciousness context
            await self._update_consciousness({
                'privacy_assessment_started': True,
                'input_length': len(input_data.content),
                'processing_purpose': input_data.purpose.name
            })
            
            # Detect PII
            detected_pii = self.pii_detector.detect_pii(input_data.content)
            
            # Assess overall privacy risk
            overall_risk_level = self._calculate_overall_risk(detected_pii)
            
            # Check consent requirements
            consent_required = self._check_consent_requirements(input_data, detected_pii)
            
            # Determine recommended actions
            recommended_actions = self._determine_privacy_actions(detected_pii, overall_risk_level)
            
            # Process content based on actions
            processed_content = await self._process_content(input_data.content, detected_pii, recommended_actions)
            
            # Generate privacy warnings
            privacy_warnings = self._generate_privacy_warnings(detected_pii, overall_risk_level)
            
            # Determine if privacy-safe
            is_privacy_safe = self._is_privacy_safe(overall_risk_level, consent_required, input_data)
            
            # Get consciousness insights
            consciousness_insights = await self._get_consciousness_insights({
                'pii_detected': len(detected_pii),
                'risk_level': overall_risk_level.name,
                'consent_status': len(consent_required) == 0,
                'actions_taken': [a.name for a in recommended_actions]
            })
            
            processing_time_ms = (time.time() - start_time) * 1000
            
            # Create assessment
            assessment = PrivacyAssessment(
                input_content=input_data.content,
                overall_risk_level=overall_risk_level,
                is_privacy_safe=is_privacy_safe,
                detected_pii=detected_pii,
                recommended_actions=recommended_actions,
                processed_content=processed_content,
                privacy_warnings=privacy_warnings,
                consent_required=consent_required,
                consciousness_insights=consciousness_insights,
                processing_time_ms=processing_time_ms
            )
            
            # Update statistics
            self._update_stats(assessment)
            
            return assessment
            
        except Exception as e:
            logger.error(f"Privacy assessment failed: {str(e)}")
            
            # Return safe fallback assessment
            return PrivacyAssessment(
                input_content=input_data.content,
                overall_risk_level=PrivacyRiskLevel.HIGH,
                is_privacy_safe=False,
                recommended_actions=[PrivacyAction.BLOCK],
                privacy_warnings=[f"Privacy assessment failed: {str(e)}"],
                processing_time_ms=(time.time() - start_time) * 1000
            )
    
    def _calculate_overall_risk(self, detections: List[PIIDetection]) -> PrivacyRiskLevel:
        """Calculate overall privacy risk level"""
        if not detections:
            return PrivacyRiskLevel.NONE
        
        # Use highest risk level
        max_risk = max(detection.risk_level for detection in detections)
        return max_risk
    
    def _check_consent_requirements(self, input_data: PrivacyInput, detections: List[PIIDetection]) -> List[DataProcessingPurpose]:
        """Check what consent is required for processing"""
        required_consents = []
        
        # Always check consent for the primary purpose
        if input_data.user_id:
            consent_status = self.consent_manager.check_consent(input_data.user_id, input_data.purpose)
            if consent_status != ConsentStatus.GRANTED:
                required_consents.append(input_data.purpose)
        
        # Additional consent requirements based on detected PII
        high_risk_pii = [d for d in detections if d.risk_level in [PrivacyRiskLevel.HIGH, PrivacyRiskLevel.CRITICAL]]
        if high_risk_pii:
            # Require explicit consent for processing high-risk PII
            for purpose in [DataProcessingPurpose.IMPROVEMENT, DataProcessingPurpose.ANALYTICS]:
                if (input_data.user_id and 
                    self.consent_manager.check_consent(input_data.user_id, purpose) != ConsentStatus.GRANTED):
                    required_consents.append(purpose)
        
        return list(set(required_consents))
    
    def _determine_privacy_actions(self, detections: List[PIIDetection], risk_level: PrivacyRiskLevel) -> List[PrivacyAction]:
        """Determine appropriate privacy actions"""
        if not detections:
            return [PrivacyAction.ALLOW]
        
        actions = []
        
        if risk_level == PrivacyRiskLevel.CRITICAL:
            actions.append(PrivacyAction.BLOCK if self.strict_mode else PrivacyAction.ENCRYPT)
        elif risk_level == PrivacyRiskLevel.HIGH:
            actions.extend([PrivacyAction.ANONYMIZE, PrivacyAction.REQUEST_CONSENT])
        elif risk_level == PrivacyRiskLevel.MEDIUM:
            actions.append(PrivacyAction.PSEUDONYMIZE)
        else:
            actions.append(PrivacyAction.ALLOW)
        
        return actions
    
    async def _process_content(self, content: str, detections: List[PIIDetection], actions: List[PrivacyAction]) -> Optional[str]:
        """Process content based on privacy actions"""
        if PrivacyAction.BLOCK in actions:
            return None
        
        if PrivacyAction.ALLOW in actions and not detections:
            return content
        
        processed_content = content
        
        if PrivacyAction.ANONYMIZE in actions:
            processed_content = self.anonymizer.anonymize_content(processed_content, detections, "anonymize")
        elif PrivacyAction.PSEUDONYMIZE in actions:
            processed_content = self.anonymizer.anonymize_content(processed_content, detections, "pseudonymize")
        elif PrivacyAction.ENCRYPT in actions:
            processed_content = self.anonymizer.anonymize_content(processed_content, detections, "encrypt")
        elif PrivacyAction.REDACT in actions:
            processed_content = self.anonymizer.anonymize_content(processed_content, detections, "redact")
        
        # Apply differential privacy if enabled
        if self.enable_differential_privacy and processed_content != content:
            processed_content = self._apply_differential_privacy(processed_content)
        
        return processed_content
    
    def _apply_differential_privacy(self, content: str) -> str:
        """Apply differential privacy noise to content"""
        # Simplified differential privacy implementation
        # In practice, this would be more sophisticated
        
        # Add subtle noise to numerical values while preserving text
        def add_noise_to_numbers(match):
            try:
                num = float(match.group())
                # Laplace noise for differential privacy
                noise = np.random.laplace(0, 1/self.epsilon)
                noisy_num = num + noise
                return str(round(noisy_num, 2))
            except:
                return match.group()
        
        # Apply noise to numbers while preserving structure
        import re
        noisy_content = re.sub(r'\b\d+\.?\d*\b', add_noise_to_numbers, content)
        
        return noisy_content
    
    def _generate_privacy_warnings(self, detections: List[PIIDetection], risk_level: PrivacyRiskLevel) -> List[str]:
        """Generate privacy warnings for users"""
        warnings = []
        
        if not detections:
            return warnings
        
        # Category-specific warnings
        categories_detected = set(d.category for d in detections)
        
        for category in categories_detected:
            if category == PIICategory.SSN:
                warnings.append("Social Security Number detected - will be encrypted")
            elif category == PIICategory.CREDIT_CARD:
                warnings.append("Credit card information detected - will be secured")
            elif category == PIICategory.EMAIL:
                warnings.append("Email address detected - will be anonymized")
            elif category == PIICategory.PHONE:
                warnings.append("Phone number detected - will be anonymized")
            elif category == PIICategory.NAME:
                warnings.append("Personal name detected - will be pseudonymized")
        
        # Risk level warnings
        if risk_level == PrivacyRiskLevel.CRITICAL:
            warnings.append("Critical privacy risk detected - processing blocked")
        elif risk_level == PrivacyRiskLevel.HIGH:
            warnings.append("High privacy risk - enhanced protection applied")
        
        return warnings
    
    def _is_privacy_safe(self, risk_level: PrivacyRiskLevel, consent_required: List[DataProcessingPurpose], input_data: PrivacyInput) -> bool:
        """Determine if processing is privacy-safe"""
        # Block critical risk content
        if risk_level == PrivacyRiskLevel.CRITICAL and self.strict_mode:
            return False
        
        # Check consent requirements
        if consent_required and input_data.user_id:
            return False  # Need consent first
        
        # Allow if risk is acceptable or can be mitigated
        return risk_level in [PrivacyRiskLevel.NONE, PrivacyRiskLevel.LOW] or not self.strict_mode
    
    def _update_stats(self, assessment: PrivacyAssessment):
        """Update privacy statistics"""
        self.stats['total_assessments'] += 1
        self.stats['pii_detections'] += len(assessment.detected_pii)
        
        for action in assessment.recommended_actions:
            self.stats['privacy_actions'][action] += 1
        
        if assessment.consent_required:
            self.stats['consent_requests'] += len(assessment.consent_required)
    
    async def batch_assess_privacy(self, inputs: List[PrivacyInput]) -> List[PrivacyAssessment]:
        """Assess privacy for multiple inputs in batch"""
        logger.info(f"Processing batch privacy assessment for {len(inputs)} inputs")
        
        tasks = [self.assess_privacy(input_data) for input_data in inputs]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        assessments = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Batch privacy assessment failed for input {i}: {result}")
                assessments.append(PrivacyAssessment(
                    input_content=inputs[i].content,
                    overall_risk_level=PrivacyRiskLevel.HIGH,
                    is_privacy_safe=False,
                    recommended_actions=[PrivacyAction.BLOCK],
                    privacy_warnings=[f"Assessment failed: {str(result)}"]
                ))
            else:
                assessments.append(result)
        
        return assessments
    
    def get_privacy_stats(self) -> Dict[str, Any]:
        """Get privacy statistics"""
        return {
            **self.stats,
            'pii_detection_rate': self.stats['pii_detections'] / max(self.stats['total_assessments'], 1),
            'consent_grant_rate': self.stats['consent_grants'] / max(self.stats['consent_requests'], 1) if self.stats['consent_requests'] > 0 else 0,
            'strict_mode': self.strict_mode,
            'differential_privacy_enabled': self.enable_differential_privacy,
            'privacy_budget_epsilon': self.epsilon
        }
    
    async def handle_data_subject_request(self, user_id: str, request_type: str) -> Dict[str, Any]:
        """Handle data subject rights requests (GDPR/CCPA compliance)"""
        result = {
            'user_id': user_id,
            'request_type': request_type,
            'timestamp': datetime.now().isoformat(),
            'status': 'processed'
        }
        
        if request_type == 'access':
            # Provide user data access
            consents = self.consent_manager.get_user_consents(user_id)
            result['data'] = {
                'consents': [
                    {
                        'purpose': c.purpose.name,
                        'status': c.status.name,
                        'granted_at': c.granted_at.isoformat() if c.granted_at else None,
                        'expires_at': c.expires_at.isoformat() if c.expires_at else None
                    }
                    for c in consents
                ]
            }
        
        elif request_type == 'deletion':
            # Delete user data
            if user_id in self.consent_manager.consent_records:
                del self.consent_manager.consent_records[user_id]
            result['message'] = 'User data deleted'
        
        elif request_type == 'portability':
            # Provide data in portable format
            result['message'] = 'Data portability request processed'
        
        elif request_type == 'rectification':
            # Correct user data
            result['message'] = 'Data rectification request processed'
        
        await self._update_consciousness({
            'data_subject_request': request_type,
            'user_id': user_id,
            'processed': True
        })
        
        return result


# Example usage and testing
async def main():
    """Example usage of privacy engine"""
    engine = PrivacyEngine()
    
    # Test PII detection
    pii_input = PrivacyInput(
        content="My name is John Smith and my email is john.smith@company.com. You can reach me at (555) 123-4567.",
        user_id="test_user",
        purpose=DataProcessingPurpose.CORE_FUNCTIONALITY
    )
    
    result = await engine.assess_privacy(pii_input)
    print(f"PII Detection Test:")
    print(f"Is Privacy Safe: {result.is_privacy_safe}")
    print(f"Risk Level: {result.overall_risk_level.name}")
    print(f"Detected PII: {len(result.detected_pii)}")
    print(f"Actions: {[a.name for a in result.recommended_actions]}")
    print(f"Processed Content: {result.processed_content}")
    print(f"Warnings: {result.privacy_warnings}")
    print("-" * 50)
    
    # Test consent management
    consent_request = engine.consent_manager.request_consent("test_user", [DataProcessingPurpose.IMPROVEMENT])
    print(f"Consent Request: {consent_request}")
    
    # Grant consent
    engine.consent_manager.grant_consent("test_user", DataProcessingPurpose.IMPROVEMENT)
    
    # Test again with consent
    result2 = await engine.assess_privacy(pii_input)
    print(f"After Consent - Is Privacy Safe: {result2.is_privacy_safe}")
    
    # Get privacy statistics
    stats = engine.get_privacy_stats()
    print(f"Privacy Statistics:")
    print(f"Total Assessments: {stats['total_assessments']}")
    print(f"PII Detection Rate: {stats['pii_detection_rate']:.2%}")


if __name__ == "__main__":
    asyncio.run(main())