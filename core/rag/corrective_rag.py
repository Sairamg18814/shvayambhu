"""
Corrective RAG (CRAG) Implementation

Advanced RAG system with retrieval validation, correction mechanisms,
and consciousness-aware quality assessment for improved factual accuracy.
"""

import asyncio
import logging
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
import hashlib
from collections import defaultdict

from .vector_database import VectorDatabaseManager, VectorSearchResult, DocumentMetadata

logger = logging.getLogger(__name__)


class RetrievalQuality(Enum):
    """Quality assessment for retrieved documents"""
    HIGHLY_RELEVANT = "highly_relevant"
    RELEVANT = "relevant"
    PARTIALLY_RELEVANT = "partially_relevant"
    IRRELEVANT = "irrelevant"
    CONTRADICTORY = "contradictory"


class CorrectionAction(Enum):
    """Actions that can be taken to correct retrieval"""
    ACCEPT = "accept"
    FILTER = "filter"
    RERANK = "rerank"
    EXPAND_SEARCH = "expand_search"
    WEB_SEARCH = "web_search"
    KNOWLEDGE_SYNTHESIS = "knowledge_synthesis"


@dataclass
class CRAGConfig:
    """Configuration for Corrective RAG system"""
    # Quality thresholds
    relevance_threshold: float = 0.7
    confidence_threshold: float = 0.8
    factual_accuracy_threshold: float = 0.75
    
    # Retrieval settings
    initial_retrieval_k: int = 20
    final_retrieval_k: int = 5
    max_correction_iterations: int = 3
    
    # Validation settings
    enable_fact_checking: bool = True
    enable_consistency_check: bool = True
    enable_relevance_scoring: bool = True
    
    # Consciousness integration
    consciousness_weight: float = 1.3
    prefer_consciousness_content: bool = True
    
    # Web search fallback
    enable_web_fallback: bool = True
    web_search_threshold: float = 0.5  # If quality below this, try web search
    
    # Knowledge synthesis
    enable_synthesis: bool = True
    synthesis_similarity_threshold: float = 0.6


@dataclass
class QualityAssessment:
    """Assessment of retrieved document quality"""
    document_id: str
    relevance_score: float
    factual_accuracy: float
    consistency_score: float
    consciousness_relevance: float
    overall_quality: RetrievalQuality
    confidence: float
    issues: List[str] = field(default_factory=list)
    corrections_applied: List[str] = field(default_factory=list)


@dataclass
class CorrectionResult:
    """Result of a correction operation"""
    original_results: List[VectorSearchResult]
    corrected_results: List[VectorSearchResult]
    quality_assessments: List[QualityAssessment]
    actions_taken: List[CorrectionAction]
    correction_confidence: float
    total_correction_time: float


class RelevanceScorer:
    """Scores relevance between query and retrieved documents"""
    
    def __init__(self, config: CRAGConfig):
        self.config = config
        
        # Simple keyword-based relevance patterns
        self.relevance_patterns = {
            'exact_match': 2.0,
            'partial_match': 1.0,
            'semantic_match': 0.8,
            'conceptual_match': 0.6
        }
    
    def score_relevance(self, query: str, document: str, metadata: DocumentMetadata) -> float:
        """Score relevance between query and document"""
        
        query_words = set(query.lower().split())
        doc_words = set(document.lower().split())
        
        if not query_words or not doc_words:
            return 0.0
        
        # Exact word matches
        exact_matches = len(query_words.intersection(doc_words))
        exact_score = (exact_matches / len(query_words)) * self.relevance_patterns['exact_match']
        
        # Partial matches (word stems, etc.)
        partial_matches = self._count_partial_matches(query_words, doc_words)
        partial_score = (partial_matches / len(query_words)) * self.relevance_patterns['partial_match']
        
        # Keyword matches from metadata
        keyword_matches = 0
        for keyword in metadata.keywords:
            if keyword.lower() in query.lower():
                keyword_matches += 1
        keyword_score = (keyword_matches / max(1, len(metadata.keywords))) * 0.3
        
        # Consciousness relevance boost
        consciousness_boost = 0.0
        if self.config.prefer_consciousness_content and metadata.consciousness_score > 0.5:
            consciousness_boost = metadata.consciousness_score * 0.2
        
        # Combine scores
        total_score = min(1.0, exact_score + partial_score + keyword_score + consciousness_boost)
        
        return total_score
    
    def _count_partial_matches(self, query_words: set, doc_words: set) -> int:
        """Count partial word matches (simple stemming)"""
        
        partial_matches = 0
        
        for query_word in query_words:
            for doc_word in doc_words:
                # Simple partial matching
                if len(query_word) > 4 and len(doc_word) > 4:
                    if (query_word[:4] == doc_word[:4] or 
                        query_word[-3:] == doc_word[-3:]):
                        partial_matches += 1
                        break
        
        return partial_matches


class FactChecker:
    """Validates factual accuracy of retrieved content"""
    
    def __init__(self, config: CRAGConfig):
        self.config = config
        
        # Simple fact-checking patterns
        self.uncertainty_indicators = [
            'maybe', 'perhaps', 'possibly', 'might', 'could be',
            'allegedly', 'reportedly', 'supposedly', 'potentially'
        ]
        
        self.certainty_indicators = [
            'definitely', 'certainly', 'clearly', 'obviously', 'proven',
            'established', 'confirmed', 'verified', 'demonstrated'
        ]
    
    def check_factual_accuracy(self, content: str, metadata: DocumentMetadata) -> float:
        """Assess factual accuracy of content"""
        
        if not self.config.enable_fact_checking:
            return 1.0
        
        content_lower = content.lower()
        
        # Source credibility (higher score for trusted sources)
        source_score = self._assess_source_credibility(metadata.source)
        
        # Uncertainty vs certainty language
        uncertainty_count = sum(1 for indicator in self.uncertainty_indicators 
                               if indicator in content_lower)
        certainty_count = sum(1 for indicator in self.certainty_indicators 
                             if indicator in content_lower)
        
        language_score = 0.5  # Neutral baseline
        total_indicators = uncertainty_count + certainty_count
        
        if total_indicators > 0:
            certainty_ratio = certainty_count / total_indicators
            language_score = 0.3 + (certainty_ratio * 0.7)  # Scale from 0.3 to 1.0
        
        # Content length factor (very short content might be less reliable)
        length_score = min(1.0, len(content) / 200)  # Full score at 200+ chars
        
        # Consciousness content gets slight factual boost (self-aware content)
        consciousness_boost = metadata.consciousness_score * 0.1
        
        # Combined factual accuracy score
        factual_accuracy = (
            source_score * 0.4 +
            language_score * 0.3 +
            length_score * 0.2 +
            consciousness_boost * 0.1
        )
        
        return min(1.0, factual_accuracy)
    
    def _assess_source_credibility(self, source: str) -> float:
        """Assess credibility of content source"""
        
        # Simple source credibility mapping
        high_credibility_sources = [
            'arxiv.org', 'nature.com', 'science.org', 'ieee.org',
            'acm.org', '.edu', 'research', 'academic'
        ]
        
        medium_credibility_sources = [
            'wikipedia', 'news', 'journal', 'publication'
        ]
        
        source_lower = source.lower()
        
        for high_source in high_credibility_sources:
            if high_source in source_lower:
                return 0.9
        
        for medium_source in medium_credibility_sources:
            if medium_source in source_lower:
                return 0.7
        
        return 0.5  # Default medium credibility


class ConsistencyChecker:
    """Checks consistency between retrieved documents"""
    
    def __init__(self, config: CRAGConfig):
        self.config = config
    
    def check_consistency(
        self, 
        documents: List[VectorSearchResult], 
        query: str
    ) -> List[float]:
        """Check consistency scores for each document against others"""
        
        if not self.config.enable_consistency_check or len(documents) < 2:
            return [1.0] * len(documents)
        
        consistency_scores = []
        
        for i, doc in enumerate(documents):
            # Compare with other documents
            agreement_scores = []
            
            for j, other_doc in enumerate(documents):
                if i != j:
                    agreement = self._calculate_agreement(doc.content, other_doc.content)
                    agreement_scores.append(agreement)
            
            # Calculate overall consistency
            if agreement_scores:
                consistency_score = np.mean(agreement_scores)
            else:
                consistency_score = 1.0
            
            consistency_scores.append(consistency_score)
        
        return consistency_scores
    
    def _calculate_agreement(self, content1: str, content2: str) -> float:
        """Calculate agreement between two pieces of content"""
        
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())
        
        if not words1 or not words2:
            return 0.5
        
        # Jaccard similarity as a proxy for agreement
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        agreement = len(intersection) / len(union) if union else 0.0
        
        return agreement


class QualityAssessor:
    """Comprehensive quality assessment for retrieved documents"""
    
    def __init__(self, config: CRAGConfig):
        self.config = config
        self.relevance_scorer = RelevanceScorer(config)
        self.fact_checker = FactChecker(config)
        self.consistency_checker = ConsistencyChecker(config)
    
    def assess_quality(
        self,
        query: str,
        documents: List[VectorSearchResult]
    ) -> List[QualityAssessment]:
        """Perform comprehensive quality assessment"""
        
        if not documents:
            return []
        
        # Get consistency scores for all documents
        consistency_scores = self.consistency_checker.check_consistency(documents, query)
        
        assessments = []
        
        for i, doc in enumerate(documents):
            # Individual quality metrics
            relevance_score = self.relevance_scorer.score_relevance(
                query, doc.content, doc.metadata
            )
            
            factual_accuracy = self.fact_checker.check_factual_accuracy(
                doc.content, doc.metadata
            )
            
            consistency_score = consistency_scores[i]
            
            # Consciousness relevance
            consciousness_relevance = self._assess_consciousness_relevance(
                query, doc.content, doc.metadata
            )
            
            # Overall quality classification
            overall_quality, confidence, issues = self._classify_quality(
                relevance_score, factual_accuracy, consistency_score, consciousness_relevance
            )
            
            assessment = QualityAssessment(
                document_id=doc.document_id,
                relevance_score=relevance_score,
                factual_accuracy=factual_accuracy,
                consistency_score=consistency_score,
                consciousness_relevance=consciousness_relevance,
                overall_quality=overall_quality,
                confidence=confidence,
                issues=issues
            )
            
            assessments.append(assessment)
        
        return assessments
    
    def _assess_consciousness_relevance(
        self, 
        query: str, 
        content: str, 
        metadata: DocumentMetadata
    ) -> float:
        """Assess consciousness relevance of content"""
        
        query_lower = query.lower()
        content_lower = content.lower()
        
        consciousness_keywords = [
            'consciousness', 'awareness', 'experience', 'subjective', 'qualia',
            'phenomenal', 'introspection', 'self-aware', 'sentient', 'cognitive'
        ]
        
        # Query consciousness relevance
        query_consciousness = sum(1 for kw in consciousness_keywords if kw in query_lower)
        query_consciousness_score = min(1.0, query_consciousness / len(consciousness_keywords))
        
        # Content consciousness relevance
        content_consciousness = sum(1 for kw in consciousness_keywords if kw in content_lower)
        content_consciousness_score = min(1.0, content_consciousness / len(consciousness_keywords))
        
        # Metadata consciousness score
        metadata_consciousness_score = metadata.consciousness_score
        
        # Combined consciousness relevance
        consciousness_relevance = (
            query_consciousness_score * 0.3 +
            content_consciousness_score * 0.4 +
            metadata_consciousness_score * 0.3
        )
        
        return consciousness_relevance
    
    def _classify_quality(
        self,
        relevance_score: float,
        factual_accuracy: float,
        consistency_score: float,
        consciousness_relevance: float
    ) -> Tuple[RetrievalQuality, float, List[str]]:
        """Classify overall quality and identify issues"""
        
        issues = []
        
        # Check individual scores
        if relevance_score < self.config.relevance_threshold:
            issues.append("Low relevance to query")
        
        if factual_accuracy < self.config.factual_accuracy_threshold:
            issues.append("Questionable factual accuracy")
        
        if consistency_score < 0.5:
            issues.append("Inconsistent with other retrieved content")
        
        # Calculate weighted overall score
        weights = [0.4, 0.3, 0.2, 0.1]  # relevance, factual, consistency, consciousness
        scores = [relevance_score, factual_accuracy, consistency_score, consciousness_relevance]
        
        overall_score = sum(w * s for w, s in zip(weights, scores))
        confidence = min(scores)  # Confidence limited by weakest score
        
        # Classify quality
        if overall_score >= 0.85 and confidence >= self.config.confidence_threshold:
            quality = RetrievalQuality.HIGHLY_RELEVANT
        elif overall_score >= 0.7 and confidence >= 0.6:
            quality = RetrievalQuality.RELEVANT
        elif overall_score >= 0.5:
            quality = RetrievalQuality.PARTIALLY_RELEVANT
        elif consistency_score < 0.3:  # Contradictory
            quality = RetrievalQuality.CONTRADICTORY
            issues.append("Contradicts other retrieved content")
        else:
            quality = RetrievalQuality.IRRELEVANT
        
        return quality, confidence, issues


class CRAGCorrector:
    """Applies corrections to improve retrieval quality"""
    
    def __init__(
        self,
        config: CRAGConfig,
        vector_manager: VectorDatabaseManager,
        web_search_function: Optional[Callable] = None
    ):
        self.config = config
        self.vector_manager = vector_manager
        self.web_search_function = web_search_function
    
    async def correct_retrieval(
        self,
        query: str,
        initial_results: List[VectorSearchResult],
        quality_assessments: List[QualityAssessment]
    ) -> CorrectionResult:
        """Apply corrections to improve retrieval quality"""
        
        start_time = datetime.now()
        actions_taken = []
        corrected_results = initial_results.copy()
        
        # Determine correction actions needed
        correction_actions = self._determine_correction_actions(quality_assessments)
        
        for action in correction_actions:
            if action == CorrectionAction.FILTER:
                corrected_results = self._filter_low_quality(corrected_results, quality_assessments)
                actions_taken.append(action)
                
            elif action == CorrectionAction.RERANK:
                corrected_results = self._rerank_results(corrected_results, quality_assessments)
                actions_taken.append(action)
                
            elif action == CorrectionAction.EXPAND_SEARCH:
                expanded_results = await self._expand_search(query, corrected_results)
                corrected_results = self._merge_results(corrected_results, expanded_results)
                actions_taken.append(action)
                
            elif action == CorrectionAction.WEB_SEARCH and self.web_search_function:
                web_results = await self._web_search_fallback(query)
                if web_results:
                    corrected_results = self._merge_results(corrected_results, web_results)
                    actions_taken.append(action)
                
            elif action == CorrectionAction.KNOWLEDGE_SYNTHESIS:
                synthesized_results = self._synthesize_knowledge(corrected_results, query)
                if synthesized_results:
                    corrected_results = synthesized_results
                    actions_taken.append(action)
        
        # Reassess quality after corrections
        quality_assessor = QualityAssessor(self.config)
        final_assessments = quality_assessor.assess_quality(query, corrected_results)
        
        # Calculate correction confidence
        correction_confidence = self._calculate_correction_confidence(
            quality_assessments, final_assessments
        )
        
        correction_time = (datetime.now() - start_time).total_seconds()
        
        return CorrectionResult(
            original_results=initial_results,
            corrected_results=corrected_results[:self.config.final_retrieval_k],
            quality_assessments=final_assessments,
            actions_taken=actions_taken,
            correction_confidence=correction_confidence,
            total_correction_time=correction_time
        )
    
    def _determine_correction_actions(
        self,
        quality_assessments: List[QualityAssessment]
    ) -> List[CorrectionAction]:
        """Determine what correction actions are needed"""
        
        actions = []
        
        # Count quality levels
        quality_counts = defaultdict(int)
        for assessment in quality_assessments:
            quality_counts[assessment.overall_quality] += 1
        
        # Check if filtering is needed
        low_quality_count = (
            quality_counts[RetrievalQuality.IRRELEVANT] + 
            quality_counts[RetrievalQuality.CONTRADICTORY]
        )
        
        if low_quality_count > 0:
            actions.append(CorrectionAction.FILTER)
        
        # Check if reranking would help
        if quality_counts[RetrievalQuality.PARTIALLY_RELEVANT] > 0:
            actions.append(CorrectionAction.RERANK)
        
        # Check if we need more results
        high_quality_count = (
            quality_counts[RetrievalQuality.HIGHLY_RELEVANT] +
            quality_counts[RetrievalQuality.RELEVANT]
        )
        
        if high_quality_count < self.config.final_retrieval_k:
            actions.append(CorrectionAction.EXPAND_SEARCH)
        
        # Check if web search fallback is needed
        if (self.config.enable_web_fallback and 
            high_quality_count == 0 and 
            self.web_search_function):
            actions.append(CorrectionAction.WEB_SEARCH)
        
        # Check if knowledge synthesis could help
        if (self.config.enable_synthesis and 
            quality_counts[RetrievalQuality.PARTIALLY_RELEVANT] >= 2):
            actions.append(CorrectionAction.KNOWLEDGE_SYNTHESIS)
        
        return actions
    
    def _filter_low_quality(
        self,
        results: List[VectorSearchResult],
        assessments: List[QualityAssessment]
    ) -> List[VectorSearchResult]:
        """Filter out low-quality results"""
        
        filtered_results = []
        
        for result, assessment in zip(results, assessments):
            if assessment.overall_quality not in [RetrievalQuality.IRRELEVANT, RetrievalQuality.CONTRADICTORY]:
                filtered_results.append(result)
        
        return filtered_results
    
    def _rerank_results(
        self,
        results: List[VectorSearchResult],
        assessments: List[QualityAssessment]
    ) -> List[VectorSearchResult]:
        """Rerank results based on quality assessments"""
        
        # Create ranking score combining similarity and quality
        ranked_items = []
        
        for result, assessment in zip(results, assessments):
            # Combine original similarity with quality metrics
            ranking_score = (
                result.similarity_score * 0.3 +
                assessment.relevance_score * 0.3 +
                assessment.factual_accuracy * 0.2 +
                assessment.consistency_score * 0.1 +
                assessment.consciousness_relevance * self.config.consciousness_weight * 0.1
            )
            
            ranked_items.append((ranking_score, result))
        
        # Sort by ranking score
        ranked_items.sort(key=lambda x: x[0], reverse=True)
        
        return [item[1] for item in ranked_items]
    
    async def _expand_search(
        self,
        query: str,
        current_results: List[VectorSearchResult]
    ) -> List[VectorSearchResult]:
        """Expand search with modified query parameters"""
        
        # Try different search strategies
        expanded_results = []
        
        # Strategy 1: Lower similarity threshold
        lower_threshold_config = self.vector_manager.config
        original_threshold = lower_threshold_config.similarity_threshold
        lower_threshold_config.similarity_threshold = max(0.3, original_threshold - 0.2)
        
        try:
            expanded = await self.vector_manager.search(
                query, 
                top_k=self.config.initial_retrieval_k * 2,
                use_cache=False
            )
            expanded_results.extend(expanded)
        finally:
            lower_threshold_config.similarity_threshold = original_threshold
        
        # Strategy 2: Query expansion with related terms
        expanded_query = self._expand_query_terms(query)
        if expanded_query != query:
            expanded = await self.vector_manager.search(
                expanded_query,
                top_k=self.config.initial_retrieval_k,
                use_cache=False
            )
            expanded_results.extend(expanded)
        
        return expanded_results
    
    def _expand_query_terms(self, query: str) -> str:
        """Expand query with related terms"""
        
        # Simple query expansion with synonyms/related terms
        expansion_map = {
            'consciousness': 'consciousness awareness experience subjective',
            'ai': 'artificial intelligence machine learning neural networks',
            'learning': 'learning training education knowledge',
            'model': 'model system architecture framework',
        }
        
        expanded_terms = []
        query_words = query.lower().split()
        
        for word in query_words:
            expanded_terms.append(word)
            if word in expansion_map:
                expanded_terms.extend(expansion_map[word].split())
        
        return ' '.join(expanded_terms)
    
    async def _web_search_fallback(self, query: str) -> List[VectorSearchResult]:
        """Fallback to web search when retrieval quality is poor"""
        
        if not self.web_search_function:
            return []
        
        try:
            web_results = await self.web_search_function(query)
            
            # Convert web results to VectorSearchResult format
            vector_results = []
            for i, result in enumerate(web_results):
                # Create metadata for web result
                metadata = DocumentMetadata(
                    document_id=f"web_{hashlib.md5(result.get('url', '').encode()).hexdigest()}",
                    source=result.get('source', 'web'),
                    title=result.get('title', 'Web Result'),
                    content_type='web',
                    timestamp=datetime.now(),
                    keywords=result.get('keywords', []),
                    consciousness_score=0.5,  # Neutral score for web content
                    relevance_score=result.get('relevance_score', 0.7)
                )
                
                vector_result = VectorSearchResult(
                    document_id=metadata.document_id,
                    content=result.get('content', ''),
                    similarity_score=result.get('similarity_score', 0.6),
                    metadata=metadata
                )
                
                vector_results.append(vector_result)
            
            return vector_results
            
        except Exception as e:
            logger.error(f"Web search fallback failed: {e}")
            return []
    
    def _synthesize_knowledge(
        self,
        results: List[VectorSearchResult],
        query: str
    ) -> Optional[List[VectorSearchResult]]:
        """Synthesize knowledge from multiple partially relevant results"""
        
        if len(results) < 2:
            return None
        
        # Group similar results for synthesis
        synthesis_groups = self._group_similar_results(results)
        
        synthesized_results = []
        
        for group in synthesis_groups:
            if len(group) >= 2:
                synthesized = self._create_synthesized_result(group, query)
                if synthesized:
                    synthesized_results.append(synthesized)
        
        # Add original high-quality results that weren't synthesized
        for result in results:
            if not any(result in group for group in synthesis_groups):
                synthesized_results.append(result)
        
        return synthesized_results if synthesized_results else None
    
    def _group_similar_results(self, results: List[VectorSearchResult]) -> List[List[VectorSearchResult]]:
        """Group similar results for synthesis"""
        
        groups = []
        used_results = set()
        
        for i, result in enumerate(results):
            if i in used_results:
                continue
            
            current_group = [result]
            used_results.add(i)
            
            # Find similar results
            for j, other_result in enumerate(results[i+1:], i+1):
                if j in used_results:
                    continue
                
                similarity = self._calculate_content_similarity(result.content, other_result.content)
                if similarity >= self.config.synthesis_similarity_threshold:
                    current_group.append(other_result)
                    used_results.add(j)
            
            groups.append(current_group)
        
        return groups
    
    def _calculate_content_similarity(self, content1: str, content2: str) -> float:
        """Calculate similarity between two pieces of content"""
        
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def _create_synthesized_result(
        self,
        group: List[VectorSearchResult],
        query: str
    ) -> Optional[VectorSearchResult]:
        """Create a synthesized result from a group of similar results"""
        
        if len(group) < 2:
            return None
        
        # Combine content from group
        combined_content = self._combine_content(group)
        
        # Create synthesized metadata
        avg_consciousness = np.mean([r.metadata.consciousness_score for r in group])
        avg_relevance = np.mean([r.metadata.relevance_score for r in group])
        max_similarity = max([r.similarity_score for r in group])
        
        # Combine keywords
        all_keywords = []
        for result in group:
            all_keywords.extend(result.metadata.keywords)
        unique_keywords = list(set(all_keywords))
        
        synthesized_metadata = DocumentMetadata(
            document_id=f"synthesized_{hashlib.md5(combined_content.encode()).hexdigest()[:8]}",
            source="synthesized",
            title=f"Synthesized Knowledge: {query[:50]}",
            content_type="synthesis",
            timestamp=datetime.now(),
            keywords=unique_keywords[:10],  # Limit keywords
            consciousness_score=avg_consciousness,
            relevance_score=avg_relevance,
            custom_metadata={
                'source_count': len(group),
                'source_ids': [r.document_id for r in group]
            }
        )
        
        return VectorSearchResult(
            document_id=synthesized_metadata.document_id,
            content=combined_content,
            similarity_score=max_similarity * 1.1,  # Slight boost for synthesis
            metadata=synthesized_metadata
        )
    
    def _combine_content(self, group: List[VectorSearchResult]) -> str:
        """Combine content from multiple results"""
        
        # Simple content combination - in a real system, this would be more sophisticated
        contents = []
        
        for result in group:
            content = result.content.strip()
            if content and content not in contents:
                contents.append(content)
        
        # Combine with synthesis markers
        if len(contents) > 1:
            combined = "**Synthesized Knowledge:**\n\n"
            for i, content in enumerate(contents, 1):
                combined += f"**Source {i}:** {content}\n\n"
        else:
            combined = contents[0] if contents else ""
        
        return combined
    
    def _merge_results(
        self,
        original_results: List[VectorSearchResult],
        new_results: List[VectorSearchResult]
    ) -> List[VectorSearchResult]:
        """Merge new results with original results, avoiding duplicates"""
        
        existing_ids = {result.document_id for result in original_results}
        merged = original_results.copy()
        
        for result in new_results:
            if result.document_id not in existing_ids:
                merged.append(result)
                existing_ids.add(result.document_id)
        
        return merged
    
    def _calculate_correction_confidence(
        self,
        original_assessments: List[QualityAssessment],
        corrected_assessments: List[QualityAssessment]
    ) -> float:
        """Calculate confidence in the correction process"""
        
        if not corrected_assessments:
            return 0.0
        
        # Calculate average quality improvement
        original_avg_confidence = np.mean([a.confidence for a in original_assessments]) if original_assessments else 0.0
        corrected_avg_confidence = np.mean([a.confidence for a in corrected_assessments])
        
        improvement = corrected_avg_confidence - original_avg_confidence
        
        # Confidence is based on final quality and improvement
        correction_confidence = min(1.0, corrected_avg_confidence + max(0, improvement * 0.5))
        
        return correction_confidence


class CorrectiveRAG:
    """Main Corrective RAG system orchestrator"""
    
    def __init__(
        self,
        config: CRAGConfig,
        vector_manager: VectorDatabaseManager,
        web_search_function: Optional[Callable] = None
    ):
        self.config = config
        self.vector_manager = vector_manager
        self.quality_assessor = QualityAssessor(config)
        self.corrector = CRAGCorrector(config, vector_manager, web_search_function)
        
        # Performance metrics
        self.metrics = {
            'total_queries': 0,
            'corrections_applied': 0,
            'average_correction_time': 0.0,
            'quality_improvements': 0,
            'web_fallbacks': 0
        }
    
    async def retrieve_and_correct(
        self,
        query: str,
        top_k: Optional[int] = None,
        max_correction_iterations: Optional[int] = None
    ) -> CorrectionResult:
        """Main method: retrieve documents and apply corrections"""
        
        start_time = datetime.now()
        top_k = top_k or self.config.final_retrieval_k
        max_iterations = max_correction_iterations or self.config.max_correction_iterations
        
        # Initial retrieval
        initial_results = await self.vector_manager.search(
            query,
            top_k=self.config.initial_retrieval_k,
            use_cache=True
        )
        
        if not initial_results:
            logger.warning(f"No initial results found for query: {query}")
            return CorrectionResult(
                original_results=[],
                corrected_results=[],
                quality_assessments=[],
                actions_taken=[],
                correction_confidence=0.0,
                total_correction_time=0.0
            )
        
        # Initial quality assessment
        quality_assessments = self.quality_assessor.assess_quality(query, initial_results)
        
        # Check if corrections are needed
        needs_correction = any(
            assessment.overall_quality in [RetrievalQuality.IRRELEVANT, RetrievalQuality.CONTRADICTORY]
            or assessment.confidence < self.config.confidence_threshold
            for assessment in quality_assessments
        )
        
        if not needs_correction:
            # No corrections needed, return top results
            final_results = initial_results[:top_k]
            final_assessments = quality_assessments[:top_k]
            
            correction_result = CorrectionResult(
                original_results=initial_results,
                corrected_results=final_results,
                quality_assessments=final_assessments,
                actions_taken=[CorrectionAction.ACCEPT],
                correction_confidence=np.mean([a.confidence for a in final_assessments]),
                total_correction_time=(datetime.now() - start_time).total_seconds()
            )
        else:
            # Apply corrections
            correction_result = await self.corrector.correct_retrieval(
                query, initial_results, quality_assessments
            )
        
        # Update metrics
        self.metrics['total_queries'] += 1
        if CorrectionAction.ACCEPT not in correction_result.actions_taken:
            self.metrics['corrections_applied'] += 1
        
        if CorrectionAction.WEB_SEARCH in correction_result.actions_taken:
            self.metrics['web_fallbacks'] += 1
        
        total_time = (datetime.now() - start_time).total_seconds()
        self.metrics['average_correction_time'] = (
            (self.metrics['average_correction_time'] * (self.metrics['total_queries'] - 1) + total_time)
            / self.metrics['total_queries']
        )
        
        logger.info(f"CRAG retrieval completed in {total_time:.3f}s with "
                   f"{len(correction_result.corrected_results)} results "
                   f"(confidence: {correction_result.correction_confidence:.3f})")
        
        return correction_result
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get CRAG performance statistics"""
        
        return {
            'crag_metrics': self.metrics.copy(),
            'vector_db_stats': await self.vector_manager.get_statistics(),
            'config': {
                'relevance_threshold': self.config.relevance_threshold,
                'confidence_threshold': self.config.confidence_threshold,
                'enable_fact_checking': self.config.enable_fact_checking,
                'enable_web_fallback': self.config.enable_web_fallback,
                'consciousness_weight': self.config.consciousness_weight
            }
        }


# Testing and example usage
async def test_corrective_rag():
    """Test Corrective RAG system"""
    
    # This would typically be integrated with the vector database
    print("Corrective RAG test would require full vector database integration")
    print("Test framework created - ready for integration")


if __name__ == "__main__":
    asyncio.run(test_corrective_rag())