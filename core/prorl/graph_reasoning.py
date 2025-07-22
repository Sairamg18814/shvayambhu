"""Graph-of-Thought reasoning strategy implementation.

This module implements the Graph-of-Thought (GoT) reasoning approach that creates
interconnected networks of thoughts, allows for complex relationships between ideas,
and uses graph algorithms for sophisticated reasoning patterns.
"""

import torch
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import uuid
import time
import numpy as np
from collections import defaultdict, deque
import networkx as nx
import warnings

from .reasoning_engine import ReasoningStep, ReasoningTrace, StepType, ReasoningStrategy


class NodeType(Enum):
    """Types of nodes in the reasoning graph."""
    ROOT = "root"
    CONCEPT = "concept"
    EVIDENCE = "evidence"
    HYPOTHESIS = "hypothesis"
    CONCLUSION = "conclusion"
    SYNTHESIS = "synthesis"
    QUESTION = "question"
    ASSUMPTION = "assumption"
    COUNTERARGUMENT = "counterargument"


class EdgeType(Enum):
    """Types of edges in the reasoning graph."""
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    ELABORATES = "elaborates"
    IMPLIES = "implies"
    CAUSED_BY = "caused_by"
    SIMILAR_TO = "similar_to"
    PART_OF = "part_of"
    LEADS_TO = "leads_to"
    DEPENDS_ON = "depends_on"
    SYNTHESIZES = "synthesizes"


class PropagationStrategy(Enum):
    """Strategies for propagating information through the graph."""
    BREADTH_FIRST = "breadth_first"
    DEPTH_FIRST = "depth_first"
    CENTRALITY_BASED = "centrality_based"
    INFLUENCE_BASED = "influence_based"
    SHORTEST_PATH = "shortest_path"
    RANDOM_WALK = "random_walk"


@dataclass
class ReasoningNode:
    """A node in the Graph-of-Thought structure."""
    node_id: str
    content: str
    node_type: NodeType
    confidence: float
    
    # Graph properties
    incoming_edges: List[str] = field(default_factory=list)
    outgoing_edges: List[str] = field(default_factory=list)
    
    # Evaluation metrics
    centrality_score: float = 0.0
    influence_score: float = 0.0
    evidence_support: float = 0.0
    contradiction_level: float = 0.0
    
    # State tracking
    last_updated: float = field(default_factory=time.time)
    activation_level: float = 0.0
    visit_count: int = 0
    
    # Metadata
    domain_tags: List[str] = field(default_factory=list)
    reasoning_step_id: Optional[str] = None
    source_references: List[str] = field(default_factory=list)


@dataclass
class ReasoningEdge:
    """An edge in the Graph-of-Thought structure."""
    edge_id: str
    source_id: str
    target_id: str
    edge_type: EdgeType
    strength: float
    
    # Edge properties
    confidence: float = 0.5
    bidirectional: bool = False
    created_at: float = field(default_factory=time.time)
    
    # Supporting information
    evidence: List[str] = field(default_factory=list)
    reasoning: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GoTConfig:
    """Configuration for Graph-of-Thought reasoning."""
    # Graph structure
    max_nodes: int = 500
    max_edges_per_node: int = 10
    min_edge_strength: float = 0.2
    
    # Node management
    activation_threshold: float = 0.3
    decay_rate: float = 0.05
    max_activation_spread: int = 3
    
    # Edge creation
    similarity_threshold: float = 0.6
    auto_create_edges: bool = True
    enable_contradiction_detection: bool = True
    
    # Graph algorithms
    propagation_strategy: PropagationStrategy = PropagationStrategy.INFLUENCE_BASED
    centrality_algorithm: str = "pagerank"  # "pagerank", "betweenness", "closeness"
    max_propagation_steps: int = 5
    
    # Synthesis and reasoning
    synthesis_threshold: float = 0.7
    max_synthesis_inputs: int = 5
    enable_emergent_insights: bool = True
    
    # Performance optimization
    enable_caching: bool = True
    cache_size: int = 1000
    periodic_cleanup: bool = True


class GraphOfThoughtReasoner:
    """Implements Graph-of-Thought reasoning strategy."""
    
    def __init__(self, config: GoTConfig):
        self.config = config
        
        # Graph structure
        self.nodes: Dict[str, ReasoningNode] = {}
        self.edges: Dict[str, ReasoningEdge] = {}
        self.graph: nx.DiGraph = nx.DiGraph()
        
        # State management
        self.active_nodes: Set[str] = set()
        self.root_id: Optional[str] = None
        self.current_focus: Optional[str] = None
        
        # Algorithm state
        self.propagation_history: List[Dict[str, Any]] = []
        self.synthesis_history: List[Dict[str, Any]] = []
        
        # Caching
        self.centrality_cache: Dict[str, float] = {}
        self.path_cache: Dict[Tuple[str, str], List[str]] = {}
        self.similarity_cache: Dict[Tuple[str, str], float] = {}
        
        # Statistics
        self.reasoning_steps = 0
        self.synthesis_count = 0
        self.edge_creation_count = 0
    
    def reason(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        max_iterations: int = 50
    ) -> Tuple[str, float, Dict[str, Any]]:
        """Execute Graph-of-Thought reasoning."""
        
        # Initialize graph with root node
        self.root_id = self._create_root_node(query)
        
        # Initial expansion
        self._initial_expansion(query, context)
        
        # Main reasoning loop
        for iteration in range(max_iterations):
            # Select focus node
            focus_node_id = self._select_focus_node()
            if not focus_node_id:
                break
            
            self.current_focus = focus_node_id
            
            # Activate surrounding nodes
            self._activate_neighborhood(focus_node_id)
            
            # Generate new insights
            new_nodes = self._generate_insights(focus_node_id, query, context)
            
            # Create edges between related nodes
            self._create_automatic_edges(new_nodes)
            
            # Propagate activation through graph
            self._propagate_activation()
            
            # Attempt synthesis
            synthesis_results = self._attempt_synthesis()
            
            # Update graph metrics
            self._update_graph_metrics()
            
            # Check for convergence
            if self._check_convergence():
                break
            
            # Periodic cleanup
            if iteration % 10 == 0 and self.config.periodic_cleanup:
                self._cleanup_graph()
        
        # Generate final answer
        final_answer, confidence = self._generate_final_answer(query)
        
        # Prepare reasoning graph for return
        reasoning_graph = self._export_reasoning_graph()
        
        return final_answer, confidence, reasoning_graph
    
    def _create_root_node(self, query: str) -> str:
        """Create the root node for the reasoning graph."""
        
        node_id = str(uuid.uuid4())
        root_node = ReasoningNode(
            node_id=node_id,
            content=f"Query: {query}",
            node_type=NodeType.ROOT,
            confidence=1.0,
            activation_level=1.0
        )
        
        self.nodes[node_id] = root_node
        self.graph.add_node(node_id, **self._node_to_dict(root_node))
        self.active_nodes.add(node_id)
        
        return node_id
    
    def _initial_expansion(self, query: str, context: Optional[Dict[str, Any]]):
        """Create initial nodes from query analysis."""
        
        # Analyze query to extract key concepts
        concepts = self._extract_concepts(query)
        questions = self._generate_questions(query)
        assumptions = self._identify_assumptions(query, context)
        
        # Create concept nodes
        for concept in concepts:
            self._create_node(concept, NodeType.CONCEPT, confidence=0.7)
        
        # Create question nodes
        for question in questions:
            self._create_node(question, NodeType.QUESTION, confidence=0.6)
        
        # Create assumption nodes
        for assumption in assumptions:
            self._create_node(assumption, NodeType.ASSUMPTION, confidence=0.5)
        
        # Connect to root
        for node_id in list(self.nodes.keys()):
            if node_id != self.root_id:
                self._create_edge(
                    self.root_id, node_id, EdgeType.ELABORATES, strength=0.5
                )
    
    def _extract_concepts(self, query: str) -> List[str]:
        """Extract key concepts from the query."""
        
        # Simple concept extraction (would use NLP in practice)
        concepts = []
        
        # Split into sentences and extract key phrases
        sentences = query.split('.')
        for sentence in sentences:
            words = sentence.strip().split()
            
            # Look for noun phrases and important terms
            if len(words) >= 3:
                # Extract potential concepts
                for i in range(len(words) - 2):
                    phrase = ' '.join(words[i:i+3])
                    if len(phrase) > 10:  # Meaningful length
                        concepts.append(f"Concept: {phrase}")
        
        # Add single important words
        important_words = [
            word for word in query.split()
            if len(word) > 4 and word.isalpha()
        ]
        
        for word in important_words[:5]:  # Limit to 5
            concepts.append(f"Key term: {word}")
        
        return concepts[:10]  # Limit concepts
    
    def _generate_questions(self, query: str) -> List[str]:
        """Generate relevant questions from the query."""
        
        questions = []
        
        # Question templates based on query type
        if "?" in query:
            # Already a question, generate related questions
            questions.extend([
                "What are the underlying assumptions?",
                "What evidence would support this?",
                "What are alternative perspectives?"
            ])
        else:
            # Statement, generate investigative questions
            questions.extend([
                f"Why is this important: {query[:50]}...?",
                "What are the implications?",
                "How can this be verified?",
                "What are potential counterarguments?"
            ])
        
        return questions[:5]
    
    def _identify_assumptions(self, query: str, context: Optional[Dict[str, Any]]) -> List[str]:
        """Identify implicit assumptions in the query."""
        
        assumptions = []
        
        # Look for assumption indicators
        assumption_indicators = [
            "assume", "given", "suppose", "if", "provided that"
        ]
        
        query_lower = query.lower()
        for indicator in assumption_indicators:
            if indicator in query_lower:
                assumptions.append(f"Assumption: {indicator} clause identified")
        
        # Default assumptions for complex queries
        if len(query.split()) > 10:
            assumptions.extend([
                "Assumption: Standard definitions apply",
                "Assumption: Context is current and relevant"
            ])
        
        return assumptions[:3]
    
    def _create_node(
        self,
        content: str,
        node_type: NodeType,
        confidence: float = 0.5,
        parent_id: Optional[str] = None
    ) -> str:
        """Create a new node in the reasoning graph."""
        
        if len(self.nodes) >= self.config.max_nodes:
            return None
        
        node_id = str(uuid.uuid4())
        node = ReasoningNode(
            node_id=node_id,
            content=content,
            node_type=node_type,
            confidence=confidence,
            activation_level=0.3 if node_type != NodeType.ROOT else 1.0
        )
        
        self.nodes[node_id] = node
        self.graph.add_node(node_id, **self._node_to_dict(node))
        
        # Add to active nodes if activation is high enough
        if node.activation_level >= self.config.activation_threshold:
            self.active_nodes.add(node_id)
        
        return node_id
    
    def _create_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: EdgeType,
        strength: float,
        confidence: float = 0.5,
        reasoning: str = ""
    ) -> str:
        """Create an edge between two nodes."""
        
        if source_id not in self.nodes or target_id not in self.nodes:
            return None
        
        # Check if edge already exists
        existing_edge = self._find_edge(source_id, target_id, edge_type)
        if existing_edge:
            # Update existing edge
            existing_edge.strength = max(existing_edge.strength, strength)
            existing_edge.confidence = max(existing_edge.confidence, confidence)
            return existing_edge.edge_id
        
        # Check edge limits
        source_node = self.nodes[source_id]
        if len(source_node.outgoing_edges) >= self.config.max_edges_per_node:
            return None
        
        # Create new edge
        edge_id = str(uuid.uuid4())
        edge = ReasoningEdge(
            edge_id=edge_id,
            source_id=source_id,
            target_id=target_id,
            edge_type=edge_type,
            strength=strength,
            confidence=confidence,
            reasoning=reasoning
        )
        
        self.edges[edge_id] = edge
        
        # Update node connections
        source_node.outgoing_edges.append(edge_id)
        self.nodes[target_id].incoming_edges.append(edge_id)
        
        # Add to NetworkX graph
        self.graph.add_edge(
            source_id, target_id,
            edge_id=edge_id,
            edge_type=edge_type.value,
            strength=strength,
            confidence=confidence
        )
        
        self.edge_creation_count += 1
        return edge_id
    
    def _find_edge(self, source_id: str, target_id: str, edge_type: EdgeType) -> Optional[ReasoningEdge]:
        """Find existing edge between nodes of specific type."""
        
        source_node = self.nodes.get(source_id)
        if not source_node:
            return None
        
        for edge_id in source_node.outgoing_edges:
            edge = self.edges.get(edge_id)
            if (edge and edge.target_id == target_id and 
                edge.edge_type == edge_type):
                return edge
        
        return None
    
    def _select_focus_node(self) -> Optional[str]:
        """Select the next node to focus reasoning on."""
        
        if not self.active_nodes:
            return None
        
        # Score nodes based on multiple factors
        node_scores = {}
        
        for node_id in self.active_nodes:
            node = self.nodes[node_id]
            
            score = 0.0
            
            # Activation level
            score += node.activation_level * 0.3
            
            # Confidence
            score += node.confidence * 0.2
            
            # Centrality (cached or computed)
            centrality = self._get_node_centrality(node_id)
            score += centrality * 0.2
            
            # Potential for synthesis
            synthesis_potential = self._calculate_synthesis_potential(node_id)
            score += synthesis_potential * 0.2
            
            # Recency bonus (prefer recently activated nodes)
            time_factor = max(0, 1.0 - (time.time() - node.last_updated) / 60.0)
            score += time_factor * 0.1
            
            node_scores[node_id] = score
        
        # Select highest scoring node
        if node_scores:
            best_node = max(node_scores.items(), key=lambda x: x[1])
            return best_node[0]
        
        return None
    
    def _activate_neighborhood(self, focus_node_id: str):
        """Activate nodes in the neighborhood of the focus node."""
        
        focus_node = self.nodes[focus_node_id]
        focus_node.activation_level = min(1.0, focus_node.activation_level + 0.2)
        focus_node.last_updated = time.time()
        
        # Spread activation to connected nodes
        connected_nodes = self._get_connected_nodes(focus_node_id, max_distance=2)
        
        for node_id, distance in connected_nodes.items():
            if node_id != focus_node_id:
                node = self.nodes[node_id]
                
                # Activation decreases with distance
                activation_boost = 0.1 / (distance + 1)
                node.activation_level = min(1.0, node.activation_level + activation_boost)
                node.last_updated = time.time()
                
                # Add to active set if above threshold
                if node.activation_level >= self.config.activation_threshold:
                    self.active_nodes.add(node_id)
    
    def _get_connected_nodes(self, node_id: str, max_distance: int = 2) -> Dict[str, int]:
        """Get nodes connected to the given node within max_distance."""
        
        connected = {}
        visited = set()
        queue = deque([(node_id, 0)])
        
        while queue:
            current_id, distance = queue.popleft()
            
            if current_id in visited or distance > max_distance:
                continue
            
            visited.add(current_id)
            connected[current_id] = distance
            
            # Add neighbors
            if current_id in self.nodes:
                current_node = self.nodes[current_id]
                
                # Outgoing edges
                for edge_id in current_node.outgoing_edges:
                    edge = self.edges.get(edge_id)
                    if edge and edge.strength >= self.config.min_edge_strength:
                        queue.append((edge.target_id, distance + 1))
                
                # Incoming edges
                for edge_id in current_node.incoming_edges:
                    edge = self.edges.get(edge_id)
                    if edge and edge.strength >= self.config.min_edge_strength:
                        queue.append((edge.source_id, distance + 1))
        
        return connected
    
    def _generate_insights(
        self,
        focus_node_id: str,
        query: str,
        context: Optional[Dict[str, Any]]
    ) -> List[str]:
        """Generate new insights based on the focus node."""
        
        focus_node = self.nodes[focus_node_id]
        new_node_ids = []
        
        # Generate different types of insights based on node type
        if focus_node.node_type == NodeType.CONCEPT:
            # Generate evidence and elaborations
            insights = self._generate_concept_insights(focus_node, query)
        elif focus_node.node_type == NodeType.QUESTION:
            # Generate hypotheses and potential answers
            insights = self._generate_question_insights(focus_node, query)
        elif focus_node.node_type == NodeType.HYPOTHESIS:
            # Generate evidence and tests
            insights = self._generate_hypothesis_insights(focus_node, query)
        elif focus_node.node_type == NodeType.EVIDENCE:
            # Generate implications and connections
            insights = self._generate_evidence_insights(focus_node, query)
        else:
            # Default insight generation
            insights = self._generate_default_insights(focus_node, query)
        
        # Create nodes for insights
        for insight_content, insight_type in insights:
            node_id = self._create_node(
                insight_content,
                insight_type,
                confidence=0.6
            )
            
            if node_id:
                new_node_ids.append(node_id)
                
                # Create edge from focus node
                self._create_edge(
                    focus_node_id, node_id,
                    EdgeType.ELABORATES,
                    strength=0.6,
                    reasoning="Generated insight"
                )
        
        return new_node_ids
    
    def _generate_concept_insights(self, node: ReasoningNode, query: str) -> List[Tuple[str, NodeType]]:
        """Generate insights for concept nodes."""
        
        insights = []
        concept = node.content
        
        insights.extend([
            (f"Evidence supporting {concept}", NodeType.EVIDENCE),
            (f"Implications of {concept}", NodeType.HYPOTHESIS),
            (f"How does {concept} relate to the main question?", NodeType.QUESTION)
        ])
        
        return insights[:3]
    
    def _generate_question_insights(self, node: ReasoningNode, query: str) -> List[Tuple[str, NodeType]]:
        """Generate insights for question nodes."""
        
        insights = []
        question = node.content
        
        insights.extend([
            (f"Hypothesis: Potential answer to {question[:50]}...", NodeType.HYPOTHESIS),
            (f"What evidence would answer {question[:30]}...?", NodeType.QUESTION),
            (f"Alternative formulation of {question[:30]}...", NodeType.QUESTION)
        ])
        
        return insights[:3]
    
    def _generate_hypothesis_insights(self, node: ReasoningNode, query: str) -> List[Tuple[str, NodeType]]:
        """Generate insights for hypothesis nodes."""
        
        insights = []
        hypothesis = node.content
        
        insights.extend([
            (f"Evidence that would support: {hypothesis[:50]}...", NodeType.EVIDENCE),
            (f"Evidence that would contradict: {hypothesis[:50]}...", NodeType.COUNTERARGUMENT),
            (f"Implications if true: {hypothesis[:50]}...", NodeType.HYPOTHESIS)
        ])
        
        return insights[:3]
    
    def _generate_evidence_insights(self, node: ReasoningNode, query: str) -> List[Tuple[str, NodeType]]:
        """Generate insights for evidence nodes."""
        
        insights = []
        evidence = node.content
        
        insights.extend([
            (f"What this evidence implies: {evidence[:50]}...", NodeType.HYPOTHESIS),
            (f"How reliable is: {evidence[:50]}...?", NodeType.QUESTION),
            (f"Alternative interpretation: {evidence[:50]}...", NodeType.HYPOTHESIS)
        ])
        
        return insights[:3]
    
    def _generate_default_insights(self, node: ReasoningNode, query: str) -> List[Tuple[str, NodeType]]:
        """Generate default insights for any node type."""
        
        insights = []
        content = node.content
        
        insights.extend([
            (f"Further analysis needed: {content[:50]}...", NodeType.QUESTION),
            (f"Related concept: {content[:50]}...", NodeType.CONCEPT),
            (f"Potential implication: {content[:50]}...", NodeType.HYPOTHESIS)
        ])
        
        return insights[:2]
    
    def _create_automatic_edges(self, new_node_ids: List[str]):
        """Automatically create edges between semantically related nodes."""
        
        if not self.config.auto_create_edges:
            return
        
        # Compare new nodes with existing nodes
        for new_id in new_node_ids:
            new_node = self.nodes[new_id]
            
            for existing_id, existing_node in self.nodes.items():
                if existing_id == new_id or existing_id in new_node_ids:
                    continue
                
                # Calculate similarity
                similarity = self._calculate_semantic_similarity(
                    new_node.content, existing_node.content
                )
                
                if similarity >= self.config.similarity_threshold:
                    # Create appropriate edge type
                    edge_type = self._determine_edge_type(new_node, existing_node, similarity)
                    
                    if edge_type:
                        self._create_edge(
                            existing_id, new_id, edge_type,
                            strength=similarity,
                            reasoning=f"Semantic similarity: {similarity:.2f}"
                        )
        
        # Check for contradictions if enabled
        if self.config.enable_contradiction_detection:
            self._detect_contradictions(new_node_ids)
    
    def _calculate_semantic_similarity(self, content1: str, content2: str) -> float:
        """Calculate semantic similarity between two pieces of content."""
        
        # Cache check
        cache_key = (content1, content2)
        if cache_key in self.similarity_cache:
            return self.similarity_cache[cache_key]
        
        # Simple word overlap similarity (would use embeddings in practice)
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())
        
        # Remove stop words
        stop_words = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
            "for", "of", "with", "by", "from", "is", "are", "was", "were"
        }
        
        words1 -= stop_words
        words2 -= stop_words
        
        if not words1 or not words2:
            similarity = 0.0
        else:
            intersection = words1 & words2
            union = words1 | words2
            similarity = len(intersection) / len(union)
        
        # Cache result
        if len(self.similarity_cache) < self.config.cache_size:
            self.similarity_cache[cache_key] = similarity
        
        return similarity
    
    def _determine_edge_type(
        self,
        node1: ReasoningNode,
        node2: ReasoningNode,
        similarity: float
    ) -> Optional[EdgeType]:
        """Determine appropriate edge type between two similar nodes."""
        
        # Type-based edge determination
        type_combinations = {
            (NodeType.CONCEPT, NodeType.EVIDENCE): EdgeType.SUPPORTS,
            (NodeType.HYPOTHESIS, NodeType.EVIDENCE): EdgeType.SUPPORTS,
            (NodeType.QUESTION, NodeType.HYPOTHESIS): EdgeType.LEADS_TO,
            (NodeType.CONCEPT, NodeType.CONCEPT): EdgeType.SIMILAR_TO,
            (NodeType.EVIDENCE, NodeType.CONCLUSION): EdgeType.SUPPORTS,
            (NodeType.HYPOTHESIS, NodeType.CONCLUSION): EdgeType.IMPLIES,
        }
        
        # Check direct mapping
        edge_type = type_combinations.get((node1.node_type, node2.node_type))
        if edge_type:
            return edge_type
        
        # Check reverse mapping
        edge_type = type_combinations.get((node2.node_type, node1.node_type))
        if edge_type:
            return edge_type
        
        # Default based on similarity strength
        if similarity > 0.8:
            return EdgeType.SIMILAR_TO
        elif similarity > 0.7:
            return EdgeType.ELABORATES
        else:
            return EdgeType.SIMILAR_TO
    
    def _detect_contradictions(self, node_ids: List[str]):
        """Detect and mark contradictions between nodes."""
        
        contradiction_indicators = [
            "not", "no", "never", "false", "incorrect", "wrong",
            "contradicts", "opposes", "refutes", "disputes"
        ]
        
        for node_id in node_ids:
            node = self.nodes[node_id]
            node_content_lower = node.content.lower()
            
            # Check if this node contains contradiction indicators
            has_contradiction_indicators = any(
                indicator in node_content_lower
                for indicator in contradiction_indicators
            )
            
            if has_contradiction_indicators:
                # Look for nodes this might contradict
                for other_id, other_node in self.nodes.items():
                    if other_id == node_id:
                        continue
                    
                    # Check for semantic opposition
                    if self._are_contradictory(node.content, other_node.content):
                        self._create_edge(
                            node_id, other_id, EdgeType.CONTRADICTS,
                            strength=0.7,
                            reasoning="Contradiction detected"
                        )
    
    def _are_contradictory(self, content1: str, content2: str) -> bool:
        """Check if two contents are contradictory."""
        
        # Simple contradiction detection
        content1_lower = content1.lower()
        content2_lower = content2.lower()
        
        # Look for explicit negations
        if "not" in content1_lower and "not" not in content2_lower:
            # Check if they're talking about the same thing
            similarity = self._calculate_semantic_similarity(
                content1.replace("not", "").replace("no", ""),
                content2
            )
            return similarity > 0.6
        
        # Look for opposite terms
        opposite_pairs = [
            ("true", "false"), ("correct", "incorrect"), ("valid", "invalid"),
            ("supports", "contradicts"), ("proves", "disproves")
        ]
        
        for term1, term2 in opposite_pairs:
            if term1 in content1_lower and term2 in content2_lower:
                return True
            if term2 in content1_lower and term1 in content2_lower:
                return True
        
        return False
    
    def _propagate_activation(self):
        """Propagate activation through the graph based on the configured strategy."""
        
        if self.config.propagation_strategy == PropagationStrategy.BREADTH_FIRST:
            self._propagate_breadth_first()
        elif self.config.propagation_strategy == PropagationStrategy.DEPTH_FIRST:
            self._propagate_depth_first()
        elif self.config.propagation_strategy == PropagationStrategy.CENTRALITY_BASED:
            self._propagate_centrality_based()
        elif self.config.propagation_strategy == PropagationStrategy.INFLUENCE_BASED:
            self._propagate_influence_based()
        elif self.config.propagation_strategy == PropagationStrategy.RANDOM_WALK:
            self._propagate_random_walk()
        else:
            self._propagate_breadth_first()  # Default
    
    def _propagate_breadth_first(self):
        """Propagate activation using breadth-first strategy."""
        
        # Start from highly activated nodes
        queue = deque([
            node_id for node_id in self.active_nodes
            if self.nodes[node_id].activation_level > 0.5
        ])
        
        propagated = set()
        
        for _ in range(self.config.max_propagation_steps):
            if not queue:
                break
            
            current_id = queue.popleft()
            if current_id in propagated:
                continue
            
            propagated.add(current_id)
            current_node = self.nodes[current_id]
            
            # Propagate to neighbors
            for edge_id in current_node.outgoing_edges:
                edge = self.edges[edge_id]
                target_node = self.nodes[edge.target_id]
                
                # Calculate activation transfer
                transfer = current_node.activation_level * edge.strength * 0.1
                target_node.activation_level = min(1.0, target_node.activation_level + transfer)
                
                if target_node.activation_level >= self.config.activation_threshold:
                    self.active_nodes.add(edge.target_id)
                    queue.append(edge.target_id)
    
    def _propagate_influence_based(self):
        """Propagate activation based on node influence scores."""
        
        # Calculate influence scores for all nodes
        self._update_influence_scores()
        
        # Sort nodes by influence and propagate from most influential
        influential_nodes = sorted(
            self.active_nodes,
            key=lambda nid: self.nodes[nid].influence_score,
            reverse=True
        )
        
        for node_id in influential_nodes[:10]:  # Top 10 influential nodes
            self._propagate_from_node(node_id)
    
    def _propagate_from_node(self, node_id: str):
        """Propagate activation from a specific node."""
        
        source_node = self.nodes[node_id]
        
        for edge_id in source_node.outgoing_edges:
            edge = self.edges[edge_id]
            target_node = self.nodes[edge.target_id]
            
            # Calculate activation based on edge strength and type
            base_transfer = source_node.activation_level * edge.strength * 0.2
            
            # Edge type modifiers
            type_modifiers = {
                EdgeType.SUPPORTS: 1.2,
                EdgeType.IMPLIES: 1.1,
                EdgeType.LEADS_TO: 1.0,
                EdgeType.ELABORATES: 0.9,
                EdgeType.SIMILAR_TO: 0.8,
                EdgeType.CONTRADICTS: 0.3  # Reduced activation
            }
            
            modifier = type_modifiers.get(edge.edge_type, 1.0)
            transfer = base_transfer * modifier
            
            target_node.activation_level = min(1.0, target_node.activation_level + transfer)
            
            if target_node.activation_level >= self.config.activation_threshold:
                self.active_nodes.add(edge.target_id)
    
    def _attempt_synthesis(self) -> List[Dict[str, Any]]:
        """Attempt to synthesize insights from highly activated nodes."""
        
        synthesis_results = []
        
        # Find groups of highly activated, related nodes
        synthesis_candidates = [
            node_id for node_id in self.active_nodes
            if self.nodes[node_id].activation_level >= self.config.synthesis_threshold
        ]
        
        if len(synthesis_candidates) < 2:
            return synthesis_results
        
        # Group nodes by connectivity
        node_groups = self._find_connected_groups(synthesis_candidates)
        
        for group in node_groups:
            if len(group) >= 2:
                synthesis_result = self._synthesize_node_group(group)
                if synthesis_result:
                    synthesis_results.append(synthesis_result)
        
        return synthesis_results
    
    def _find_connected_groups(self, node_ids: List[str]) -> List[List[str]]:
        """Find groups of connected nodes."""
        
        groups = []
        visited = set()
        
        for node_id in node_ids:
            if node_id in visited:
                continue
            
            # Find connected component
            group = []
            queue = deque([node_id])
            
            while queue:
                current_id = queue.popleft()
                if current_id in visited:
                    continue
                
                visited.add(current_id)
                group.append(current_id)
                
                # Add connected nodes from the candidate set
                connected = self._get_connected_nodes(current_id, max_distance=1)
                for connected_id in connected:
                    if connected_id in node_ids and connected_id not in visited:
                        queue.append(connected_id)
            
            if len(group) >= 2:
                groups.append(group)
        
        return groups
    
    def _synthesize_node_group(self, node_ids: List[str]) -> Optional[Dict[str, Any]]:
        """Synthesize insights from a group of related nodes."""
        
        if len(node_ids) > self.config.max_synthesis_inputs:
            # Select most important nodes
            scored_nodes = [
                (node_id, self.nodes[node_id].activation_level * self.nodes[node_id].confidence)
                for node_id in node_ids
            ]
            scored_nodes.sort(key=lambda x: x[1], reverse=True)
            node_ids = [node_id for node_id, _ in scored_nodes[:self.config.max_synthesis_inputs]]
        
        # Collect contents
        contents = [self.nodes[node_id].content for node_id in node_ids]
        node_types = [self.nodes[node_id].node_type for node_id in node_ids]
        
        # Generate synthesis
        synthesis_content = self._generate_synthesis_content(contents, node_types)
        
        if synthesis_content:
            # Create synthesis node
            synthesis_id = self._create_node(
                synthesis_content,
                NodeType.SYNTHESIS,
                confidence=0.8
            )
            
            if synthesis_id:
                # Connect synthesis to source nodes
                for source_id in node_ids:
                    self._create_edge(
                        source_id, synthesis_id, EdgeType.SYNTHESIZES,
                        strength=0.8,
                        reasoning="Synthesis of multiple insights"
                    )
                
                self.synthesis_count += 1
                
                return {
                    "synthesis_id": synthesis_id,
                    "content": synthesis_content,
                    "source_nodes": node_ids,
                    "timestamp": time.time()
                }
        
        return None
    
    def _generate_synthesis_content(self, contents: List[str], node_types: List[NodeType]) -> str:
        """Generate synthesized content from multiple inputs."""
        
        # Simple synthesis strategy
        if len(contents) == 2:
            return f"Synthesizing: {contents[0][:50]}... and {contents[1][:50]}... suggests new insight"
        elif len(contents) == 3:
            return f"Integration of three concepts: {', '.join(c[:30] + '...' for c in contents)}"
        else:
            return f"Complex synthesis of {len(contents)} related insights"
    
    def _update_graph_metrics(self):
        """Update graph-level metrics like centrality and influence."""
        
        # Update centrality scores
        try:
            if self.config.centrality_algorithm == "pagerank":
                centrality = nx.pagerank(self.graph)
            elif self.config.centrality_algorithm == "betweenness":
                centrality = nx.betweenness_centrality(self.graph)
            elif self.config.centrality_algorithm == "closeness":
                centrality = nx.closeness_centrality(self.graph)
            else:
                centrality = nx.pagerank(self.graph)
            
            # Update node centrality scores
            for node_id, score in centrality.items():
                if node_id in self.nodes:
                    self.nodes[node_id].centrality_score = score
            
            # Cache centrality scores
            self.centrality_cache.update(centrality)
            
        except Exception as e:
            warnings.warn(f"Error updating centrality: {str(e)}")
        
        # Update influence scores
        self._update_influence_scores()
        
        # Apply activation decay
        self._apply_activation_decay()
    
    def _update_influence_scores(self):
        """Update influence scores for all nodes."""
        
        for node_id, node in self.nodes.items():
            influence = 0.0
            
            # Base influence from centrality
            influence += node.centrality_score * 0.4
            
            # Influence from outgoing connections
            outgoing_strength = sum(
                self.edges[edge_id].strength
                for edge_id in node.outgoing_edges
                if edge_id in self.edges
            )
            influence += min(outgoing_strength, 1.0) * 0.3
            
            # Influence from confidence and activation
            influence += node.confidence * node.activation_level * 0.3
            
            node.influence_score = influence
    
    def _apply_activation_decay(self):
        """Apply decay to node activations."""
        
        nodes_to_deactivate = []
        
        for node_id in list(self.active_nodes):
            node = self.nodes[node_id]
            
            # Apply decay
            node.activation_level = max(0.0, node.activation_level - self.config.decay_rate)
            
            # Remove from active set if below threshold
            if node.activation_level < self.config.activation_threshold:
                nodes_to_deactivate.append(node_id)
        
        for node_id in nodes_to_deactivate:
            self.active_nodes.discard(node_id)
    
    def _check_convergence(self) -> bool:
        """Check if reasoning has converged."""
        
        # Check if we have high-quality synthesis or conclusion nodes
        conclusion_nodes = [
            node for node in self.nodes.values()
            if node.node_type in [NodeType.CONCLUSION, NodeType.SYNTHESIS] and
            node.confidence > 0.8 and node.activation_level > 0.5
        ]
        
        return len(conclusion_nodes) >= 2
    
    def _cleanup_graph(self):
        """Perform periodic cleanup of the graph."""
        
        # Remove low-quality nodes
        nodes_to_remove = [
            node_id for node_id, node in self.nodes.items()
            if (node.confidence < 0.2 and node.activation_level < 0.1 and
                node.node_type != NodeType.ROOT)
        ]
        
        for node_id in nodes_to_remove:
            self._remove_node(node_id)
        
        # Clean cache if it's too large
        if len(self.similarity_cache) > self.config.cache_size:
            # Keep most recent entries
            items = list(self.similarity_cache.items())
            self.similarity_cache = dict(items[-self.config.cache_size//2:])
    
    def _remove_node(self, node_id: str):
        """Remove a node and its connections from the graph."""
        
        if node_id not in self.nodes:
            return
        
        node = self.nodes[node_id]
        
        # Remove edges
        edges_to_remove = node.incoming_edges + node.outgoing_edges
        for edge_id in edges_to_remove:
            if edge_id in self.edges:
                edge = self.edges[edge_id]
                
                # Update connected nodes
                if edge.source_id in self.nodes:
                    self.nodes[edge.source_id].outgoing_edges.remove(edge_id)
                if edge.target_id in self.nodes:
                    self.nodes[edge.target_id].incoming_edges.remove(edge_id)
                
                # Remove edge
                del self.edges[edge_id]
        
        # Remove node
        del self.nodes[node_id]
        self.active_nodes.discard(node_id)
        
        # Remove from NetworkX graph
        if self.graph.has_node(node_id):
            self.graph.remove_node(node_id)
    
    def _generate_final_answer(self, query: str) -> Tuple[str, float]:
        """Generate final answer from the reasoning graph."""
        
        # Find best conclusion or synthesis nodes
        final_candidates = [
            node for node in self.nodes.values()
            if node.node_type in [NodeType.CONCLUSION, NodeType.SYNTHESIS] and
            not node.node_id == self.root_id
        ]
        
        if final_candidates:
            # Select best candidate
            best_candidate = max(
                final_candidates,
                key=lambda n: n.confidence * n.activation_level * n.centrality_score
            )
            
            # Generate comprehensive answer
            answer = self._create_comprehensive_answer(best_candidate, query)
            confidence = best_candidate.confidence
            
        else:
            # Fallback: synthesize from highly activated nodes
            high_activation_nodes = [
                node for node in self.nodes.values()
                if node.activation_level > 0.6 and node.node_type != NodeType.ROOT
            ]
            
            if high_activation_nodes:
                best_nodes = sorted(
                    high_activation_nodes,
                    key=lambda n: n.activation_level * n.confidence,
                    reverse=True
                )[:3]
                
                answer = self._synthesize_answer_from_nodes(best_nodes, query)
                confidence = sum(n.confidence for n in best_nodes) / len(best_nodes)
            else:
                answer = "Unable to generate a conclusive answer from graph reasoning."
                confidence = 0.1
        
        return answer, confidence
    
    def _create_comprehensive_answer(self, conclusion_node: ReasoningNode, query: str) -> str:
        """Create comprehensive answer from conclusion node and supporting evidence."""
        
        # Get supporting path
        supporting_nodes = self._get_supporting_evidence(conclusion_node.node_id)
        
        answer_parts = [conclusion_node.content]
        
        if supporting_nodes:
            answer_parts.append("\nSupporting reasoning:")
            for i, node in enumerate(supporting_nodes[:3], 1):
                answer_parts.append(f"{i}. {node.content}")
        
        return "\n".join(answer_parts)
    
    def _synthesize_answer_from_nodes(self, nodes: List[ReasoningNode], query: str) -> str:
        """Synthesize answer from multiple high-quality nodes."""
        
        key_insights = [node.content for node in nodes]
        
        answer_parts = [
            "Based on graph analysis of interconnected reasoning:",
            *[f"" {insight}" for insight in key_insights]
        ]
        
        return "\n".join(answer_parts)
    
    def _get_supporting_evidence(self, conclusion_id: str) -> List[ReasoningNode]:
        """Get nodes that provide supporting evidence for a conclusion."""
        
        supporting_nodes = []
        
        conclusion_node = self.nodes[conclusion_id]
        
        # Follow incoming edges to find support
        for edge_id in conclusion_node.incoming_edges:
            edge = self.edges[edge_id]
            
            if edge.edge_type in [EdgeType.SUPPORTS, EdgeType.IMPLIES, EdgeType.LEADS_TO]:
                source_node = self.nodes[edge.source_id]
                if source_node.node_type in [NodeType.EVIDENCE, NodeType.HYPOTHESIS]:
                    supporting_nodes.append(source_node)
        
        # Sort by strength and confidence
        supporting_nodes.sort(
            key=lambda n: n.confidence * n.activation_level,
            reverse=True
        )
        
        return supporting_nodes
    
    def _export_reasoning_graph(self) -> Dict[str, Any]:
        """Export reasoning graph for analysis and visualization."""
        
        # Node information
        nodes_data = []
        for node in self.nodes.values():
            nodes_data.append({
                "id": node.node_id,
                "content": node.content,
                "type": node.node_type.value,
                "confidence": node.confidence,
                "activation": node.activation_level,
                "centrality": node.centrality_score,
                "influence": node.influence_score
            })
        
        # Edge information
        edges_data = []
        for edge in self.edges.values():
            edges_data.append({
                "id": edge.edge_id,
                "source": edge.source_id,
                "target": edge.target_id,
                "type": edge.edge_type.value,
                "strength": edge.strength,
                "confidence": edge.confidence
            })
        
        return {
            "nodes": nodes_data,
            "edges": edges_data,
            "statistics": self.get_graph_statistics(),
            "synthesis_history": self.synthesis_history,
            "active_nodes": list(self.active_nodes)
        }
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """Get comprehensive graph statistics."""
        
        return {
            "total_nodes": len(self.nodes),
            "total_edges": len(self.edges),
            "active_nodes": len(self.active_nodes),
            "reasoning_steps": self.reasoning_steps,
            "synthesis_count": self.synthesis_count,
            "edge_creation_count": self.edge_creation_count,
            "node_type_distribution": {
                node_type.value: sum(1 for n in self.nodes.values() if n.node_type == node_type)
                for node_type in NodeType
            },
            "edge_type_distribution": {
                edge_type.value: sum(1 for e in self.edges.values() if e.edge_type == edge_type)
                for edge_type in EdgeType
            },
            "avg_node_confidence": sum(n.confidence for n in self.nodes.values()) / len(self.nodes) if self.nodes else 0,
            "avg_edge_strength": sum(e.strength for e in self.edges.values()) / len(self.edges) if self.edges else 0,
            "graph_density": nx.density(self.graph) if self.graph.number_of_nodes() > 1 else 0
        }
    
    def _get_node_centrality(self, node_id: str) -> float:
        """Get cached or compute node centrality."""
        
        if node_id in self.centrality_cache:
            return self.centrality_cache[node_id]
        
        # Compute if not cached
        try:
            centrality = nx.pagerank(self.graph)
            self.centrality_cache.update(centrality)
            return centrality.get(node_id, 0.0)
        except:
            return 0.5  # Default value
    
    def _calculate_synthesis_potential(self, node_id: str) -> float:
        """Calculate potential for synthesis from this node."""
        
        node = self.nodes[node_id]
        
        # Base potential from node properties
        potential = node.confidence * 0.4 + node.activation_level * 0.3
        
        # Bonus for nodes with multiple high-quality connections
        high_quality_connections = sum(
            1 for edge_id in node.outgoing_edges + node.incoming_edges
            if edge_id in self.edges and self.edges[edge_id].strength > 0.6
        )
        
        potential += min(high_quality_connections / 5.0, 0.3)
        
        return potential
    
    def _node_to_dict(self, node: ReasoningNode) -> Dict[str, Any]:
        """Convert node to dictionary for NetworkX."""
        
        return {
            "content": node.content,
            "node_type": node.node_type.value,
            "confidence": node.confidence,
            "activation": node.activation_level,
            "centrality": node.centrality_score,
            "influence": node.influence_score
        }