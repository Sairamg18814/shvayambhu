"""Tree-of-Thought reasoning strategy implementation.

This module implements the Tree-of-Thought (ToT) reasoning approach that explores
multiple reasoning paths simultaneously, maintains a tree structure of thoughts,
and uses systematic evaluation to select the best paths forward.
"""

import torch
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import uuid
import time
import heapq
import numpy as np
from collections import defaultdict, deque

from .reasoning_engine import ReasoningStep, ReasoningTrace, StepType, ReasoningStrategy
from .reward_model import ProcessReward, RewardType


class NodeType(Enum):
    """Types of nodes in the reasoning tree."""
    ROOT = "root"
    THOUGHT = "thought"
    EVALUATION = "evaluation"
    SYNTHESIS = "synthesis"
    CONCLUSION = "conclusion"
    PRUNED = "pruned"


class SelectionStrategy(Enum):
    """Strategies for selecting promising nodes to expand."""
    BEST_FIRST = "best_first"
    BREADTH_FIRST = "breadth_first"
    DEPTH_FIRST = "depth_first"
    UCB = "upper_confidence_bound"
    THOMPSON_SAMPLING = "thompson_sampling"
    MIXED = "mixed"


@dataclass
class ThoughtNode:
    """A node in the Tree-of-Thought structure."""
    node_id: str
    content: str
    node_type: NodeType
    depth: int
    confidence: float
    
    # Tree structure
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    
    # Evaluation metrics
    value_estimate: float = 0.0
    visit_count: int = 0
    reward_sum: float = 0.0
    
    # State tracking
    is_expanded: bool = False
    is_pruned: bool = False
    expansion_priority: float = 0.0
    
    # Metadata
    created_at: float = field(default_factory=time.time)
    reasoning_step_id: Optional[str] = None
    domain_specific_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToTConfig:
    """Configuration for Tree-of-Thought reasoning."""
    # Tree structure
    max_depth: int = 20
    max_children_per_node: int = 4
    max_total_nodes: int = 1000
    
    # Selection and expansion
    selection_strategy: SelectionStrategy = SelectionStrategy.UCB
    exploration_constant: float = 1.4  # UCB exploration parameter
    expansion_threshold: float = 0.3   # Minimum confidence to expand
    
    # Evaluation
    value_function_weight: float = 0.4
    reward_weight: float = 0.3
    confidence_weight: float = 0.2
    novelty_weight: float = 0.1
    
    # Pruning
    enable_pruning: bool = True
    pruning_threshold: float = 0.2
    min_visits_before_pruning: int = 3
    aggressive_pruning: bool = False
    
    # Search control
    beam_width: int = 5  # Number of best paths to maintain
    rollout_depth: int = 3  # Depth for evaluation rollouts
    enable_backpropagation: bool = True
    
    # Domain adaptation
    domain_specific_expansion: bool = True
    semantic_similarity_threshold: float = 0.7


class TreeOfThoughtReasoner:
    """Implements Tree-of-Thought reasoning strategy."""
    
    def __init__(self, config: ToTConfig):
        self.config = config
        
        # Tree structure
        self.nodes: Dict[str, ThoughtNode] = {}
        self.root_id: Optional[str] = None
        
        # Search state
        self.frontier: List[Tuple[float, str]] = []  # Priority queue of (priority, node_id)
        self.best_paths: List[List[str]] = []  # Best reasoning paths
        self.pruned_nodes: Set[str] = set()
        
        # Statistics
        self.expansion_count = 0
        self.evaluation_count = 0
        self.pruning_count = 0
        
        # Cache for efficiency
        self.path_cache: Dict[str, List[str]] = {}
        self.value_cache: Dict[str, float] = {}
    
    def reason(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        max_iterations: int = 100
    ) -> Tuple[str, float, List[ThoughtNode]]:
        """Execute Tree-of-Thought reasoning."""
        
        # Initialize tree with root node
        self.root_id = self._create_root_node(query)
        
        # Initialize frontier
        heapq.heappush(self.frontier, (-1.0, self.root_id))  # Negative for max-heap behavior
        
        iteration = 0
        best_solution = None
        best_confidence = 0.0
        
        while self.frontier and iteration < max_iterations:
            iteration += 1
            
            # Select next node to expand
            selected_node_id = self._select_node()
            if not selected_node_id:
                break
            
            selected_node = self.nodes[selected_node_id]
            
            # Check if we should expand this node
            if self._should_expand(selected_node):
                # Expand node
                children = self._expand_node(selected_node, query, context)
                
                # Evaluate children
                for child in children:
                    self._evaluate_node(child, query, context)
                    
                    # Check if this is a potential solution
                    if child.node_type == NodeType.CONCLUSION:
                        if child.confidence > best_confidence:
                            best_solution = child
                            best_confidence = child.confidence
                
                # Update frontier with new children
                self._update_frontier(children)
                
                # Backpropagate values if enabled
                if self.config.enable_backpropagation:
                    self._backpropagate_values(selected_node_id)
            
            # Prune unpromising branches
            if self.config.enable_pruning:
                self._prune_branches()
            
            # Update best paths
            self._update_best_paths()
        
        # Generate final answer
        if best_solution:
            final_answer = self._generate_final_answer(best_solution)
            reasoning_path = self._get_reasoning_path(best_solution.node_id)
        else:
            # Fallback to best available node
            best_nodes = sorted(
                [node for node in self.nodes.values() if not node.is_pruned],
                key=lambda x: x.confidence,
                reverse=True
            )
            
            if best_nodes:
                best_node = best_nodes[0]
                final_answer = f"Based on analysis: {best_node.content}"
                reasoning_path = self._get_reasoning_path(best_node.node_id)
                best_confidence = best_node.confidence
            else:
                final_answer = "Unable to generate conclusion from tree reasoning."
                reasoning_path = []
                best_confidence = 0.0
        
        return final_answer, best_confidence, reasoning_path
    
    def _create_root_node(self, query: str) -> str:
        """Create the root node of the reasoning tree."""
        
        node_id = str(uuid.uuid4())
        root_node = ThoughtNode(
            node_id=node_id,
            content=f"Analyzing query: {query}",
            node_type=NodeType.ROOT,
            depth=0,
            confidence=1.0,
            value_estimate=0.0
        )
        
        self.nodes[node_id] = root_node
        return node_id
    
    def _select_node(self) -> Optional[str]:
        """Select the most promising node for expansion."""
        
        if not self.frontier:
            return None
        
        if self.config.selection_strategy == SelectionStrategy.BEST_FIRST:
            return self._select_best_first()
        elif self.config.selection_strategy == SelectionStrategy.UCB:
            return self._select_ucb()
        elif self.config.selection_strategy == SelectionStrategy.BREADTH_FIRST:
            return self._select_breadth_first()
        elif self.config.selection_strategy == SelectionStrategy.DEPTH_FIRST:
            return self._select_depth_first()
        elif self.config.selection_strategy == SelectionStrategy.THOMPSON_SAMPLING:
            return self._select_thompson_sampling()
        else:  # MIXED
            return self._select_mixed()
    
    def _select_best_first(self) -> Optional[str]:
        """Select node with highest priority."""
        
        while self.frontier:
            _, node_id = heapq.heappop(self.frontier)
            
            if node_id in self.nodes and not self.nodes[node_id].is_pruned:
                return node_id
        
        return None
    
    def _select_ucb(self) -> Optional[str]:
        """Select using Upper Confidence Bound."""
        
        if not self.frontier:
            return None
        
        total_visits = sum(node.visit_count for node in self.nodes.values())
        best_node_id = None
        best_ucb = float('-inf')
        
        # Calculate UCB for all frontier nodes
        frontier_nodes = []
        while self.frontier:
            _, node_id = heapq.heappop(self.frontier)
            if node_id in self.nodes and not self.nodes[node_id].is_pruned:
                frontier_nodes.append(node_id)
        
        # Restore frontier
        for node_id in frontier_nodes:
            node = self.nodes[node_id]
            priority = self._calculate_node_priority(node)
            heapq.heappush(self.frontier, (-priority, node_id))
        
        # Select best UCB
        for node_id in frontier_nodes:
            node = self.nodes[node_id]
            
            if node.visit_count == 0:
                ucb_value = float('inf')
            else:
                avg_reward = node.reward_sum / node.visit_count
                exploration_term = self.config.exploration_constant * np.sqrt(
                    np.log(total_visits) / node.visit_count
                )
                ucb_value = avg_reward + exploration_term
            
            if ucb_value > best_ucb:
                best_ucb = ucb_value
                best_node_id = node_id
        
        return best_node_id
    
    def _select_breadth_first(self) -> Optional[str]:
        """Select shallowest unexpanded node."""
        
        # Find all frontier nodes and sort by depth
        frontier_nodes = []
        while self.frontier:
            _, node_id = heapq.heappop(self.frontier)
            if node_id in self.nodes and not self.nodes[node_id].is_pruned:
                frontier_nodes.append(node_id)
        
        if not frontier_nodes:
            return None
        
        # Sort by depth (shallowest first)
        frontier_nodes.sort(key=lambda nid: self.nodes[nid].depth)
        selected = frontier_nodes[0]
        
        # Restore frontier
        for node_id in frontier_nodes:
            if node_id != selected:
                node = self.nodes[node_id]
                priority = self._calculate_node_priority(node)
                heapq.heappush(self.frontier, (-priority, node_id))
        
        return selected
    
    def _select_depth_first(self) -> Optional[str]:
        """Select deepest unexpanded node."""
        
        # Similar to breadth-first but select deepest
        frontier_nodes = []
        while self.frontier:
            _, node_id = heapq.heappop(self.frontier)
            if node_id in self.nodes and not self.nodes[node_id].is_pruned:
                frontier_nodes.append(node_id)
        
        if not frontier_nodes:
            return None
        
        # Sort by depth (deepest first)
        frontier_nodes.sort(key=lambda nid: self.nodes[nid].depth, reverse=True)
        selected = frontier_nodes[0]
        
        # Restore frontier
        for node_id in frontier_nodes:
            if node_id != selected:
                node = self.nodes[node_id]
                priority = self._calculate_node_priority(node)
                heapq.heappush(self.frontier, (-priority, node_id))
        
        return selected
    
    def _select_thompson_sampling(self) -> Optional[str]:
        """Select using Thompson sampling."""
        
        frontier_nodes = []
        while self.frontier:
            _, node_id = heapq.heappop(self.frontier)
            if node_id in self.nodes and not self.nodes[node_id].is_pruned:
                frontier_nodes.append(node_id)
        
        if not frontier_nodes:
            return None
        
        # Sample from beta distributions
        best_sample = float('-inf')
        best_node_id = None
        
        for node_id in frontier_nodes:
            node = self.nodes[node_id]
            
            # Beta distribution parameters
            alpha = max(1, node.reward_sum + 1)
            beta = max(1, node.visit_count - node.reward_sum + 1)
            
            # Sample from beta distribution
            sample = np.random.beta(alpha, beta)
            
            if sample > best_sample:
                best_sample = sample
                best_node_id = node_id
        
        # Restore frontier
        for node_id in frontier_nodes:
            if node_id != best_node_id:
                node = self.nodes[node_id]
                priority = self._calculate_node_priority(node)
                heapq.heappush(self.frontier, (-priority, node_id))
        
        return best_node_id
    
    def _select_mixed(self) -> Optional[str]:
        """Mixed selection strategy combining multiple approaches."""
        
        # Randomly choose between different strategies
        strategies = [
            SelectionStrategy.BEST_FIRST,
            SelectionStrategy.UCB,
            SelectionStrategy.BREADTH_FIRST
        ]
        
        weights = [0.4, 0.4, 0.2]  # Favor best-first and UCB
        chosen_strategy = np.random.choice(strategies, p=weights)
        
        # Temporarily change strategy and select
        original_strategy = self.config.selection_strategy
        self.config.selection_strategy = chosen_strategy
        
        if chosen_strategy == SelectionStrategy.BEST_FIRST:
            selected = self._select_best_first()
        elif chosen_strategy == SelectionStrategy.UCB:
            selected = self._select_ucb()
        else:  # BREADTH_FIRST
            selected = self._select_breadth_first()
        
        # Restore original strategy
        self.config.selection_strategy = original_strategy
        
        return selected
    
    def _should_expand(self, node: ThoughtNode) -> bool:
        """Determine if a node should be expanded."""
        
        # Don't expand already expanded nodes
        if node.is_expanded:
            return False
        
        # Don't expand pruned nodes
        if node.is_pruned:
            return False
        
        # Don't expand beyond max depth
        if node.depth >= self.config.max_depth:
            return False
        
        # Don't expand if we've reached node limit
        if len(self.nodes) >= self.config.max_total_nodes:
            return False
        
        # Check confidence threshold
        if node.confidence < self.config.expansion_threshold:
            return False
        
        # Don't expand conclusion nodes
        if node.node_type == NodeType.CONCLUSION:
            return False
        
        return True
    
    def _expand_node(
        self,
        node: ThoughtNode,
        query: str,
        context: Optional[Dict[str, Any]]
    ) -> List[ThoughtNode]:
        """Expand a node by generating child thoughts."""
        
        children = []
        max_children = min(self.config.max_children_per_node, 
                          self.config.max_total_nodes - len(self.nodes))
        
        if max_children <= 0:
            return children
        
        # Generate child thoughts based on node type and content
        child_thoughts = self._generate_child_thoughts(node, query, context, max_children)
        
        for i, thought_content in enumerate(child_thoughts):
            child_id = str(uuid.uuid4())
            
            # Determine child node type
            child_type = self._determine_node_type(thought_content, node.depth + 1)
            
            # Create child node
            child = ThoughtNode(
                node_id=child_id,
                content=thought_content,
                node_type=child_type,
                depth=node.depth + 1,
                confidence=0.5,  # Will be updated by evaluation
                parent_id=node.node_id
            )
            
            # Add to tree structure
            self.nodes[child_id] = child
            node.children_ids.append(child_id)
            children.append(child)
        
        # Mark node as expanded
        node.is_expanded = True
        self.expansion_count += 1
        
        return children
    
    def _generate_child_thoughts(
        self,
        node: ThoughtNode,
        query: str,
        context: Optional[Dict[str, Any]],
        max_children: int
    ) -> List[str]:
        """Generate child thought contents."""
        
        thoughts = []
        
        if node.node_type == NodeType.ROOT:
            # Generate initial analysis directions
            thoughts = [
                "Break down the problem into components",
                "Consider different approaches to solve this",
                "Analyze the context and constraints",
                "Identify key concepts and relationships"
            ]
        
        elif node.node_type == NodeType.THOUGHT:
            # Generate follow-up thoughts
            if "analyze" in node.content.lower():
                thoughts = [
                    "Deep dive into the first component",
                    "Examine relationships between components",
                    "Consider alternative interpretations"
                ]
            elif "approach" in node.content.lower():
                thoughts = [
                    "Method 1: Systematic step-by-step analysis",
                    "Method 2: Comparative evaluation",
                    "Method 3: Synthesis of multiple perspectives"
                ]
            elif "component" in node.content.lower():
                thoughts = [
                    "Examine the most critical component first",
                    "Look for dependencies between components",
                    "Assess the impact of each component"
                ]
            else:
                # Generic follow-up thoughts
                thoughts = [
                    "Elaborate on this idea further",
                    "Consider counterarguments or alternatives",
                    "Synthesize with previous insights"
                ]
        
        elif node.node_type == NodeType.EVALUATION:
            # Generate synthesis or conclusion thoughts
            thoughts = [
                "Synthesize findings into coherent answer",
                "Draw final conclusions",
                "Validate reasoning consistency"
            ]
        
        # Limit to requested number
        return thoughts[:max_children]
    
    def _determine_node_type(self, content: str, depth: int) -> NodeType:
        """Determine the type of a node based on its content and depth."""
        
        content_lower = content.lower()
        
        # Check for conclusion indicators
        if any(word in content_lower for word in ["conclude", "final", "answer", "solution"]):
            return NodeType.CONCLUSION
        
        # Check for evaluation indicators
        if any(word in content_lower for word in ["evaluate", "assess", "validate", "check"]):
            return NodeType.EVALUATION
        
        # Check for synthesis indicators
        if any(word in content_lower for word in ["synthesize", "combine", "integrate", "merge"]):
            return NodeType.SYNTHESIS
        
        # Default to thought
        return NodeType.THOUGHT
    
    def _evaluate_node(
        self,
        node: ThoughtNode,
        query: str,
        context: Optional[Dict[str, Any]]
    ) -> float:
        """Evaluate a node's value and update its metrics."""
        
        # Calculate value based on multiple factors
        value_components = {}
        
        # Content quality (simplified heuristic)
        content_quality = self._evaluate_content_quality(node.content)
        value_components["content_quality"] = content_quality
        
        # Relevance to query
        relevance = self._evaluate_relevance(node.content, query)
        value_components["relevance"] = relevance
        
        # Depth bonus/penalty
        depth_factor = 1.0 - (node.depth * 0.05)  # Slight penalty for depth
        value_components["depth_factor"] = depth_factor
        
        # Novelty (compared to sibling nodes)
        novelty = self._evaluate_novelty(node)
        value_components["novelty"] = novelty
        
        # Potential for leading to solution
        solution_potential = self._evaluate_solution_potential(node, query)
        value_components["solution_potential"] = solution_potential
        
        # Combine components
        total_value = (
            content_quality * self.config.value_function_weight +
            relevance * 0.3 +
            depth_factor * 0.1 +
            novelty * self.config.novelty_weight +
            solution_potential * 0.2
        )
        
        # Update node metrics
        node.value_estimate = total_value
        node.confidence = min(1.0, max(0.0, total_value))
        node.visit_count += 1
        node.reward_sum += total_value
        
        # Store evaluation metadata
        node.domain_specific_data["evaluation"] = {
            "components": value_components,
            "total_value": total_value,
            "evaluated_at": time.time()
        }
        
        self.evaluation_count += 1
        
        return total_value
    
    def _evaluate_content_quality(self, content: str) -> float:
        """Evaluate the quality of thought content."""
        
        # Simple heuristics for content quality
        quality_score = 0.5  # Base score
        
        # Length considerations
        if 20 <= len(content) <= 200:
            quality_score += 0.2
        elif len(content) < 10:
            quality_score -= 0.3
        
        # Presence of reasoning indicators
        reasoning_words = [
            "because", "therefore", "since", "due to", "as a result",
            "consequently", "thus", "hence", "implies", "suggests"
        ]
        
        content_lower = content.lower()
        reasoning_count = sum(1 for word in reasoning_words if word in content_lower)
        quality_score += min(0.3, reasoning_count * 0.1)
        
        # Specificity indicators
        specific_words = [
            "specifically", "particularly", "exactly", "precisely",
            "detailed", "comprehensive", "thorough"
        ]
        
        specificity_count = sum(1 for word in specific_words if word in content_lower)
        quality_score += min(0.2, specificity_count * 0.1)
        
        return min(1.0, max(0.0, quality_score))
    
    def _evaluate_relevance(self, content: str, query: str) -> float:
        """Evaluate relevance of content to the original query."""
        
        # Simple word overlap metric
        content_words = set(content.lower().split())
        query_words = set(query.lower().split())
        
        # Remove common stop words
        stop_words = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
            "for", "of", "with", "by", "from", "is", "are", "was", "were"
        }
        
        content_words -= stop_words
        query_words -= stop_words
        
        if not query_words:
            return 0.5
        
        # Calculate Jaccard similarity
        intersection = content_words & query_words
        union = content_words | query_words
        
        jaccard = len(intersection) / len(union) if union else 0.0
        
        return min(1.0, jaccard * 2.0)  # Scale up for more sensitivity
    
    def _evaluate_novelty(self, node: ThoughtNode) -> float:
        """Evaluate novelty compared to sibling nodes."""
        
        if not node.parent_id:
            return 1.0  # Root node is inherently novel
        
        parent = self.nodes.get(node.parent_id)
        if not parent:
            return 1.0
        
        # Compare with sibling nodes
        siblings = [self.nodes[child_id] for child_id in parent.children_ids 
                   if child_id != node.node_id and child_id in self.nodes]
        
        if not siblings:
            return 1.0
        
        # Calculate similarity with siblings
        max_similarity = 0.0
        
        for sibling in siblings:
            similarity = self._calculate_content_similarity(node.content, sibling.content)
            max_similarity = max(max_similarity, similarity)
        
        # Novelty is inverse of maximum similarity
        novelty = 1.0 - max_similarity
        
        return novelty
    
    def _calculate_content_similarity(self, content1: str, content2: str) -> float:
        """Calculate similarity between two content strings."""
        
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union) if union else 0.0
    
    def _evaluate_solution_potential(self, node: ThoughtNode, query: str) -> float:
        """Evaluate potential for node to lead to a solution."""
        
        potential = 0.5  # Base potential
        
        # Node type considerations
        if node.node_type == NodeType.CONCLUSION:
            potential += 0.4
        elif node.node_type == NodeType.SYNTHESIS:
            potential += 0.3
        elif node.node_type == NodeType.EVALUATION:
            potential += 0.2
        
        # Content analysis
        content_lower = node.content.lower()
        
        # Solution indicators
        solution_words = [
            "solution", "answer", "resolve", "conclude", "determine",
            "result", "outcome", "final", "complete"
        ]
        
        solution_count = sum(1 for word in solution_words if word in content_lower)
        potential += min(0.3, solution_count * 0.1)
        
        # Confidence in path (based on ancestor quality)
        path_quality = self._calculate_path_quality(node.node_id)
        potential += path_quality * 0.2
        
        return min(1.0, max(0.0, potential))
    
    def _calculate_path_quality(self, node_id: str) -> float:
        """Calculate quality of path from root to node."""
        
        path = self._get_reasoning_path(node_id)
        if not path:
            return 0.5
        
        total_quality = 0.0
        for node in path:
            total_quality += node.confidence
        
        return total_quality / len(path)
    
    def _update_frontier(self, new_nodes: List[ThoughtNode]):
        """Update frontier with new nodes."""
        
        for node in new_nodes:
            if not node.is_pruned and node.depth < self.config.max_depth:
                priority = self._calculate_node_priority(node)
                heapq.heappush(self.frontier, (-priority, node.node_id))
    
    def _calculate_node_priority(self, node: ThoughtNode) -> float:
        """Calculate priority for frontier ordering."""
        
        # Combine value estimate with other factors
        priority = node.value_estimate
        
        # Add confidence bonus
        priority += node.confidence * 0.2
        
        # Add type bonus
        type_bonuses = {
            NodeType.CONCLUSION: 0.3,
            NodeType.SYNTHESIS: 0.2,
            NodeType.EVALUATION: 0.1,
            NodeType.THOUGHT: 0.0,
            NodeType.ROOT: 0.0
        }
        
        priority += type_bonuses.get(node.node_type, 0.0)
        
        # Depth consideration (slight preference for deeper nodes)
        priority += node.depth * 0.01
        
        return priority
    
    def _backpropagate_values(self, node_id: str):
        """Backpropagate values up the tree."""
        
        current_id = node_id
        
        while current_id:
            current_node = self.nodes[current_id]
            
            # Update value based on children
            if current_node.children_ids:
                children_values = [
                    self.nodes[child_id].value_estimate
                    for child_id in current_node.children_ids
                    if child_id in self.nodes and not self.nodes[child_id].is_pruned
                ]
                
                if children_values:
                    # Use max child value with some averaging
                    max_child_value = max(children_values)
                    avg_child_value = sum(children_values) / len(children_values)
                    
                    # Combine current value with child information
                    updated_value = (
                        current_node.value_estimate * 0.3 +
                        max_child_value * 0.5 +
                        avg_child_value * 0.2
                    )
                    
                    current_node.value_estimate = min(1.0, updated_value)
                    current_node.confidence = min(1.0, updated_value)
            
            # Move to parent
            current_id = current_node.parent_id
    
    def _prune_branches(self):
        """Prune unpromising branches from the tree."""
        
        if not self.config.enable_pruning:
            return
        
        nodes_to_prune = []
        
        for node in self.nodes.values():
            if (not node.is_pruned and 
                node.visit_count >= self.config.min_visits_before_pruning and
                node.node_type != NodeType.ROOT):
                
                # Check if node meets pruning criteria
                should_prune = False
                
                # Low value pruning
                if node.value_estimate < self.config.pruning_threshold:
                    should_prune = True
                
                # Aggressive pruning for nodes with low confidence and high visits
                if (self.config.aggressive_pruning and 
                    node.confidence < 0.3 and 
                    node.visit_count > 5):
                    should_prune = True
                
                # Prune if all children are pruned
                if (node.children_ids and 
                    all(self.nodes[child_id].is_pruned for child_id in node.children_ids 
                        if child_id in self.nodes)):
                    should_prune = True
                
                if should_prune:
                    nodes_to_prune.append(node.node_id)
        
        # Perform pruning
        for node_id in nodes_to_prune:
            self._prune_subtree(node_id)
    
    def _prune_subtree(self, node_id: str):
        """Recursively prune a subtree."""
        
        if node_id not in self.nodes:
            return
        
        node = self.nodes[node_id]
        node.is_pruned = True
        node.node_type = NodeType.PRUNED
        self.pruned_nodes.add(node_id)
        self.pruning_count += 1
        
        # Recursively prune children
        for child_id in node.children_ids:
            if child_id in self.nodes:
                self._prune_subtree(child_id)
    
    def _update_best_paths(self):
        """Update the list of best reasoning paths."""
        
        # Find all conclusion nodes
        conclusion_nodes = [
            node for node in self.nodes.values()
            if node.node_type == NodeType.CONCLUSION and not node.is_pruned
        ]
        
        # Sort by confidence
        conclusion_nodes.sort(key=lambda x: x.confidence, reverse=True)
        
        # Update best paths
        self.best_paths = []
        for node in conclusion_nodes[:self.config.beam_width]:
            path = self._get_node_path(node.node_id)
            if path:
                self.best_paths.append(path)
    
    def _get_node_path(self, node_id: str) -> List[str]:
        """Get path of node IDs from root to specified node."""
        
        if node_id in self.path_cache:
            return self.path_cache[node_id]
        
        path = []
        current_id = node_id
        
        while current_id:
            path.append(current_id)
            current_node = self.nodes.get(current_id)
            if not current_node:
                break
            current_id = current_node.parent_id
        
        path.reverse()
        self.path_cache[node_id] = path
        
        return path
    
    def _get_reasoning_path(self, node_id: str) -> List[ThoughtNode]:
        """Get reasoning path as list of nodes."""
        
        node_ids = self._get_node_path(node_id)
        return [self.nodes[nid] for nid in node_ids if nid in self.nodes]
    
    def _generate_final_answer(self, conclusion_node: ThoughtNode) -> str:
        """Generate final answer from conclusion node."""
        
        # Get full reasoning path
        reasoning_path = self._get_reasoning_path(conclusion_node.node_id)
        
        # Synthesize insights from the path
        key_insights = []
        for node in reasoning_path:
            if node.node_type in [NodeType.THOUGHT, NodeType.SYNTHESIS, NodeType.EVALUATION]:
                key_insights.append(node.content)
        
        # Create comprehensive answer
        if key_insights:
            answer_parts = [
                f"Based on systematic analysis: {conclusion_node.content}",
                "\nKey reasoning steps:",
                *[f"" {insight}" for insight in key_insights[-3:]]  # Last 3 insights
            ]
            
            final_answer = "\n".join(answer_parts)
        else:
            final_answer = conclusion_node.content
        
        return final_answer
    
    def get_tree_statistics(self) -> Dict[str, Any]:
        """Get comprehensive tree statistics."""
        
        total_nodes = len(self.nodes)
        pruned_count = len(self.pruned_nodes)
        active_nodes = total_nodes - pruned_count
        
        # Node type distribution
        type_counts = defaultdict(int)
        for node in self.nodes.values():
            if not node.is_pruned:
                type_counts[node.node_type.value] += 1
        
        # Depth distribution
        depth_counts = defaultdict(int)
        max_depth = 0
        for node in self.nodes.values():
            if not node.is_pruned:
                depth_counts[node.depth] += 1
                max_depth = max(max_depth, node.depth)
        
        # Value distribution
        values = [node.value_estimate for node in self.nodes.values() if not node.is_pruned]
        avg_value = sum(values) / len(values) if values else 0.0
        
        return {
            "total_nodes": total_nodes,
            "active_nodes": active_nodes,
            "pruned_nodes": pruned_count,
            "pruning_rate": pruned_count / total_nodes if total_nodes > 0 else 0,
            "max_depth_reached": max_depth,
            "avg_node_value": avg_value,
            "expansions": self.expansion_count,
            "evaluations": self.evaluation_count,
            "node_type_distribution": dict(type_counts),
            "depth_distribution": dict(depth_counts),
            "best_paths_count": len(self.best_paths),
            "frontier_size": len(self.frontier)
        }
    
    def visualize_tree(self, max_nodes: int = 50) -> Dict[str, Any]:
        """Create visualization data for the reasoning tree."""
        
        # Select most important nodes for visualization
        important_nodes = sorted(
            [node for node in self.nodes.values() if not node.is_pruned],
            key=lambda x: x.value_estimate,
            reverse=True
        )[:max_nodes]
        
        # Create visualization structure
        viz_data = {
            "nodes": [],
            "edges": [],
            "stats": self.get_tree_statistics()
        }
        
        for node in important_nodes:
            viz_data["nodes"].append({
                "id": node.node_id,
                "content": node.content[:100] + "..." if len(node.content) > 100 else node.content,
                "type": node.node_type.value,
                "depth": node.depth,
                "confidence": node.confidence,
                "value": node.value_estimate,
                "visits": node.visit_count
            })
            
            # Add edge to parent
            if node.parent_id and node.parent_id in [n.node_id for n in important_nodes]:
                viz_data["edges"].append({
                    "from": node.parent_id,
                    "to": node.node_id
                })
        
        return viz_data