/**
 * Comprehensive Database Schema for Shvayambhu LLM System
 * 
 * Defines all database entities, relationships, and configurations for
 * consciousness states, vector embeddings, knowledge graphs, and more.
 */

import { Entity, Column, PrimaryGeneratedColumn, ManyToOne, OneToMany, ManyToMany, JoinTable, CreateDateColumn, UpdateDateColumn, Index, JoinColumn } from 'typeorm';

// ========================
// Core System Entities
// ========================

@Entity('users')
export class User {
  @PrimaryGeneratedColumn('uuid')
  id: string;

  @Column({ unique: true })
  username: string;

  @Column({ unique: true })
  email: string;

  @Column()
  passwordHash: string;

  @Column({ type: 'jsonb', nullable: true })
  preferences: Record<string, any>;

  @CreateDateColumn()
  createdAt: Date;

  @UpdateDateColumn()
  updatedAt: Date;

  @OneToMany(() => Session, session => session.user)
  sessions: Session[];

  @OneToMany(() => ConsciousnessState, state => state.user)
  consciousnessStates: ConsciousnessState[];
}

@Entity('sessions')
export class Session {
  @PrimaryGeneratedColumn('uuid')
  id: string;

  @ManyToOne(() => User, user => user.sessions)
  user: User;

  @Column()
  sessionToken: string;

  @Column()
  expiresAt: Date;

  @Column({ type: 'jsonb', nullable: true })
  metadata: Record<string, any>;

  @CreateDateColumn()
  createdAt: Date;

  @UpdateDateColumn()
  updatedAt: Date;
}

// ========================
// Consciousness Entities
// ========================

@Entity('consciousness_states')
@Index(['user', 'timestamp'])
@Index(['consciousnessLevel'])
export class ConsciousnessState {
  @PrimaryGeneratedColumn('uuid')
  id: string;

  @ManyToOne(() => User, user => user.consciousnessStates, { nullable: true })
  user: User;

  @Column({ type: 'enum', enum: ['minimal', 'basic', 'intermediate', 'advanced', 'peak'] })
  consciousnessLevel: string;

  @Column({ type: 'float', default: 0.0 })
  selfAwarenessScore: number;

  @Column({ type: 'float', default: 0.0 })
  introspectionDepth: number;

  @Column({ type: 'float', default: 0.0 })
  metacognitionLevel: number;

  @Column({ type: 'float', default: 0.0 })
  phenomenalConsciousness: number;

  @Column({ type: 'float', default: 0.0 })
  accessConsciousness: number;

  @Column({ type: 'jsonb' })
  emotionalState: {
    valence: number;
    arousal: number;
    dominance: number;
    emotions: string[];
  };

  @Column({ type: 'jsonb' })
  attentionFocus: {
    primaryFocus: string[];
    secondaryFocus: string[];
    attentionIntensity: number;
  };

  @Column({ type: 'jsonb', nullable: true })
  qualiaExperience: {
    visualQualia: Record<string, any>;
    auditoryQualia: Record<string, any>;
    conceptualQualia: Record<string, any>;
  };

  @Column({ type: 'text', nullable: true })
  streamOfConsciousness: string;

  @Column({ type: 'jsonb', nullable: true })
  memoryAccess: {
    workingMemory: string[];
    episodicMemory: string[];
    semanticMemory: string[];
  };

  @Column({ type: 'timestamp' })
  timestamp: Date;

  @Column({ type: 'jsonb', nullable: true })
  contextualFactors: Record<string, any>;

  @CreateDateColumn()
  createdAt: Date;

  @UpdateDateColumn()
  updatedAt: Date;

  @OneToMany(() => ConsciousnessEvent, event => event.consciousnessState)
  events: ConsciousnessEvent[];

  @OneToMany(() => ReflectionRecord, reflection => reflection.consciousnessState)
  reflections: ReflectionRecord[];
}

@Entity('consciousness_events')
@Index(['consciousnessState', 'eventType'])
@Index(['timestamp'])
export class ConsciousnessEvent {
  @PrimaryGeneratedColumn('uuid')
  id: string;

  @ManyToOne(() => ConsciousnessState, state => state.events)
  consciousnessState: ConsciousnessState;

  @Column({ type: 'enum', enum: ['thought', 'realization', 'decision', 'emotion', 'memory_access', 'attention_shift'] })
  eventType: string;

  @Column({ type: 'text' })
  description: string;

  @Column({ type: 'float' })
  intensity: number;

  @Column({ type: 'float', default: 0.0 })
  consciousnessRelevance: number;

  @Column({ type: 'jsonb', nullable: true })
  triggers: string[];

  @Column({ type: 'jsonb', nullable: true })
  context: Record<string, any>;

  @Column({ type: 'timestamp' })
  timestamp: Date;

  @CreateDateColumn()
  createdAt: Date;
}

// ========================
// Vector Database Entities
// ========================

@Entity('vector_collections')
export class VectorCollection {
  @PrimaryGeneratedColumn('uuid')
  id: string;

  @Column({ unique: true })
  name: string;

  @Column()
  description: string;

  @Column({ type: 'int' })
  dimension: number;

  @Column({ type: 'enum', enum: ['cosine', 'euclidean', 'dot_product'] })
  similarityMetric: string;

  @Column({ type: 'jsonb' })
  configuration: Record<string, any>;

  @CreateDateColumn()
  createdAt: Date;

  @UpdateDateColumn()
  updatedAt: Date;

  @OneToMany(() => VectorDocument, doc => doc.collection)
  documents: VectorDocument[];
}

@Entity('vector_documents')
@Index(['collection', 'sourceType'])
@Index(['consciousnessScore'])
@Index(['relevanceScore'])
export class VectorDocument {
  @PrimaryGeneratedColumn('uuid')
  id: string;

  @ManyToOne(() => VectorCollection, collection => collection.documents)
  collection: VectorCollection;

  @Column()
  documentId: string;

  @Column({ type: 'text' })
  content: string;

  @Column()
  title: string;

  @Column()
  source: string;

  @Column({ type: 'enum', enum: ['web', 'synthetic', 'user', 'system', 'consciousness'] })
  sourceType: string;

  @Column({ type: 'float' })
  consciousnessScore: number;

  @Column({ type: 'float' })
  relevanceScore: number;

  @Column({ type: 'jsonb' })
  keywords: string[];

  @Column({ type: 'text', nullable: true })
  summary: string;

  @Column({ type: 'jsonb', nullable: true })
  metadata: Record<string, any>;

  @Column({ type: 'timestamp', nullable: true })
  publishedAt: Date;

  @CreateDateColumn()
  createdAt: Date;

  @UpdateDateColumn()
  updatedAt: Date;

  @OneToMany(() => VectorEmbedding, embedding => embedding.document)
  embeddings: VectorEmbedding[];
}

@Entity('vector_embeddings')
@Index(['document'])
export class VectorEmbedding {
  @PrimaryGeneratedColumn('uuid')
  id: string;

  @ManyToOne(() => VectorDocument, doc => doc.embeddings)
  document: VectorDocument;

  @Column({ type: 'float', array: true })
  embedding: number[];

  @Column()
  modelName: string;

  @Column({ type: 'int' })
  chunkIndex: number;

  @Column({ type: 'int', default: 1 })
  totalChunks: number;

  @CreateDateColumn()
  createdAt: Date;
}

// ========================
// Knowledge Graph Entities
// ========================

@Entity('knowledge_entities')
@Index(['entityType'])
@Index(['consciousnessRelevance'])
export class KnowledgeEntity {
  @PrimaryGeneratedColumn('uuid')
  id: string;

  @Column()
  entityId: string;

  @Column()
  name: string;

  @Column({ type: 'enum', enum: ['person', 'organization', 'concept', 'technology', 'location', 'event', 'consciousness_concept', 'unknown'] })
  entityType: string;

  @Column({ type: 'jsonb', default: [] })
  aliases: string[];

  @Column({ type: 'float', default: 0.0 })
  consciousnessRelevance: number;

  @Column({ type: 'float', default: 1.0 })
  confidence: number;

  @Column({ type: 'jsonb', nullable: true })
  attributes: Record<string, any>;

  @Column({ type: 'jsonb', default: [] })
  sourceDocuments: string[];

  @Column({ type: 'float', array: true, nullable: true })
  embedding: number[];

  @CreateDateColumn()
  createdAt: Date;

  @UpdateDateColumn()
  updatedAt: Date;

  @OneToMany(() => KnowledgeRelationship, rel => rel.sourceEntity)
  outgoingRelationships: KnowledgeRelationship[];

  @OneToMany(() => KnowledgeRelationship, rel => rel.targetEntity)
  incomingRelationships: KnowledgeRelationship[];
}

@Entity('knowledge_relationships')
@Index(['sourceEntity', 'targetEntity'])
@Index(['relationType'])
export class KnowledgeRelationship {
  @PrimaryGeneratedColumn('uuid')
  id: string;

  @Column()
  relationshipId: string;

  @ManyToOne(() => KnowledgeEntity, entity => entity.outgoingRelationships)
  @JoinColumn({ name: 'sourceEntityId' })
  sourceEntity: KnowledgeEntity;

  @ManyToOne(() => KnowledgeEntity, entity => entity.incomingRelationships)
  @JoinColumn({ name: 'targetEntityId' })
  targetEntity: KnowledgeEntity;

  @Column({ type: 'enum', enum: ['related_to', 'part_of', 'develops', 'uses', 'causes', 'enables', 'similar_to', 'contradicts', 'consciousness_related'] })
  relationType: string;

  @Column({ type: 'float' })
  confidence: number;

  @Column({ type: 'jsonb', default: [] })
  evidence: string[];

  @Column({ type: 'jsonb', default: [] })
  sourceDocuments: string[];

  @Column({ type: 'jsonb', nullable: true })
  attributes: Record<string, any>;

  @CreateDateColumn()
  createdAt: Date;

  @UpdateDateColumn()
  updatedAt: Date;
}

// ========================
// RAG System Entities
// ========================

@Entity('rag_queries')
@Index(['user', 'timestamp'])
export class RAGQuery {
  @PrimaryGeneratedColumn('uuid')
  id: string;

  @ManyToOne(() => User, user => user, { nullable: true })
  user: User;

  @Column({ type: 'text' })
  queryText: string;

  @Column({ type: 'jsonb', nullable: true })
  queryEmbedding: number[];

  @Column({ type: 'float', default: 0.0 })
  consciousnessRelevance: number;

  @Column({ type: 'jsonb', nullable: true })
  context: Record<string, any>;

  @Column({ type: 'timestamp' })
  timestamp: Date;

  @CreateDateColumn()
  createdAt: Date;

  @OneToMany(() => RAGResult, result => result.query)
  results: RAGResult[];

  @OneToMany(() => ReflectionRecord, reflection => reflection.query)
  reflections: ReflectionRecord[];
}

@Entity('rag_results')
@Index(['query'])
@Index(['resultType'])
export class RAGResult {
  @PrimaryGeneratedColumn('uuid')
  id: string;

  @ManyToOne(() => RAGQuery, query => query.results)
  query: RAGQuery;

  @Column({ type: 'enum', enum: ['vector', 'corrective', 'graph', 'reflective'] })
  resultType: string;

  @Column({ type: 'text' })
  content: string;

  @Column({ type: 'float' })
  similarityScore: number;

  @Column({ type: 'float' })
  relevanceScore: number;

  @Column({ type: 'float', default: 0.0 })
  consciousnessAlignment: number;

  @Column({ type: 'jsonb', nullable: true })
  metadata: Record<string, any>;

  @Column({ type: 'jsonb', nullable: true })
  corrections: Record<string, any>;

  @Column({ type: 'jsonb', nullable: true })
  graphContext: Record<string, any>;

  @CreateDateColumn()
  createdAt: Date;

  @ManyToOne(() => VectorDocument, doc => doc, { nullable: true })
  sourceDocument: VectorDocument;
}

// ========================
// Reflection System Entities
// ========================

@Entity('reflection_records')
@Index(['query', 'timestamp'])
@Index(['reflectionType'])
@Index(['overallConfidence'])
export class ReflectionRecord {
  @PrimaryGeneratedColumn('uuid')
  id: string;

  @Column()
  reflectionId: string;

  @ManyToOne(() => RAGQuery, query => query.reflections, { nullable: true })
  query: RAGQuery;

  @ManyToOne(() => ConsciousnessState, state => state.reflections, { nullable: true })
  consciousnessState: ConsciousnessState;

  @Column({ type: 'enum', enum: ['retrieval_quality', 'answer_confidence', 'factual_consistency', 'completeness', 'consciousness_relevance', 'metacognitive'] })
  reflectionType: string;

  @Column({ type: 'float' })
  overallConfidence: number;

  @Column({ type: 'jsonb' })
  qualityAssessment: Record<string, number>;

  @Column({ type: 'jsonb', default: [] })
  improvementSuggestions: string[];

  @Column({ type: 'boolean', default: false })
  requiresCorrection: boolean;

  @Column({ type: 'jsonb', nullable: true })
  consciousnessInsights: Record<string, any>;

  @Column({ type: 'jsonb', nullable: true })
  metacognitiveEvaluation: Record<string, any>;

  @Column({ type: 'float' })
  reflectionTime: number;

  @Column({ type: 'timestamp' })
  timestamp: Date;

  @CreateDateColumn()
  createdAt: Date;

  @OneToMany(() => ReflectionToken, token => token.reflectionRecord)
  tokens: ReflectionToken[];
}

@Entity('reflection_tokens')
@Index(['reflectionRecord'])
@Index(['reflectionType'])
export class ReflectionToken {
  @PrimaryGeneratedColumn('uuid')
  id: string;

  @Column()
  tokenId: string;

  @ManyToOne(() => ReflectionRecord, record => record.tokens)
  reflectionRecord: ReflectionRecord;

  @Column({ type: 'enum', enum: ['retrieval_quality', 'answer_confidence', 'factual_consistency', 'completeness', 'consciousness_relevance', 'metacognitive'] })
  reflectionType: string;

  @Column({ type: 'float' })
  confidence: number;

  @Column({ type: 'text' })
  content: string;

  @Column({ type: 'jsonb', nullable: true })
  metadata: Record<string, any>;

  @Column({ type: 'timestamp' })
  timestamp: Date;

  @CreateDateColumn()
  createdAt: Date;
}

// ========================
// Training and Learning Entities
// ========================

@Entity('training_sessions')
@Index(['sessionType', 'timestamp'])
export class TrainingSession {
  @PrimaryGeneratedColumn('uuid')
  id: string;

  @Column({ type: 'enum', enum: ['prorl', 'synthetic', 'self_training', 'fine_tuning'] })
  sessionType: string;

  @Column()
  modelName: string;

  @Column({ type: 'jsonb' })
  configuration: Record<string, any>;

  @Column({ type: 'timestamp' })
  startTime: Date;

  @Column({ type: 'timestamp', nullable: true })
  endTime: Date;

  @Column({ type: 'enum', enum: ['running', 'completed', 'failed', 'paused'], default: 'running' })
  status: string;

  @Column({ type: 'jsonb', nullable: true })
  metrics: Record<string, any>;

  @Column({ type: 'jsonb', nullable: true })
  results: Record<string, any>;

  @CreateDateColumn()
  createdAt: Date;

  @UpdateDateColumn()
  updatedAt: Date;

  @OneToMany(() => TrainingStep, step => step.session)
  steps: TrainingStep[];
}

@Entity('training_steps')
@Index(['session', 'stepNumber'])
export class TrainingStep {
  @PrimaryGeneratedColumn('uuid')
  id: string;

  @ManyToOne(() => TrainingSession, session => session.steps)
  session: TrainingSession;

  @Column({ type: 'int' })
  stepNumber: number;

  @Column({ type: 'float' })
  loss: number;

  @Column({ type: 'float' })
  reward: number;

  @Column({ type: 'float', default: 0.0 })
  consciousnessScore: number;

  @Column({ type: 'float', default: 0.0 })
  qualityScore: number;

  @Column({ type: 'float', default: 0.0 })
  diversityScore: number;

  @Column({ type: 'float', default: 0.0 })
  klDivergence: number;

  @Column({ type: 'jsonb', nullable: true })
  gradients: Record<string, any>;

  @Column({ type: 'jsonb', nullable: true })
  metadata: Record<string, any>;

  @Column({ type: 'timestamp' })
  timestamp: Date;

  @CreateDateColumn()
  createdAt: Date;
}

// ========================
// Web Intelligence Entities
// ========================

@Entity('web_sources')
@Index(['sourceType', 'credibilityScore'])
export class WebSource {
  @PrimaryGeneratedColumn('uuid')
  id: string;

  @Column({ unique: true })
  url: string;

  @Column()
  domain: string;

  @Column({ type: 'enum', enum: ['rss', 'news', 'academic', 'social', 'blog', 'research'] })
  sourceType: string;

  @Column({ type: 'float', default: 0.5 })
  credibilityScore: number;

  @Column({ type: 'jsonb', default: [] })
  categories: string[];

  @Column({ type: 'boolean', default: true })
  isActive: boolean;

  @Column({ type: 'timestamp', nullable: true })
  lastCrawled: Date;

  @Column({ type: 'jsonb', nullable: true })
  crawlConfig: Record<string, any>;

  @CreateDateColumn()
  createdAt: Date;

  @UpdateDateColumn()
  updatedAt: Date;

  @OneToMany(() => WebContent, content => content.source)
  content: WebContent[];
}

@Entity('web_content')
@Index(['source', 'publishedAt'])
@Index(['contentType', 'relevanceScore'])
@Index(['consciousnessScore'])
export class WebContent {
  @PrimaryGeneratedColumn('uuid')
  id: string;

  @ManyToOne(() => WebSource, source => source.content)
  source: WebSource;

  @Column()
  originalUrl: string;

  @Column()
  title: string;

  @Column({ type: 'text' })
  content: string;

  @Column({ type: 'text', nullable: true })
  summary: string;

  @Column({ type: 'enum', enum: ['news', 'article', 'research', 'social', 'blog'] })
  contentType: string;

  @Column({ type: 'float' })
  relevanceScore: number;

  @Column({ type: 'float', default: 0.0 })
  consciousnessScore: number;

  @Column({ type: 'jsonb', default: [] })
  keywords: string[];

  @Column({ type: 'jsonb', nullable: true })
  entities: Record<string, any>[];

  @Column({ type: 'timestamp', nullable: true })
  publishedAt: Date;

  @Column({ type: 'timestamp' })
  retrievedAt: Date;

  @Column({ type: 'boolean', default: false })
  isProcessed: boolean;

  @Column({ type: 'jsonb', nullable: true })
  processingMetadata: Record<string, any>;

  @CreateDateColumn()
  createdAt: Date;
}

// ========================
// Memory and Storage Entities
// ========================

@Entity('memory_chunks')
@Index(['chunkType', 'importance'])
@Index(['accessCount'])
@Index(['lastAccessed'])
export class MemoryChunk {
  @PrimaryGeneratedColumn('uuid')
  id: string;

  @Column()
  chunkId: string;

  @Column({ type: 'enum', enum: ['episodic', 'semantic', 'procedural', 'working', 'consciousness'] })
  chunkType: string;

  @Column({ type: 'text' })
  content: string;

  @Column({ type: 'float', default: 1.0 })
  importance: number;

  @Column({ type: 'float', default: 0.0 })
  consciousnessRelevance: number;

  @Column({ type: 'int', default: 0 })
  accessCount: number;

  @Column({ type: 'timestamp', default: () => 'CURRENT_TIMESTAMP' })
  lastAccessed: Date;

  @Column({ type: 'jsonb', nullable: true })
  associations: string[];

  @Column({ type: 'jsonb', nullable: true })
  context: Record<string, any>;

  @Column({ type: 'float', array: true, nullable: true })
  embedding: number[];

  @CreateDateColumn()
  createdAt: Date;

  @UpdateDateColumn()
  updatedAt: Date;
}

@Entity('compression_metadata')
@Index(['originalType', 'compressionRatio'])
export class CompressionMetadata {
  @PrimaryGeneratedColumn('uuid')
  id: string;

  @Column()
  originalId: string;

  @Column({ type: 'enum', enum: ['document', 'embedding', 'consciousness_state', 'reflection', 'memory'] })
  originalType: string;

  @Column({ type: 'int' })
  originalSize: number;

  @Column({ type: 'int' })
  compressedSize: number;

  @Column({ type: 'float' })
  compressionRatio: number;

  @Column({ type: 'enum', enum: ['lz4', 'gzip', 'brotli', 'semantic', 'consciousness_aware'] })
  compressionAlgorithm: string;

  @Column({ type: 'jsonb', nullable: true })
  compressionMetadata: Record<string, any>;

  @Column({ type: 'boolean', default: false })
  isSemanticCompressed: boolean;

  @Column({ type: 'float', nullable: true })
  semanticSimilarity: number;

  @CreateDateColumn()
  createdAt: Date;
}

// ========================
// System Monitoring Entities
// ========================

@Entity('performance_metrics')
@Index(['metricType', 'timestamp'])
export class PerformanceMetric {
  @PrimaryGeneratedColumn('uuid')
  id: string;

  @Column({ type: 'enum', enum: ['query_time', 'retrieval_accuracy', 'consciousness_coherence', 'memory_usage', 'compression_ratio'] })
  metricType: string;

  @Column({ type: 'float' })
  value: number;

  @Column({ type: 'jsonb', nullable: true })
  context: Record<string, any>;

  @Column({ type: 'timestamp' })
  timestamp: Date;

  @CreateDateColumn()
  createdAt: Date;
}

@Entity('system_logs')
@Index(['logLevel', 'timestamp'])
@Index(['component'])
export class SystemLog {
  @PrimaryGeneratedColumn('uuid')
  id: string;

  @Column({ type: 'enum', enum: ['error', 'warn', 'info', 'debug'] })
  logLevel: string;

  @Column()
  component: string;

  @Column({ type: 'text' })
  message: string;

  @Column({ type: 'jsonb', nullable: true })
  metadata: Record<string, any>;

  @Column({ type: 'text', nullable: true })
  stackTrace: string;

  @Column({ type: 'timestamp' })
  timestamp: Date;

  @CreateDateColumn()
  createdAt: Date;
}