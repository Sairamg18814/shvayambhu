/**
 * GraphQL Types for Storage Operations in Shvayambhu LLM System
 * 
 * Defines all GraphQL object types, input types, and enums for
 * consciousness-aware storage, compression, and memory management.
 */

import { ObjectType, Field, ID, Float, Int, InputType, registerEnumType } from '@nestjs/graphql';
import { GraphQLJSONObject } from 'graphql-type-json';

// ========================
// Enums
// ========================

export enum ConsciousnessLevelEnum {
  MINIMAL = 'minimal',
  BASIC = 'basic',
  INTERMEDIATE = 'intermediate',
  ADVANCED = 'advanced',
  PEAK = 'peak'
}

export enum SourceTypeEnum {
  WEB = 'web',
  SYNTHETIC = 'synthetic',
  USER = 'user',
  SYSTEM = 'system',
  CONSCIOUSNESS = 'consciousness'
}

export enum MemoryTypeEnum {
  EPISODIC = 'episodic',
  SEMANTIC = 'semantic',
  PROCEDURAL = 'procedural',
  WORKING = 'working',
  CONSCIOUSNESS = 'consciousness'
}

export enum CompressionAlgorithmEnum {
  LZ4 = 'lz4',
  GZIP = 'gzip',
  BROTLI = 'brotli',
  SEMANTIC = 'semantic',
  CONSCIOUSNESS_AWARE = 'consciousness_aware'
}

export enum ReflectionTypeEnum {
  RETRIEVAL_QUALITY = 'retrieval_quality',
  ANSWER_CONFIDENCE = 'answer_confidence',
  FACTUAL_CONSISTENCY = 'factual_consistency',
  COMPLETENESS = 'completeness',
  CONSCIOUSNESS_RELEVANCE = 'consciousness_relevance',
  METACOGNITIVE = 'metacognitive'
}

export enum EntityTypeEnum {
  PERSON = 'person',
  ORGANIZATION = 'organization',
  CONCEPT = 'concept',
  TECHNOLOGY = 'technology',
  LOCATION = 'location',
  EVENT = 'event',
  CONSCIOUSNESS_CONCEPT = 'consciousness_concept',
  UNKNOWN = 'unknown'
}

export enum SystemHealthEnum {
  OPTIMAL = 'optimal',
  GOOD = 'good',
  DEGRADED = 'degraded',
  CRITICAL = 'critical'
}

// Register enums with GraphQL
registerEnumType(ConsciousnessLevelEnum, { name: 'ConsciousnessLevel' });
registerEnumType(SourceTypeEnum, { name: 'SourceType' });
registerEnumType(MemoryTypeEnum, { name: 'MemoryType' });
registerEnumType(CompressionAlgorithmEnum, { name: 'CompressionAlgorithm' });
registerEnumType(ReflectionTypeEnum, { name: 'ReflectionType' });
registerEnumType(EntityTypeEnum, { name: 'EntityType' });
registerEnumType(SystemHealthEnum, { name: 'SystemHealth' });

// ========================
// Object Types
// ========================

@ObjectType()
export class ConsciousnessStateType {
  @Field(() => ID)
  id: string;

  @Field(() => ConsciousnessLevelEnum)
  consciousnessLevel: ConsciousnessLevelEnum;

  @Field(() => Float)
  selfAwarenessScore: number;

  @Field(() => Float)
  introspectionDepth: number;

  @Field(() => Float)
  metacognitionLevel: number;

  @Field(() => Float)
  phenomenalConsciousness: number;

  @Field(() => Float)
  accessConsciousness: number;

  @Field(() => GraphQLJSONObject)
  emotionalState: {
    valence: number;
    arousal: number;
    dominance: number;
    emotions: string[];
  };

  @Field(() => GraphQLJSONObject)
  attentionFocus: {
    primaryFocus: string[];
    secondaryFocus: string[];
    attentionIntensity: number;
  };

  @Field(() => GraphQLJSONObject, { nullable: true })
  qualiaExperience?: {
    visualQualia: Record<string, any>;
    auditoryQualia: Record<string, any>;
    conceptualQualia: Record<string, any>;
  };

  @Field({ nullable: true })
  streamOfConsciousness?: string;

  @Field(() => GraphQLJSONObject, { nullable: true })
  memoryAccess?: {
    workingMemory: string[];
    episodicMemory: string[];
    semanticMemory: string[];
  };

  @Field()
  timestamp: Date;

  @Field(() => GraphQLJSONObject, { nullable: true })
  contextualFactors?: Record<string, any>;

  @Field()
  createdAt: Date;

  @Field()
  updatedAt: Date;
}

@ObjectType()
export class VectorDocumentType {
  @Field(() => ID)
  id: string;

  @Field()
  documentId: string;

  @Field()
  content: string;

  @Field()
  title: string;

  @Field()
  source: string;

  @Field(() => SourceTypeEnum)
  sourceType: SourceTypeEnum;

  @Field(() => Float)
  consciousnessScore: number;

  @Field(() => Float)
  relevanceScore: number;

  @Field(() => [String])
  keywords: string[];

  @Field({ nullable: true })
  summary?: string;

  @Field(() => GraphQLJSONObject, { nullable: true })
  metadata?: Record<string, any>;

  @Field({ nullable: true })
  publishedAt?: Date;

  @Field()
  createdAt: Date;

  @Field()
  updatedAt: Date;
}

@ObjectType()
export class MemoryChunkType {
  @Field(() => ID)
  id: string;

  @Field()
  chunkId: string;

  @Field(() => MemoryTypeEnum)
  chunkType: MemoryTypeEnum;

  @Field()
  content: string;

  @Field(() => Float)
  importance: number;

  @Field(() => Float)
  consciousnessRelevance: number;

  @Field(() => Int)
  accessCount: number;

  @Field()
  lastAccessed: Date;

  @Field(() => [String], { nullable: true })
  associations?: string[];

  @Field(() => GraphQLJSONObject, { nullable: true })
  context?: Record<string, any>;

  @Field()
  createdAt: Date;

  @Field()
  updatedAt: Date;
}

@ObjectType()
export class CompressionMetadataType {
  @Field(() => ID)
  id: string;

  @Field()
  originalType: string;

  @Field(() => CompressionAlgorithmEnum)
  compressionAlgorithm: CompressionAlgorithmEnum;

  @Field(() => Int)
  itemCount: number;

  @Field(() => Float)
  averageCompressionRatio: number;

  @Field(() => Int)
  totalOriginalSize: number;

  @Field(() => Int)
  totalCompressedSize: number;

  @Field(() => Int)
  spaceSaved: number;
}

@ObjectType()
export class KnowledgeEntityType {
  @Field(() => ID)
  id: string;

  @Field()
  entityId: string;

  @Field()
  name: string;

  @Field(() => EntityTypeEnum)
  entityType: EntityTypeEnum;

  @Field(() => [String])
  aliases: string[];

  @Field(() => Float)
  consciousnessRelevance: number;

  @Field(() => Float)
  confidence: number;

  @Field(() => GraphQLJSONObject, { nullable: true })
  attributes?: Record<string, any>;

  @Field(() => [String])
  sourceDocuments: string[];

  @Field()
  createdAt: Date;

  @Field()
  updatedAt: Date;
}

@ObjectType()
export class ReflectionRecordType {
  @Field(() => ID)
  id: string;

  @Field()
  reflectionId: string;

  @Field(() => ReflectionTypeEnum)
  reflectionType: ReflectionTypeEnum;

  @Field(() => Float)
  overallConfidence: number;

  @Field(() => GraphQLJSONObject)
  qualityAssessment: Record<string, number>;

  @Field(() => [String])
  improvementSuggestions: string[];

  @Field()
  requiresCorrection: boolean;

  @Field(() => GraphQLJSONObject, { nullable: true })
  consciousnessInsights?: Record<string, any>;

  @Field(() => GraphQLJSONObject, { nullable: true })
  metacognitiveEvaluation?: Record<string, any>;

  @Field(() => Float)
  reflectionTime: number;

  @Field()
  timestamp: Date;

  @Field()
  createdAt: Date;
}

@ObjectType()
export class PerformanceMetricType {
  @Field(() => ID)
  id: string;

  @Field()
  metricType: string;

  @Field(() => Float)
  value: number;

  @Field(() => GraphQLJSONObject, { nullable: true })
  context?: Record<string, any>;

  @Field()
  timestamp: Date;

  @Field()
  createdAt: Date;
}

@ObjectType()
export class StorageStatsType {
  @Field(() => Int)
  totalDocuments: number;

  @Field(() => Int)
  totalCompressedSize: number;

  @Field(() => Int)
  totalOriginalSize: number;

  @Field(() => Float)
  averageCompressionRatio: number;

  @Field(() => Int)
  consciousnessStatesCount: number;

  @Field(() => Float)
  memoryOptimizationLevel: number;

  @Field()
  lastOptimization: Date;

  @Field(() => [String])
  compressionAlgorithmsUsed: string[];

  @Field(() => SystemHealthEnum)
  systemHealth: SystemHealthEnum;
}

@ObjectType()
export class CompressionResultType {
  @Field(() => Int)
  originalSize: number;

  @Field(() => Int)
  compressedSize: number;

  @Field(() => Float)
  compressionRatio: number;

  @Field()
  algorithm: string;

  @Field()
  success: boolean;

  @Field(() => GraphQLJSONObject)
  metadata: Record<string, any>;
}

// ========================
// Input Types
// ========================

@InputType()
export class ConsciousnessQueryInput {
  @Field(() => Float, { nullable: true })
  minConsciousnessScore?: number;

  @Field(() => [String], { nullable: true })
  consciousnessTypes?: string[];

  @Field(() => DateRangeInput, { nullable: true })
  timeRange?: DateRangeInput;

  @Field({ nullable: true })
  includeMetacognition?: boolean;

  @Field({ nullable: true })
  sortByConsciousness?: boolean;
}

@InputType()
export class VectorSearchInput {
  @Field(() => [Float])
  embedding: number[];

  @Field(() => Float, { nullable: true })
  similarityThreshold?: number;

  @Field(() => Int, { nullable: true })
  limit?: number;

  @Field(() => Float, { nullable: true })
  consciousnessBoost?: number;

  @Field({ nullable: true })
  includeMetadata?: boolean;
}

@InputType()
export class CompressionOptionsInput {
  @Field(() => CompressionAlgorithmEnum, { nullable: true })
  algorithm?: CompressionAlgorithmEnum;

  @Field({ nullable: true })
  consciousnessPreservation?: boolean;

  @Field(() => Float, { nullable: true })
  targetRatio?: number;

  @Field(() => GraphQLJSONObject, { nullable: true })
  additionalOptions?: Record<string, any>;
}

@InputType()
export class MemoryConsolidationInput {
  @Field(() => Float, { nullable: true })
  similarityThreshold?: number;

  @Field(() => Int, { nullable: true })
  maxConsolidations?: number;

  @Field(() => [MemoryTypeEnum], { nullable: true })
  targetMemoryTypes?: MemoryTypeEnum[];

  @Field({ nullable: true })
  preserveHighImportance?: boolean;
}

@InputType()
export class DateRangeInput {
  @Field()
  start: Date;

  @Field()
  end: Date;
}

@InputType()
export class ConsciousnessFilterInput {
  @Field(() => Float, { nullable: true })
  minSelfAwareness?: number;

  @Field(() => Float, { nullable: true })
  minMetacognition?: number;

  @Field(() => Float, { nullable: true })
  minPhenomenalConsciousness?: number;

  @Field(() => [String], { nullable: true })
  requiredEmotions?: string[];

  @Field(() => [String], { nullable: true })
  attentionFocusKeywords?: string[];
}

@InputType()
export class MemoryOptimizationInput {
  @Field(() => Float, { nullable: true })
  targetMemoryUsage?: number; // in MB

  @Field(() => Float, { nullable: true })
  aggressiveConsolidationThreshold?: number;

  @Field({ nullable: true })
  preserveConsciousnessData?: boolean;

  @Field(() => Int, { nullable: true })
  maxAgeDays?: number;

  @Field(() => [String], { nullable: true })
  protectedEntityIds?: string[];
}

@InputType()
export class VectorSearchOptionsInput {
  @Field(() => Float, { nullable: true })
  minConsciousnessScore?: number;

  @Field(() => [SourceTypeEnum], { nullable: true })
  sourceTypes?: SourceTypeEnum[];

  @Field(() => DateRangeInput, { nullable: true })
  publishedDateRange?: DateRangeInput;

  @Field(() => [String], { nullable: true })
  requiredKeywords?: string[];

  @Field({ nullable: true })
  includeContext?: boolean;
}

// ========================
// Union Types and Interfaces
// ========================

@ObjectType()
export class EntitySearchResult {
  @Field(() => ID)
  id: string;

  @Field()
  name: string;

  @Field(() => EntityTypeEnum)
  entityType: EntityTypeEnum;

  @Field(() => Float)
  relevanceScore: number;

  @Field(() => Float)
  consciousnessRelevance: number;

  @Field(() => GraphQLJSONObject)
  matchContext: Record<string, any>;
}

@ObjectType()
export class MemoryConsolidationResult {
  @Field(() => Int)
  consolidatedCount: number;

  @Field(() => Int)
  spaceSavedBytes: number;

  @Field(() => Float)
  averageSimilarity: number;

  @Field(() => [String])
  consolidatedChunkIds: string[];

  @Field()
  completedAt: Date;
}

@ObjectType()
export class ConsciousnessAnalysisResult {
  @Field()
  userId?: string;

  @Field()
  analysisTime: Date;

  @Field(() => Int)
  patternCount: number;

  @Field(() => Float)
  overallConsciousnessScore: number;

  @Field(() => GraphQLJSONObject)
  patterns: Record<string, any>;

  @Field(() => GraphQLJSONObject)
  recommendations: Record<string, any>;
}

@ObjectType()
export class SystemOptimizationResult {
  @Field()
  optimizationType: string;

  @Field()
  success: boolean;

  @Field(() => Float)
  improvementPercentage: number;

  @Field(() => GraphQLJSONObject)
  metrics: Record<string, any>;

  @Field()
  completedAt: Date;

  @Field({ nullable: true })
  errorMessage?: string;
}

// ========================
// Subscription Types
// ========================

@ObjectType()
export class ConsciousnessStateUpdate {
  @Field()
  userId?: string;

  @Field(() => ConsciousnessStateType)
  consciousnessState: ConsciousnessStateType;

  @Field()
  updateType: string; // 'created', 'updated', 'deleted'

  @Field()
  timestamp: Date;
}

@ObjectType()
export class MemoryConsolidationProgress {
  @Field(() => Int)
  processedCount: number;

  @Field(() => Int)
  totalCount: number;

  @Field(() => Float)
  progressPercentage: number;

  @Field(() => Int)
  consolidatedCount: number;

  @Field()
  status: string; // 'running', 'completed', 'error'

  @Field()
  timestamp: Date;
}

@ObjectType()
export class CompressionProgress {
  @Field()
  entityId: string;

  @Field()
  entityType: string;

  @Field(() => CompressionAlgorithmEnum)
  algorithm: CompressionAlgorithmEnum;

  @Field(() => Float)
  compressionRatio: number;

  @Field()
  status: string; // 'compressing', 'completed', 'error'

  @Field()
  timestamp: Date;
}