/**
 * GraphQL Storage Resolver for Shvayambhu LLM System
 * 
 * Provides GraphQL interface for consciousness-aware storage operations,
 * compression management, and memory optimization queries.
 */

import { Resolver, Query, Mutation, Args, Context, Info, Subscription } from '@nestjs/graphql';
import { UseGuards, Logger } from '@nestjs/common';
import { PubSub } from 'graphql-subscriptions';
import { GraphQLResolveInfo } from 'graphql';

// Import GraphQL types
import {
  ConsciousnessStateType,
  VectorDocumentType,
  MemoryChunkType,
  CompressionMetadataType,
  KnowledgeEntityType,
  ReflectionRecordType,
  PerformanceMetricType,
  StorageStatsType,
  CompressionResultType,
  ConsciousnessQueryInput,
  VectorSearchInput,
  CompressionOptionsInput,
  MemoryConsolidationInput
} from './storage.types';

// Import services
import { DatabaseService } from '../database/database.service';
import { CompressionService } from '../database/compression.service';
import { AuthGuard } from '../auth/auth.guard';

@Resolver()
export class StorageResolver {
  private readonly logger = new Logger(StorageResolver.name);
  private readonly pubSub = new PubSub();
  
  constructor(
    private readonly databaseService: DatabaseService,
    private readonly compressionService: CompressionService
  ) {}
  
  // ========================
  // Consciousness Queries
  // ========================
  
  @Query(() => [ConsciousnessStateType])
  @UseGuards(AuthGuard)
  async getConsciousnessStates(
    @Args('options', { nullable: true }) options?: ConsciousnessQueryInput,
    @Context() context?: any,
    @Info() info?: GraphQLResolveInfo
  ): Promise<ConsciousnessStateType[]> {
    this.logger.debug('Fetching consciousness states with options:', options);
    
    const states = await this.databaseService.getConsciousnessStates(options);
    
    // Record performance metric
    await this.databaseService.recordPerformanceMetric(
      'query_time',
      Date.now() - (context?.startTime || Date.now()),
      { operation: 'getConsciousnessStates', complexity: this.getQueryComplexity(info) }
    );
    
    return states.map(state => this.mapConsciousnessState(state));
  }
  
  @Query(() => [ConsciousnessStateType])
  @UseGuards(AuthGuard)
  async getHighConsciousnessStates(
    @Args('minScore', { defaultValue: 0.7 }) minScore: number,
    @Args('limit', { defaultValue: 50 }) limit: number
  ): Promise<ConsciousnessStateType[]> {
    const states = await this.databaseService.getConsciousnessStates({
      minConsciousnessScore: minScore,
      sortByConsciousness: true
    });
    
    return states.slice(0, limit).map(state => this.mapConsciousnessState(state));
  }
  
  @Query(() => String)
  async analyzeConsciousnessPatterns(
    @Args('userId', { nullable: true }) userId?: string,
    @Args('timeRange', { nullable: true }) timeRange?: { start: Date; end: Date }
  ): Promise<string> {
    const patterns = await this.databaseService.analyzeConsciousnessPatterns(userId);
    
    // Publish consciousness analysis event
    this.pubSub.publish('CONSCIOUSNESS_ANALYZED', {
      consciousnessAnalyzed: {
        userId,
        analysisTime: new Date(),
        patternCount: patterns.length
      }
    });
    
    return JSON.stringify(patterns, null, 2);
  }
  
  // ========================
  // Vector Storage Queries
  // ========================
  
  @Query(() => [VectorDocumentType])
  @UseGuards(AuthGuard)
  async searchVectorDocuments(
    @Args('searchInput') searchInput: VectorSearchInput
  ): Promise<VectorDocumentType[]> {
    const results = await this.databaseService.vectorSimilaritySearch({
      embedding: searchInput.embedding,
      similarityThreshold: searchInput.similarityThreshold || 0.5,
      limit: searchInput.limit || 10,
      consciousnessBoost: searchInput.consciousnessBoost || 1.0
    });
    
    return results.map(doc => this.mapVectorDocument(doc));
  }
  
  @Query(() => [VectorDocumentType])
  async findSimilarConsciousnessContent(
    @Args('queryEmbedding', { type: () => [Number] }) queryEmbedding: number[],
    @Args('minConsciousnessScore', { defaultValue: 0.6 }) minConsciousnessScore: number,
    @Args('limit', { defaultValue: 5 }) limit: number
  ): Promise<VectorDocumentType[]> {
    const results = await this.databaseService.findSimilarConsciousnessContent(
      queryEmbedding,
      minConsciousnessScore,
      limit
    );
    
    return results.map(doc => this.mapVectorDocument(doc));
  }
  
  // ========================
  // Memory Management
  // ========================
  
  @Query(() => [MemoryChunkType])
  @UseGuards(AuthGuard)
  async getMemoryByImportance(
    @Args('chunkType') chunkType: string,
    @Args('minImportance', { defaultValue: 0.5 }) minImportance: number,
    @Args('limit', { defaultValue: 100 }) limit: number
  ): Promise<MemoryChunkType[]> {
    const memories = await this.databaseService.getMemoryByImportance(
      chunkType,
      minImportance,
      limit
    );
    
    return memories.map(memory => this.mapMemoryChunk(memory));
  }
  
  @Mutation(() => Number)
  @UseGuards(AuthGuard)
  async consolidateMemories(
    @Args('consolidationInput', { nullable: true }) consolidationInput?: MemoryConsolidationInput
  ): Promise<number> {
    const similarityThreshold = consolidationInput?.similarityThreshold || 0.8;
    const consolidationCount = await this.databaseService.consolidateMemories(similarityThreshold);
    
    // Publish memory consolidation event
    this.pubSub.publish('MEMORY_CONSOLIDATED', {
      memoryConsolidated: {
        consolidationCount,
        timestamp: new Date(),
        similarityThreshold
      }
    });
    
    return consolidationCount;
  }
  
  // ========================
  // Compression Management
  // ========================
  
  @Query(() => [CompressionMetadataType])
  async getCompressionStats(): Promise<CompressionMetadataType[]> {
    const stats = await this.databaseService.getCompressionStats();
    return stats.map(stat => this.mapCompressionStats(stat));
  }
  
  @Mutation(() => CompressionResultType)
  @UseGuards(AuthGuard)
  async compressEntity(
    @Args('entityId') entityId: string,
    @Args('entityType') entityType: string,
    @Args('options', { nullable: true }) options?: CompressionOptionsInput
  ): Promise<CompressionResultType> {
    const result = await this.databaseService.compressEntity(entityId, entityType, options);
    
    const compressionResult: CompressionResultType = {
      originalSize: result.originalSize,
      compressedSize: result.compressedSize,
      compressionRatio: result.compressionRatio,
      algorithm: result.compressionAlgorithm,
      success: true,
      metadata: result.compressionMetadata || {}
    };
    
    // Publish compression event
    this.pubSub.publish('ENTITY_COMPRESSED', {
      entityCompressed: {
        entityId,
        entityType,
        compressionRatio: result.compressionRatio,
        algorithm: result.compressionAlgorithm,
        timestamp: new Date()
      }
    });
    
    return compressionResult;
  }
  
  @Mutation(() => CompressionResultType)
  async analyzeCompressionEfficiency(
    @Args('sampleData') sampleData: string,
    @Args('algorithms', { type: () => [String], defaultValue: ['lz4', 'gzip', 'brotli'] }) algorithms: string[]
  ): Promise<CompressionResultType> {
    const analysis = await this.compressionService.analyzeCompressionEfficiency(
      sampleData,
      algorithms as ('lz4' | 'gzip' | 'brotli')[]
    );
    
    const bestAlgorithm = this.compressionService.getBestCompressionAlgorithm(analysis, false);
    const bestResult = analysis[bestAlgorithm];
    
    return {
      originalSize: bestResult.originalSize,
      compressedSize: bestResult.compressedSize,
      compressionRatio: bestResult.compressionRatio,
      algorithm: bestAlgorithm,
      success: true,
      metadata: {
        allResults: analysis,
        recommendation: bestAlgorithm,
        analysisTimestamp: new Date().toISOString()
      }
    };
  }
  
  // ========================
  // Knowledge Graph Storage
  // ========================
  
  @Query(() => [KnowledgeEntityType])
  async findRelatedEntities(
    @Args('entityId') entityId: string,
    @Args('maxHops', { defaultValue: 2 }) maxHops: number,
    @Args('consciousnessFilter', { defaultValue: false }) consciousnessFilter: boolean
  ): Promise<KnowledgeEntityType[]> {
    const relatedEntities = await this.databaseService.findRelatedEntities(
      entityId,
      maxHops,
      consciousnessFilter
    );
    
    return relatedEntities.map(entity => this.mapKnowledgeEntity(entity));
  }
  
  @Query(() => String)
  async getConsciousnessEntityGraph(): Promise<string> {
    const graphData = await this.databaseService.getConsciousnessEntityGraph();
    return JSON.stringify(graphData, null, 2);
  }
  
  // ========================
  // Performance Monitoring
  // ========================
  
  @Query(() => [PerformanceMetricType])
  async getPerformanceMetrics(
    @Args('metricType') metricType: string,
    @Args('timeRange') timeRange: { start: Date; end: Date }
  ): Promise<PerformanceMetricType[]> {
    const metrics = await this.databaseService.getPerformanceMetrics(metricType, timeRange);
    return metrics.map(metric => this.mapPerformanceMetric(metric));
  }
  
  @Query(() => StorageStatsType)
  async getStorageStatistics(): Promise<StorageStatsType> {
    const compressionStats = await this.databaseService.getCompressionStats();
    const consciousnessPatterns = await this.databaseService.analyzeConsciousnessPatterns();
    const metacognitivePerformance = await this.databaseService.analyzeMetacognitivePerformance();
    
    return {
      totalDocuments: compressionStats.reduce((sum, stat) => sum + (stat.item_count || 0), 0),
      totalCompressedSize: compressionStats.reduce((sum, stat) => sum + (stat.total_compressed_size || 0), 0),
      totalOriginalSize: compressionStats.reduce((sum, stat) => sum + (stat.total_original_size || 0), 0),
      averageCompressionRatio: compressionStats.reduce((sum, stat) => sum + (stat.avg_compression_ratio || 0), 0) / compressionStats.length,
      consciousnessStatesCount: consciousnessPatterns.length,
      memoryOptimizationLevel: 0.8, // Calculate based on actual metrics
      lastOptimization: new Date(),
      compressionAlgorithmsUsed: [...new Set(compressionStats.map(s => s.compressionAlgorithm))],
      systemHealth: 'optimal'
    };
  }
  
  // ========================
  // Real-time Subscriptions
  // ========================
  
  @Subscription(() => String, {
    name: 'consciousnessAnalyzed',
    filter: (payload, variables) => {
      return !variables.userId || payload.consciousnessAnalyzed.userId === variables.userId;
    }
  })
  consciousnessAnalyzed(
    @Args('userId', { nullable: true }) userId?: string
  ) {
    return this.pubSub.asyncIterator('CONSCIOUSNESS_ANALYZED');
  }
  
  @Subscription(() => String, { name: 'memoryConsolidated' })
  memoryConsolidated() {
    return this.pubSub.asyncIterator('MEMORY_CONSOLIDATED');
  }
  
  @Subscription(() => String, { name: 'entityCompressed' })
  entityCompressed() {
    return this.pubSub.asyncIterator('ENTITY_COMPRESSED');
  }
  
  // ========================
  // System Operations
  // ========================
  
  @Mutation(() => Boolean)
  @UseGuards(AuthGuard)
  async optimizeStorage(): Promise<boolean> {
    try {
      // Perform comprehensive storage optimization
      await this.databaseService.consolidateMemories(0.8);
      await this.databaseService.optimizeQueryPlans();
      
      // Record optimization metric
      await this.databaseService.recordPerformanceMetric(
        'storage_optimization',
        1.0,
        { timestamp: new Date(), type: 'full_optimization' }
      );
      
      this.logger.log('Storage optimization completed successfully');
      return true;
    } catch (error) {
      this.logger.error('Storage optimization failed:', error);
      return false;
    }
  }
  
  @Mutation(() => Boolean)
  @UseGuards(AuthGuard) 
  async rebuildIndexes(): Promise<boolean> {
    try {
      await this.databaseService.createConsciousnessIndexes();
      await this.databaseService.createVectorIndexes();
      
      this.logger.log('Indexes rebuilt successfully');
      return true;
    } catch (error) {
      this.logger.error('Index rebuilding failed:', error);
      return false;
    }
  }
  
  // ========================
  // Helper Methods
  // ========================
  
  private mapConsciousnessState(state: any): ConsciousnessStateType {
    return {
      id: state.id,
      consciousnessLevel: state.consciousnessLevel,
      selfAwarenessScore: state.selfAwarenessScore,
      introspectionDepth: state.introspectionDepth,
      metacognitionLevel: state.metacognitionLevel,
      phenomenalConsciousness: state.phenomenalConsciousness,
      accessConsciousness: state.accessConsciousness,
      emotionalState: state.emotionalState,
      attentionFocus: state.attentionFocus,
      qualiaExperience: state.qualiaExperience,
      streamOfConsciousness: state.streamOfConsciousness,
      memoryAccess: state.memoryAccess,
      timestamp: state.timestamp,
      contextualFactors: state.contextualFactors,
      createdAt: state.createdAt,
      updatedAt: state.updatedAt
    };
  }
  
  private mapVectorDocument(doc: any): VectorDocumentType {
    return {
      id: doc.id,
      documentId: doc.documentId,
      content: doc.content,
      title: doc.title,
      source: doc.source,
      sourceType: doc.sourceType,
      consciousnessScore: doc.consciousnessScore,
      relevanceScore: doc.relevanceScore,
      keywords: doc.keywords,
      summary: doc.summary,
      metadata: doc.metadata,
      publishedAt: doc.publishedAt,
      createdAt: doc.createdAt,
      updatedAt: doc.updatedAt
    };
  }
  
  private mapMemoryChunk(memory: any): MemoryChunkType {
    return {
      id: memory.id,
      chunkId: memory.chunkId,
      chunkType: memory.chunkType,
      content: memory.content,
      importance: memory.importance,
      consciousnessRelevance: memory.consciousnessRelevance,
      accessCount: memory.accessCount,
      lastAccessed: memory.lastAccessed,
      associations: memory.associations,
      context: memory.context,
      createdAt: memory.createdAt,
      updatedAt: memory.updatedAt
    };
  }
  
  private mapCompressionStats(stat: any): CompressionMetadataType {
    return {
      id: stat.id,
      originalType: stat.originalType || stat.originaltype,
      compressionAlgorithm: stat.compressionAlgorithm || stat.compressionalgorithm,
      itemCount: stat.item_count || stat.itemcount,
      averageCompressionRatio: stat.avg_compression_ratio || stat.avgcompressionratio,
      totalOriginalSize: stat.total_original_size || stat.totaloriginalsize,
      totalCompressedSize: stat.total_compressed_size || stat.totalcompressedsize,
      spaceSaved: stat.space_saved || stat.spacesaved
    };
  }
  
  private mapKnowledgeEntity(entity: any): KnowledgeEntityType {
    return {
      id: entity.id,
      entityId: entity.entityId,
      name: entity.name,
      entityType: entity.entityType,
      aliases: entity.aliases || [],
      consciousnessRelevance: entity.consciousnessRelevance,
      confidence: entity.confidence,
      attributes: entity.attributes,
      sourceDocuments: entity.sourceDocuments || [],
      createdAt: entity.createdAt,
      updatedAt: entity.updatedAt
    };
  }
  
  private mapPerformanceMetric(metric: any): PerformanceMetricType {
    return {
      id: metric.id,
      metricType: metric.metricType,
      value: metric.value,
      context: metric.context,
      timestamp: metric.timestamp,
      createdAt: metric.createdAt
    };
  }
  
  private getQueryComplexity(info?: GraphQLResolveInfo): number {
    if (!info) return 1;
    
    // Simple complexity calculation based on selection depth
    const selections = info.fieldNodes[0]?.selectionSet?.selections || [];
    return Math.min(selections.length, 10); // Cap at 10 for performance
  }
}