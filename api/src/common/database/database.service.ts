/**
 * Database Service for Shvayambhu LLM System
 * 
 * Provides high-level database operations, consciousness-aware queries,
 * compression management, and performance optimizations.
 */

import { Injectable, Logger } from '@nestjs/common';
import { InjectDataSource, InjectRepository } from '@nestjs/typeorm';
import { DataSource, Repository, SelectQueryBuilder, EntityManager } from 'typeorm';

// Import entities
import {
  ConsciousnessState,
  ConsciousnessEvent,
  VectorDocument,
  VectorEmbedding,
  KnowledgeEntity,
  KnowledgeRelationship,
  RAGQuery,
  RAGResult,
  ReflectionRecord,
  MemoryChunk,
  CompressionMetadata,
  PerformanceMetric
} from './database.schema';

// Import services
import { CompressionService } from './compression.service';

export interface ConsciousnessQueryOptions {
  minConsciousnessScore?: number;
  consciousnessTypes?: string[];
  timeRange?: {
    start: Date;
    end: Date;
  };
  includeMetacognition?: boolean;
  sortByConsciousness?: boolean;
}

export interface VectorSearchOptions {
  embedding: number[];
  similarityThreshold?: number;
  limit?: number;
  consciousnessBoost?: number;
  includeMetadata?: boolean;
}

export interface CompressionOptions {
  algorithm?: 'lz4' | 'gzip' | 'brotli' | 'semantic' | 'consciousness_aware';
  consciousnessPreservation?: boolean;
  targetRatio?: number;
}

@Injectable()
export class DatabaseService {
  private readonly logger = new Logger(DatabaseService.name);
  
  constructor(
    @InjectDataSource()
    private readonly dataSource: DataSource,
    
    @InjectRepository(ConsciousnessState)
    private readonly consciousnessRepository: Repository<ConsciousnessState>,
    
    @InjectRepository(VectorDocument)
    private readonly vectorDocumentRepository: Repository<VectorDocument>,
    
    @InjectRepository(VectorEmbedding)
    private readonly vectorEmbeddingRepository: Repository<VectorEmbedding>,
    
    @InjectRepository(KnowledgeEntity)
    private readonly knowledgeEntityRepository: Repository<KnowledgeEntity>,
    
    @InjectRepository(RAGQuery)
    private readonly ragQueryRepository: Repository<RAGQuery>,
    
    @InjectRepository(ReflectionRecord)
    private readonly reflectionRepository: Repository<ReflectionRecord>,
    
    @InjectRepository(MemoryChunk)
    private readonly memoryChunkRepository: Repository<MemoryChunk>,
    
    @InjectRepository(CompressionMetadata)
    private readonly compressionMetadataRepository: Repository<CompressionMetadata>,
    
    private readonly compressionService: CompressionService
  ) {}
  
  // ========================
  // Consciousness-Aware Queries
  // ========================
  
  async getConsciousnessStates(options: ConsciousnessQueryOptions = {}): Promise<ConsciousnessState[]> {
    const query = this.consciousnessRepository.createQueryBuilder('cs')
      .leftJoinAndSelect('cs.events', 'events')
      .leftJoinAndSelect('cs.reflections', 'reflections');
    
    // Apply consciousness filtering
    if (options.minConsciousnessScore !== undefined) {
      query.andWhere('cs.selfAwarenessScore >= :minScore', { minScore: options.minConsciousnessScore });
    }
    
    // Apply consciousness type filtering
    if (options.consciousnessTypes && options.consciousnessTypes.length > 0) {
      query.andWhere('cs.consciousnessLevel IN (:...types)', { types: options.consciousnessTypes });
    }
    
    // Apply time range filtering
    if (options.timeRange) {
      query.andWhere('cs.timestamp BETWEEN :start AND :end', {
        start: options.timeRange.start,
        end: options.timeRange.end
      });
    }
    
    // Add metacognition filtering
    if (options.includeMetacognition) {
      query.andWhere('cs.metacognitionLevel > 0');
    }
    
    // Apply consciousness-based sorting
    if (options.sortByConsciousness) {
      query.orderBy('cs.selfAwarenessScore', 'DESC')
        .addOrderBy('cs.metacognitionLevel', 'DESC')
        .addOrderBy('cs.phenomenalConsciousness', 'DESC');
    } else {
      query.orderBy('cs.timestamp', 'DESC');
    }
    
    // Optimize for consciousness queries
    query.cache(`consciousness_states_${JSON.stringify(options)}`, 300000); // 5 min cache
    
    return await query.getMany();
  }
  
  async getHighConsciousnessEvents(limit: number = 100): Promise<ConsciousnessEvent[]> {
    return await this.dataSource.query(`
      SELECT ce.*, cs."selfAwarenessScore", cs."metacognitionLevel"
      FROM consciousness_events ce
      INNER JOIN consciousness_states cs ON ce."consciousnessStateId" = cs.id
      WHERE ce."consciousnessRelevance" > 0.7
        AND cs."selfAwarenessScore" > 0.6
      ORDER BY ce."consciousnessRelevance" DESC, ce.intensity DESC
      LIMIT $1
    `, [limit]);
  }
  
  async analyzeConsciousnessPatterns(userId?: string): Promise<any> {
    const baseQuery = `
      WITH consciousness_aggregates AS (
        SELECT 
          DATE_TRUNC('hour', timestamp) as hour,
          AVG("selfAwarenessScore") as avg_self_awareness,
          AVG("metacognitionLevel") as avg_metacognition,
          AVG("phenomenalConsciousness") as avg_phenomenal,
          COUNT(*) as state_count
        FROM consciousness_states
        ${userId ? 'WHERE "userId" = $1' : ''}
        GROUP BY DATE_TRUNC('hour', timestamp)
      )
      SELECT 
        hour,
        avg_self_awareness,
        avg_metacognition,
        avg_phenomenal,
        state_count,
        (avg_self_awareness + avg_metacognition + avg_phenomenal) / 3.0 as overall_consciousness
      FROM consciousness_aggregates
      ORDER BY hour DESC
      LIMIT 168
    `;
    
    return await this.dataSource.query(baseQuery, userId ? [userId] : []);
  }
  
  // ========================
  // Vector Similarity Search
  // ========================
  
  async vectorSimilaritySearch(options: VectorSearchOptions): Promise<VectorDocument[]> {
    const { embedding, similarityThreshold = 0.5, limit = 10, consciousnessBoost = 1.0 } = options;
    
    // Use PostgreSQL pgvector extension for efficient similarity search
    const query = `
      WITH vector_similarities AS (
        SELECT 
          vd.*,
          ve.embedding,
          (1 - (ve.embedding <=> $1::vector)) as similarity_score,
          CASE 
            WHEN vd."consciousnessScore" > 0.5 
            THEN (1 - (ve.embedding <=> $1::vector)) * $2
            ELSE (1 - (ve.embedding <=> $1::vector))
          END as boosted_similarity
        FROM vector_documents vd
        INNER JOIN vector_embeddings ve ON vd.id = ve."documentId"
        WHERE (1 - (ve.embedding <=> $1::vector)) >= $3
      )
      SELECT *
      FROM vector_similarities
      ORDER BY boosted_similarity DESC
      LIMIT $4
    `;
    
    const results = await this.dataSource.query(query, [
      JSON.stringify(embedding),
      consciousnessBoost,
      similarityThreshold,
      limit
    ]);
    
    // Convert raw results back to VectorDocument entities
    return results.map(result => {
      const document = new VectorDocument();
      Object.assign(document, result);
      return document;
    });
  }
  
  async findSimilarConsciousnessContent(
    queryEmbedding: number[],
    minConsciousnessScore: number = 0.6,
    limit: number = 5
  ): Promise<VectorDocument[]> {
    return await this.vectorSimilaritySearch({
      embedding: queryEmbedding,
      similarityThreshold: 0.4,
      limit,
      consciousnessBoost: 1.5,
      includeMetadata: true
    });
  }
  
  // ========================
  // Knowledge Graph Queries
  // ========================
  
  async findRelatedEntities(
    entityId: string,
    maxHops: number = 2,
    consciousnessFilter: boolean = false
  ): Promise<any> {
    const query = `
      WITH RECURSIVE entity_paths AS (
        -- Base case: start entity
        SELECT 
          ke.id,
          ke.name,
          ke."entityType",
          ke."consciousnessRelevance",
          0 as hop_count,
          ARRAY[ke.id] as path
        FROM knowledge_entities ke
        WHERE ke."entityId" = $1
        
        UNION ALL
        
        -- Recursive case: follow relationships
        SELECT 
          ke.id,
          ke.name,
          ke."entityType",
          ke."consciousnessRelevance",
          ep.hop_count + 1,
          ep.path || ke.id
        FROM entity_paths ep
        INNER JOIN knowledge_relationships kr ON (
          kr."sourceEntityId" = ep.id OR kr."targetEntityId" = ep.id
        )
        INNER JOIN knowledge_entities ke ON (
          CASE 
            WHEN kr."sourceEntityId" = ep.id THEN ke.id = kr."targetEntityId"
            ELSE ke.id = kr."sourceEntityId"
          END
        )
        WHERE ep.hop_count < $2
          AND NOT ke.id = ANY(ep.path)
          ${consciousnessFilter ? 'AND ke."consciousnessRelevance" > 0.5' : ''}
      )
      SELECT DISTINCT * FROM entity_paths
      ORDER BY hop_count, "consciousnessRelevance" DESC
    `;
    
    return await this.dataSource.query(query, [entityId, maxHops]);
  }
  
  async getConsciousnessEntityGraph(): Promise<any> {
    return await this.dataSource.query(`
      WITH consciousness_entities AS (
        SELECT ke.*
        FROM knowledge_entities ke
        WHERE ke."entityType" = 'consciousness_concept' 
           OR ke."consciousnessRelevance" > 0.7
      ),
      consciousness_relationships AS (
        SELECT kr.*
        FROM knowledge_relationships kr
        INNER JOIN consciousness_entities ce1 ON kr."sourceEntityId" = ce1.id
        INNER JOIN consciousness_entities ce2 ON kr."targetEntityId" = ce2.id
      )
      SELECT 
        json_build_object(
          'entities', (SELECT json_agg(ce.*) FROM consciousness_entities ce),
          'relationships', (SELECT json_agg(cr.*) FROM consciousness_relationships cr)
        ) as graph_data
    `);
  }
  
  // ========================
  // Reflection Analysis
  // ========================
  
  async getReflectionTrends(timeRange: { start: Date; end: Date }): Promise<any> {
    return await this.dataSource.query(`
      WITH reflection_trends AS (
        SELECT 
          DATE_TRUNC('day', timestamp) as day,
          "reflectionType",
          AVG("overallConfidence") as avg_confidence,
          COUNT(*) as reflection_count,
          AVG(CASE WHEN "requiresCorrection" THEN 1.0 ELSE 0.0 END) as correction_rate
        FROM reflection_records
        WHERE timestamp BETWEEN $1 AND $2
        GROUP BY DATE_TRUNC('day', timestamp), "reflectionType"
      )
      SELECT 
        day,
        json_object_agg("reflectionType", json_build_object(
          'avg_confidence', avg_confidence,
          'count', reflection_count,
          'correction_rate', correction_rate
        )) as daily_metrics
      FROM reflection_trends
      GROUP BY day
      ORDER BY day DESC
    `, [timeRange.start, timeRange.end]);
  }
  
  async analyzeMetacognitivePerformance(): Promise<any> {
    return await this.dataSource.query(`
      SELECT 
        rr."reflectionType",
        AVG(rr."overallConfidence") as avg_confidence,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY rr."overallConfidence") as median_confidence,
        COUNT(*) as total_reflections,
        AVG(rr."reflectionTime") as avg_reflection_time,
        COUNT(*) FILTER (WHERE rr."requiresCorrection") as corrections_needed
      FROM reflection_records rr
      WHERE rr."reflectionType" = 'metacognitive'
        AND rr.timestamp > NOW() - INTERVAL '7 days'
      GROUP BY rr."reflectionType"
    `);
  }
  
  // ========================
  // Memory Management
  // ========================
  
  async getMemoryByImportance(
    chunkType: string,
    minImportance: number = 0.5,
    limit: number = 100
  ): Promise<MemoryChunk[]> {
    return await this.memoryChunkRepository
      .createQueryBuilder('mc')
      .where('mc.chunkType = :chunkType', { chunkType })
      .andWhere('mc.importance >= :minImportance', { minImportance })
      .orderBy('mc.importance', 'DESC')
      .addOrderBy('mc.lastAccessed', 'DESC')
      .limit(limit)
      .getMany();
  }
  
  async consolidateMemories(similarityThreshold: number = 0.8): Promise<number> {
    // Find similar memories and consolidate them
    const query = `
      WITH memory_similarities AS (
        SELECT 
          m1.id as id1,
          m2.id as id2,
          (1 - (m1.embedding <=> m2.embedding)) as similarity
        FROM memory_chunks m1
        CROSS JOIN memory_chunks m2
        WHERE m1.id < m2.id
          AND m1."chunkType" = m2."chunkType"
          AND m1.embedding IS NOT NULL
          AND m2.embedding IS NOT NULL
          AND (1 - (m1.embedding <=> m2.embedding)) > $1
      )
      SELECT id1, id2, similarity
      FROM memory_similarities
      ORDER BY similarity DESC
    `;
    
    const similarMemories = await this.dataSource.query(query, [similarityThreshold]);
    
    let consolidationCount = 0;
    
    // Process consolidations in transaction
    await this.dataSource.transaction(async (manager) => {
      for (const { id1, id2 } of similarMemories) {
        const memory1 = await manager.findOne(MemoryChunk, { where: { id: id1 } });
        const memory2 = await manager.findOne(MemoryChunk, { where: { id: id2 } });
        
        if (memory1 && memory2) {
          // Consolidate memories
          memory1.content = `${memory1.content}\n\nConsolidated with: ${memory2.content}`;
          memory1.importance = Math.max(memory1.importance, memory2.importance);
          memory1.accessCount += memory2.accessCount;
          memory1.consciousnessRelevance = Math.max(
            memory1.consciousnessRelevance,
            memory2.consciousnessRelevance
          );
          
          // Merge associations
          const combinedAssociations = [
            ...(memory1.associations || []),
            ...(memory2.associations || [])
          ];
          memory1.associations = [...new Set(combinedAssociations)];
          
          await manager.save(memory1);
          await manager.remove(memory2);
          
          consolidationCount++;
        }
      }
    });
    
    this.logger.log(`Consolidated ${consolidationCount} memory pairs`);
    return consolidationCount;
  }
  
  // ========================
  // Compression Management
  // ========================
  
  async compressEntity(
    entityId: string,
    entityType: string,
    options: CompressionOptions = {}
  ): Promise<CompressionMetadata> {
    const entity = await this.getEntityById(entityId, entityType);
    if (!entity) {
      throw new Error(`Entity not found: ${entityId}`);
    }
    
    const originalData = JSON.stringify(entity);
    const originalSize = Buffer.byteLength(originalData, 'utf8');
    
    let compressedData: Buffer;
    let algorithm = options.algorithm || 'lz4';
    
    // Use consciousness-aware compression if applicable
    if (options.consciousnessPreservation && this.hasConsciousnessData(entity)) {
      algorithm = 'consciousness_aware';
      compressedData = await this.compressionService.compressConsciousnessAware(
        originalData,
        entity
      );
    } else {
      compressedData = await this.compressionService.compress(originalData, algorithm);
    }
    
    const compressedSize = compressedData.length;
    const compressionRatio = 1 - (compressedSize / originalSize);
    
    // Store compression metadata
    const metadata = this.compressionMetadataRepository.create({
      originalId: entityId,
      originalType: entityType,
      originalSize,
      compressedSize,
      compressionRatio,
      compressionAlgorithm: algorithm,
      compressionMetadata: {
        consciousnessPreservation: options.consciousnessPreservation,
        targetRatio: options.targetRatio
      }
    });
    
    return await this.compressionMetadataRepository.save(metadata);
  }
  
  async getCompressionStats(): Promise<any> {
    return await this.dataSource.query(`
      SELECT 
        "originalType",
        "compressionAlgorithm",
        COUNT(*) as item_count,
        AVG("compressionRatio") as avg_compression_ratio,
        SUM("originalSize") as total_original_size,
        SUM("compressedSize") as total_compressed_size,
        (SUM("originalSize") - SUM("compressedSize")) as space_saved
      FROM compression_metadata
      GROUP BY "originalType", "compressionAlgorithm"
      ORDER BY avg_compression_ratio DESC
    `);
  }
  
  // ========================
  // Database Optimization
  // ========================
  
  async createConsciousnessIndexes(): Promise<void> {
    const queries = [
      // Consciousness state indexes
      `CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_consciousness_awareness_score 
       ON consciousness_states ("selfAwarenessScore" DESC) 
       WHERE "selfAwarenessScore" > 0.5`,
      
      `CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_consciousness_metacognition 
       ON consciousness_states ("metacognitionLevel" DESC, "phenomenalConsciousness" DESC)`,
      
      // Consciousness events indexes
      `CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_consciousness_events_relevance 
       ON consciousness_events ("consciousnessRelevance" DESC, intensity DESC)
       WHERE "consciousnessRelevance" > 0.6`,
      
      // Vector documents consciousness index
      `CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_vector_docs_consciousness 
       ON vector_documents ("consciousnessScore" DESC, "relevanceScore" DESC)
       WHERE "consciousnessScore" > 0.3`,
      
      // Knowledge entities consciousness index
      `CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_knowledge_entities_consciousness 
       ON knowledge_entities ("consciousnessRelevance" DESC) 
       WHERE "consciousnessRelevance" > 0.4`
    ];
    
    for (const query of queries) {
      try {
        await this.dataSource.query(query);
        this.logger.debug(`Created consciousness index: ${query.split('\n')[0]}`);
      } catch (error) {
        this.logger.warn(`Failed to create index: ${error.message}`);
      }
    }
  }
  
  async createVectorIndexes(): Promise<void> {
    const queries = [
      // Vector similarity indexes (requires pgvector extension)
      `CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_vector_embeddings_cosine 
       ON vector_embeddings USING ivfflat (embedding vector_cosine_ops)`,
      
      `CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_memory_chunks_embedding 
       ON memory_chunks USING ivfflat (embedding vector_cosine_ops)
       WHERE embedding IS NOT NULL`
    ];
    
    for (const query of queries) {
      try {
        await this.dataSource.query(query);
        this.logger.debug(`Created vector index: ${query.split('\n')[0]}`);
      } catch (error) {
        this.logger.warn(`Failed to create vector index: ${error.message}`);
      }
    }
  }
  
  async initializeCompressionMetadata(): Promise<void> {
    // Initialize compression tracking for existing data
    this.logger.log('Initializing compression metadata tracking');
    
    await this.dataSource.query(`
      INSERT INTO compression_metadata ("originalId", "originalType", "originalSize", "compressedSize", "compressionRatio", "compressionAlgorithm")
      SELECT 
        id::text,
        'consciousness_state',
        LENGTH(("streamOfConsciousness" || COALESCE("emotionalState"::text, ''))::text),
        LENGTH(("streamOfConsciousness" || COALESCE("emotionalState"::text, ''))::text), -- Placeholder, not actually compressed yet
        0.0,
        'none'
      FROM consciousness_states
      WHERE id NOT IN (SELECT "originalId"::uuid FROM compression_metadata WHERE "originalType" = 'consciousness_state')
      ON CONFLICT DO NOTHING
    `);
  }
  
  async optimizeQueryPlans(): Promise<void> {
    // Update table statistics for better query planning
    const tables = [
      'consciousness_states',
      'consciousness_events',
      'vector_documents',
      'vector_embeddings',
      'knowledge_entities',
      'knowledge_relationships',
      'reflection_records',
      'memory_chunks'
    ];
    
    for (const table of tables) {
      try {
        await this.dataSource.query(`ANALYZE ${table}`);
        this.logger.debug(`Updated statistics for table: ${table}`);
      } catch (error) {
        this.logger.warn(`Failed to analyze table ${table}: ${error.message}`);
      }
    }
  }
  
  // ========================
  // Performance Monitoring
  // ========================
  
  async recordPerformanceMetric(
    metricType: string,
    value: number,
    context: any = {}
  ): Promise<void> {
    const metric = this.dataSource.getRepository(PerformanceMetric).create({
      metricType,
      value,
      context,
      timestamp: new Date()
    });
    
    await this.dataSource.getRepository(PerformanceMetric).save(metric);
  }
  
  async getPerformanceMetrics(
    metricType: string,
    timeRange: { start: Date; end: Date }
  ): Promise<PerformanceMetric[]> {
    return await this.dataSource.getRepository(PerformanceMetric)
      .createQueryBuilder('pm')
      .where('pm.metricType = :metricType', { metricType })
      .andWhere('pm.timestamp BETWEEN :start AND :end', timeRange)
      .orderBy('pm.timestamp', 'DESC')
      .getMany();
  }
  
  // ========================
  // Helper Methods
  // ========================
  
  private async getEntityById(id: string, type: string): Promise<any> {
    const repositoryMap = {
      'consciousness_state': this.consciousnessRepository,
      'vector_document': this.vectorDocumentRepository,
      'memory_chunk': this.memoryChunkRepository
    };
    
    const repository = repositoryMap[type];
    if (!repository) {
      throw new Error(`Unknown entity type: ${type}`);
    }
    
    return await repository.findOne({ where: { id } });
  }
  
  private hasConsciousnessData(entity: any): boolean {
    return !!(
      entity.consciousnessScore ||
      entity.selfAwarenessScore ||
      entity.consciousnessRelevance ||
      entity.streamOfConsciousness
    );
  }
}