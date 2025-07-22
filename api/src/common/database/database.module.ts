/**
 * Database Module for Shvayambhu LLM System
 * 
 * Configures TypeORM, database connections, and provides repositories
 * for all system entities with compression and optimization features.
 */

import { Module, Global } from '@nestjs/common';
import { TypeOrmModule } from '@nestjs/typeorm';
import { ConfigModule, ConfigService } from '@nestjs/config';

// Import all entities
import {
  // Core entities
  User,
  Session,
  
  // Consciousness entities
  ConsciousnessState,
  ConsciousnessEvent,
  
  // Vector database entities
  VectorCollection,
  VectorDocument,
  VectorEmbedding,
  
  // Knowledge graph entities
  KnowledgeEntity,
  KnowledgeRelationship,
  
  // RAG system entities
  RAGQuery,
  RAGResult,
  
  // Reflection system entities
  ReflectionRecord,
  ReflectionToken,
  
  // Training entities
  TrainingSession,
  TrainingStep,
  
  // Web intelligence entities
  WebSource,
  WebContent,
  
  // Memory and storage entities
  MemoryChunk,
  CompressionMetadata,
  
  // System monitoring entities
  PerformanceMetric,
  SystemLog
} from './database.schema';

// Import services
import { DatabaseService } from './database.service';
import { DatabaseHealthService } from './database-health.service';
import { CompressionService } from './compression.service';
import { QueryOptimizationService } from './query-optimization.service';

@Global()
@Module({
  imports: [
    ConfigModule,
    TypeOrmModule.forRootAsync({
      imports: [ConfigModule],
      useFactory: (configService: ConfigService) => ({
        type: 'postgres',
        host: configService.get('DATABASE_HOST', 'localhost'),
        port: configService.get('DATABASE_PORT', 5432),
        username: configService.get('DATABASE_USERNAME', 'shvayambhu'),
        password: configService.get('DATABASE_PASSWORD'),
        database: configService.get('DATABASE_NAME', 'shvayambhu'),
        
        // Performance optimizations
        extra: {
          // Connection pool configuration
          max: configService.get('DATABASE_POOL_SIZE', 20),
          min: configService.get('DATABASE_POOL_MIN', 5),
          idleTimeoutMillis: 30000,
          connectionTimeoutMillis: 2000,
          
          // Query optimizations
          statement_timeout: '30s',
          idle_in_transaction_session_timeout: '5s',
          
          // Memory optimizations for M4 Pro
          shared_buffers: '256MB',
          effective_cache_size: '4GB',
          work_mem: '64MB',
          maintenance_work_mem: '512MB',
          
          // WAL and checkpoint settings
          checkpoint_completion_target: 0.9,
          wal_buffers: '16MB',
          max_wal_size: '2GB',
          min_wal_size: '512MB',
          
          // Consciousness-aware query planning
          enable_nestloop: true,
          enable_hashjoin: true,
          enable_mergejoin: true,
          random_page_cost: 1.5, // SSD optimized
          
          // JSON/JSONB optimizations
          gin_pending_list_limit: '4MB'
        },
        
        entities: [
          // Core entities
          User,
          Session,
          
          // Consciousness entities
          ConsciousnessState,
          ConsciousnessEvent,
          
          // Vector database entities
          VectorCollection,
          VectorDocument,
          VectorEmbedding,
          
          // Knowledge graph entities
          KnowledgeEntity,
          KnowledgeRelationship,
          
          // RAG system entities
          RAGQuery,
          RAGResult,
          
          // Reflection system entities
          ReflectionRecord,
          ReflectionToken,
          
          // Training entities
          TrainingSession,
          TrainingStep,
          
          // Web intelligence entities
          WebSource,
          WebContent,
          
          // Memory and storage entities
          MemoryChunk,
          CompressionMetadata,
          
          // System monitoring entities
          PerformanceMetric,
          SystemLog
        ],
        
        // Migration settings
        synchronize: configService.get('NODE_ENV') === 'development',
        migrations: ['src/database/migrations/*.ts'],
        migrationsRun: configService.get('DATABASE_RUN_MIGRATIONS', false),
        
        // Logging
        logging: configService.get('DATABASE_LOGGING', false),
        logger: 'advanced-console',
        
        // Cache configuration
        cache: {
          type: 'redis',
          options: {
            host: configService.get('REDIS_HOST', 'localhost'),
            port: configService.get('REDIS_PORT', 6379),
            password: configService.get('REDIS_PASSWORD'),
            db: configService.get('REDIS_DB', 0)
          },
          duration: 300000, // 5 minutes cache
          ignoreErrors: true
        },
        
        // Entity metadata caching
        metadataCache: {
          type: 'redis',
          options: {
            host: configService.get('REDIS_HOST', 'localhost'),
            port: configService.get('REDIS_PORT', 6379),
            password: configService.get('REDIS_PASSWORD'),
            db: configService.get('REDIS_METADATA_DB', 1)
          }
        }
      }),
      inject: [ConfigService]
    }),
    
    // Feature modules for specific entity groups
    TypeOrmModule.forFeature([
      // Core entities
      User,
      Session,
      
      // Consciousness entities
      ConsciousnessState,
      ConsciousnessEvent,
      
      // Vector database entities
      VectorCollection,
      VectorDocument,
      VectorEmbedding,
      
      // Knowledge graph entities
      KnowledgeEntity,
      KnowledgeRelationship,
      
      // RAG system entities
      RAGQuery,
      RAGResult,
      
      // Reflection system entities
      ReflectionRecord,
      ReflectionToken,
      
      // Training entities
      TrainingSession,
      TrainingStep,
      
      // Web intelligence entities
      WebSource,
      WebContent,
      
      // Memory and storage entities
      MemoryChunk,
      CompressionMetadata,
      
      // System monitoring entities
      PerformanceMetric,
      SystemLog
    ])
  ],
  providers: [
    DatabaseService,
    DatabaseHealthService,
    CompressionService,
    QueryOptimizationService
  ],
  exports: [
    TypeOrmModule,
    DatabaseService,
    DatabaseHealthService,
    CompressionService,
    QueryOptimizationService
  ]
})
export class DatabaseModule {
  constructor(private readonly databaseService: DatabaseService) {
    // Initialize database optimizations on module load
    this.initializeDatabaseOptimizations();
  }
  
  private async initializeDatabaseOptimizations() {
    try {
      // Create consciousness-aware indexes
      await this.databaseService.createConsciousnessIndexes();
      
      // Setup vector similarity indexes
      await this.databaseService.createVectorIndexes();
      
      // Initialize compression metadata
      await this.databaseService.initializeCompressionMetadata();
      
      // Setup query plan optimization
      await this.databaseService.optimizeQueryPlans();
      
      console.log('✅ Database optimizations initialized successfully');
    } catch (error) {
      console.error('❌ Failed to initialize database optimizations:', error);
    }
  }
}