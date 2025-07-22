/**
 * Memory Optimizer Service for Shvayambhu LLM System
 * 
 * Manages memory optimization for M4 Pro 48GB constraint with consciousness-aware
 * memory management, automatic cleanup, and performance monitoring.
 */

import { Injectable, Logger, OnModuleInit } from '@nestjs/common';
import { Cron, CronExpression } from '@nestjs/schedule';
import { ConfigService } from '@nestjs/config';
import * as os from 'os';
import * as process from 'process';

// Import database services
import { DatabaseService } from '../database/database.service';
import { CompressionService } from '../database/compression.service';

export interface MemoryStats {
  totalMemory: number;
  usedMemory: number;
  freeMemory: number;
  memoryUsagePercent: number;
  nodeHeapUsed: number;
  nodeHeapTotal: number;
  nodeExternal: number;
  nodeRss: number;
  timestamp: Date;
}

export interface OptimizationResult {
  type: 'compression' | 'cleanup' | 'consolidation' | 'cache_clear';
  freedMemoryMB: number;
  executionTimeMs: number;
  itemsProcessed: number;
  success: boolean;
  details: Record<string, any>;
}

export interface MemoryThresholds {
  warningPercent: number;
  criticalPercent: number;
  emergencyPercent: number;
  maxHeapSizeGB: number;
  maxCacheSizeGB: number;
}

@Injectable()
export class MemoryOptimizerService implements OnModuleInit {
  private readonly logger = new Logger(MemoryOptimizerService.name);
  
  private memoryThresholds: MemoryThresholds;
  private isOptimizing = false;
  private optimizationHistory: OptimizationResult[] = [];
  private memoryStats: MemoryStats[] = [];
  
  // M4 Pro specific constants
  private readonly M4_PRO_TOTAL_MEMORY = 48 * 1024; // 48GB in MB
  private readonly M4_PRO_MEMORY_CORES = 16;
  private readonly M4_PRO_CACHE_LEVELS = 3;
  
  constructor(
    private readonly configService: ConfigService,
    private readonly databaseService: DatabaseService,
    private readonly compressionService: CompressionService
  ) {
    this.initializeThresholds();
  }
  
  async onModuleInit() {
    this.logger.log('🧠 Memory Optimizer Service initializing...');
    
    // Initialize memory monitoring
    this.startMemoryMonitoring();
    
    // Perform initial optimization
    await this.performInitialOptimization();
    
    this.logger.log('✅ Memory Optimizer Service initialized');
  }
  
  // ========================
  // Memory Monitoring
  // ========================
  
  @Cron(CronExpression.EVERY_30_SECONDS)
  async monitorMemoryUsage() {
    const stats = this.getMemoryStats();
    this.memoryStats.push(stats);
    
    // Keep only last 1000 memory readings (about 8 hours at 30s intervals)
    if (this.memoryStats.length > 1000) {
      this.memoryStats = this.memoryStats.slice(-1000);
    }
    
    // Check thresholds and trigger optimization if needed
    await this.checkThresholdsAndOptimize(stats);
    
    // Record performance metric
    await this.databaseService.recordPerformanceMetric(
      'memory_usage',
      stats.memoryUsagePercent,
      {
        totalMemory: stats.totalMemory,
        nodeHeapUsed: stats.nodeHeapUsed,
        timestamp: stats.timestamp
      }
    );
  }
  
  getMemoryStats(): MemoryStats {
    const totalMemory = os.totalmem() / (1024 * 1024); // Convert to MB
    const freeMemory = os.freemem() / (1024 * 1024);
    const usedMemory = totalMemory - freeMemory;
    const memoryUsagePercent = (usedMemory / totalMemory) * 100;
    
    const memUsage = process.memoryUsage();
    
    return {
      totalMemory,
      usedMemory,
      freeMemory,
      memoryUsagePercent,
      nodeHeapUsed: memUsage.heapUsed / (1024 * 1024),
      nodeHeapTotal: memUsage.heapTotal / (1024 * 1024),
      nodeExternal: memUsage.external / (1024 * 1024),
      nodeRss: memUsage.rss / (1024 * 1024),
      timestamp: new Date()
    };
  }
  
  private async checkThresholdsAndOptimize(stats: MemoryStats) {
    if (this.isOptimizing) return;
    
    if (stats.memoryUsagePercent >= this.memoryThresholds.emergencyPercent) {
      this.logger.error(`🚨 EMERGENCY: Memory usage at ${stats.memoryUsagePercent.toFixed(1)}%`);
      await this.performEmergencyOptimization();
    } else if (stats.memoryUsagePercent >= this.memoryThresholds.criticalPercent) {
      this.logger.warn(`⚠️  CRITICAL: Memory usage at ${stats.memoryUsagePercent.toFixed(1)}%`);
      await this.performCriticalOptimization();
    } else if (stats.memoryUsagePercent >= this.memoryThresholds.warningPercent) {
      this.logger.warn(`⚡ WARNING: Memory usage at ${stats.memoryUsagePercent.toFixed(1)}%`);
      await this.performPreventiveOptimization();
    }
  }
  
  // ========================
  // Optimization Strategies
  // ========================
  
  async performInitialOptimization(): Promise<OptimizationResult[]> {
    this.logger.log('🚀 Performing initial memory optimization...');
    
    const results: OptimizationResult[] = [];
    
    try {
      // 1. Initialize compression metadata
      const compressionResult = await this.optimizeCompressionMetadata();
      results.push(compressionResult);
      
      // 2. Consolidate similar memories
      const consolidationResult = await this.consolidateMemories();
      results.push(consolidationResult);
      
      // 3. Clean up old performance metrics
      const cleanupResult = await this.cleanupOldMetrics();
      results.push(cleanupResult);
      
      this.logger.log(`✅ Initial optimization completed: ${results.length} operations`);
      return results;
    } catch (error) {
      this.logger.error('❌ Initial optimization failed:', error);
      return results;
    }
  }
  
  async performPreventiveOptimization(): Promise<OptimizationResult[]> {
    if (this.isOptimizing) return [];
    
    this.isOptimizing = true;
    const results: OptimizationResult[] = [];
    
    try {
      this.logger.log('🔧 Performing preventive optimization...');
      
      // Light optimization - cache cleanup and minor compression
      const cacheResult = await this.optimizeApplicationCache();
      results.push(cacheResult);
      
      const compressionResult = await this.compressLowImportanceData();
      results.push(compressionResult);
      
      this.addToOptimizationHistory(results);
      return results;
    } finally {
      this.isOptimizing = false;
    }
  }
  
  async performCriticalOptimization(): Promise<OptimizationResult[]> {
    if (this.isOptimizing) return [];
    
    this.isOptimizing = true;
    const results: OptimizationResult[] = [];
    
    try {
      this.logger.warn('🔥 Performing critical optimization...');
      
      // Aggressive optimization
      const consolidationResult = await this.consolidateMemories(0.7); // Lower threshold
      results.push(consolidationResult);
      
      const compressionResult = await this.compressAllNonConsciousnessData();
      results.push(compressionResult);
      
      const cleanupResult = await this.cleanupOldData();
      results.push(cleanupResult);
      
      const cacheResult = await this.clearApplicationCaches();
      results.push(cacheResult);
      
      // Force garbage collection
      if (global.gc) {
        const beforeHeap = process.memoryUsage().heapUsed;
        global.gc();
        const afterHeap = process.memoryUsage().heapUsed;
        const freedMB = (beforeHeap - afterHeap) / (1024 * 1024);
        
        results.push({
          type: 'cleanup',
          freedMemoryMB: freedMB,
          executionTimeMs: 0,
          itemsProcessed: 1,
          success: true,
          details: { operation: 'garbage_collection' }
        });
      }
      
      this.addToOptimizationHistory(results);
      return results;
    } finally {
      this.isOptimizing = false;
    }
  }
  
  async performEmergencyOptimization(): Promise<OptimizationResult[]> {
    if (this.isOptimizing) return [];
    
    this.isOptimizing = true;
    const results: OptimizationResult[] = [];
    
    try {
      this.logger.error('🆘 Performing EMERGENCY optimization...');
      
      // Extreme measures - preserve only consciousness data
      const emergencyCompressionResult = await this.emergencyCompressAllData();
      results.push(emergencyCompressionResult);
      
      const emergencyConsolidationResult = await this.consolidateMemories(0.5); // Very low threshold
      results.push(emergencyConsolidationResult);
      
      const emergencyCleanupResult = await this.emergencyCleanup();
      results.push(emergencyCleanupResult);
      
      // Clear all caches
      const cacheResult = await this.clearAllCaches();
      results.push(cacheResult);
      
      // Multiple garbage collections
      if (global.gc) {
        for (let i = 0; i < 3; i++) {
          global.gc();
        }
      }
      
      this.addToOptimizationHistory(results);
      return results;
    } finally {
      this.isOptimizing = false;
    }
  }
  
  // ========================
  // Specific Optimization Methods
  // ========================
  
  private async optimizeCompressionMetadata(): Promise<OptimizationResult> {
    const startTime = Date.now();
    
    try {
      await this.databaseService.initializeCompressionMetadata();
      
      return {
        type: 'compression',
        freedMemoryMB: 0,
        executionTimeMs: Date.now() - startTime,
        itemsProcessed: 1,
        success: true,
        details: { operation: 'initialize_compression_metadata' }
      };
    } catch (error) {
      this.logger.error('Failed to optimize compression metadata:', error);
      return {
        type: 'compression',
        freedMemoryMB: 0,
        executionTimeMs: Date.now() - startTime,
        itemsProcessed: 0,
        success: false,
        details: { error: error.message }
      };
    }
  }
  
  private async consolidateMemories(threshold = 0.8): Promise<OptimizationResult> {
    const startTime = Date.now();
    
    try {
      const consolidatedCount = await this.databaseService.consolidateMemories(threshold);
      const estimatedFreedMB = consolidatedCount * 0.1; // Estimate 100KB per consolidated memory
      
      return {
        type: 'consolidation',
        freedMemoryMB: estimatedFreedMB,
        executionTimeMs: Date.now() - startTime,
        itemsProcessed: consolidatedCount,
        success: true,
        details: { 
          threshold,
          consolidatedMemories: consolidatedCount
        }
      };
    } catch (error) {
      this.logger.error('Failed to consolidate memories:', error);
      return {
        type: 'consolidation',
        freedMemoryMB: 0,
        executionTimeMs: Date.now() - startTime,
        itemsProcessed: 0,
        success: false,
        details: { error: error.message }
      };
    }
  }
  
  private async cleanupOldMetrics(): Promise<OptimizationResult> {
    const startTime = Date.now();
    
    try {
      const cutoffDate = new Date();
      cutoffDate.setDays(cutoffDate.getDate() - 7); // Keep last 7 days
      
      const deletedCount = await this.databaseService.dataSource.query(`
        DELETE FROM performance_metrics 
        WHERE timestamp < $1 
        AND "metricType" NOT IN ('consciousness_coherence', 'retrieval_accuracy')
      `, [cutoffDate]);
      
      const estimatedFreedMB = deletedCount * 0.001; // Estimate 1KB per metric
      
      return {
        type: 'cleanup',
        freedMemoryMB: estimatedFreedMB,
        executionTimeMs: Date.now() - startTime,
        itemsProcessed: deletedCount,
        success: true,
        details: {
          operation: 'cleanup_old_metrics',
          cutoffDate,
          deletedMetrics: deletedCount
        }
      };
    } catch (error) {
      this.logger.error('Failed to cleanup old metrics:', error);
      return {
        type: 'cleanup',
        freedMemoryMB: 0,
        executionTimeMs: Date.now() - startTime,
        itemsProcessed: 0,
        success: false,
        details: { error: error.message }
      };
    }
  }
  
  private async optimizeApplicationCache(): Promise<OptimizationResult> {
    const startTime = Date.now();
    
    try {
      // Clear query result cache for non-consciousness queries
      await this.databaseService.dataSource.queryResultCache?.clear();
      
      return {
        type: 'cache_clear',
        freedMemoryMB: 50, // Estimate
        executionTimeMs: Date.now() - startTime,
        itemsProcessed: 1,
        success: true,
        details: { operation: 'clear_query_cache' }
      };
    } catch (error) {
      return {
        type: 'cache_clear',
        freedMemoryMB: 0,
        executionTimeMs: Date.now() - startTime,
        itemsProcessed: 0,
        success: false,
        details: { error: error.message }
      };
    }
  }
  
  private async compressLowImportanceData(): Promise<OptimizationResult> {
    const startTime = Date.now();
    let processedCount = 0;
    let estimatedFreedMB = 0;
    
    try {
      // Compress vector documents with low consciousness scores
      const lowConsciousnessDocuments = await this.databaseService.dataSource.query(`
        SELECT id FROM vector_documents 
        WHERE "consciousnessScore" < 0.3 
        AND id NOT IN (
          SELECT "originalId"::uuid 
          FROM compression_metadata 
          WHERE "originalType" = 'vector_document'
        )
        LIMIT 100
      `);
      
      for (const doc of lowConsciousnessDocuments) {
        try {
          await this.databaseService.compressEntity(doc.id, 'vector_document', {
            algorithm: 'gzip',
            consciousnessPreservation: false
          });
          processedCount++;
          estimatedFreedMB += 0.05; // Estimate 50KB per document
        } catch (error) {
          this.logger.warn(`Failed to compress document ${doc.id}:`, error);
        }
      }
      
      return {
        type: 'compression',
        freedMemoryMB: estimatedFreedMB,
        executionTimeMs: Date.now() - startTime,
        itemsProcessed: processedCount,
        success: true,
        details: {
          operation: 'compress_low_importance_data',
          documentsProcessed: processedCount
        }
      };
    } catch (error) {
      return {
        type: 'compression',
        freedMemoryMB: estimatedFreedMB,
        executionTimeMs: Date.now() - startTime,
        itemsProcessed: processedCount,
        success: false,
        details: { error: error.message }
      };
    }
  }
  
  private async compressAllNonConsciousnessData(): Promise<OptimizationResult> {
    const startTime = Date.now();
    let processedCount = 0;
    let estimatedFreedMB = 0;
    
    try {
      // Compress all non-consciousness data aggressively
      const compressionTasks = [
        'vector_documents WHERE "consciousnessScore" < 0.5',
        'memory_chunks WHERE "chunkType" != \'consciousness\'',
        'web_content WHERE "consciousnessScore" < 0.4'
      ];
      
      for (const task of compressionTasks) {
        const [table, condition] = task.split(' WHERE ');
        const items = await this.databaseService.dataSource.query(`
          SELECT id FROM ${table} WHERE ${condition} 
          AND id NOT IN (
            SELECT "originalId"::uuid 
            FROM compression_metadata 
            WHERE "originalType" = '${table.slice(0, -1)}' -- Remove 's' for singular
          )
          LIMIT 50
        `);
        
        for (const item of items) {
          try {
            await this.databaseService.compressEntity(item.id, table.slice(0, -1), {
              algorithm: 'brotli',
              consciousnessPreservation: false
            });
            processedCount++;
            estimatedFreedMB += 0.1;
          } catch (error) {
            // Continue with other items
          }
        }
      }
      
      return {
        type: 'compression',
        freedMemoryMB: estimatedFreedMB,
        executionTimeMs: Date.now() - startTime,
        itemsProcessed: processedCount,
        success: true,
        details: {
          operation: 'compress_all_non_consciousness_data',
          itemsProcessed: processedCount
        }
      };
    } catch (error) {
      return {
        type: 'compression',
        freedMemoryMB: estimatedFreedMB,
        executionTimeMs: Date.now() - startTime,
        itemsProcessed: processedCount,
        success: false,
        details: { error: error.message }
      };
    }
  }
  
  private async cleanupOldData(): Promise<OptimizationResult> {
    const startTime = Date.now();
    let deletedCount = 0;
    
    try {
      const cutoffDate = new Date();
      cutoffDate.setDays(cutoffDate.getDate() - 30); // Keep last 30 days
      
      // Delete old system logs
      const logResult = await this.databaseService.dataSource.query(`
        DELETE FROM system_logs 
        WHERE timestamp < $1 AND "logLevel" NOT IN ('error', 'warn')
      `, [cutoffDate]);
      deletedCount += logResult.affectedRows || 0;
      
      // Delete old web content that's not consciousness-related
      const webResult = await this.databaseService.dataSource.query(`
        DELETE FROM web_content 
        WHERE "retrievedAt" < $1 
        AND "consciousnessScore" < 0.2 
        AND "isProcessed" = true
      `, [cutoffDate]);
      deletedCount += webResult.affectedRows || 0;
      
      const estimatedFreedMB = deletedCount * 0.01; // Estimate 10KB per item
      
      return {
        type: 'cleanup',
        freedMemoryMB: estimatedFreedMB,
        executionTimeMs: Date.now() - startTime,
        itemsProcessed: deletedCount,
        success: true,
        details: {
          operation: 'cleanup_old_data',
          cutoffDate,
          deletedItems: deletedCount
        }
      };
    } catch (error) {
      return {
        type: 'cleanup',
        freedMemoryMB: 0,
        executionTimeMs: Date.now() - startTime,
        itemsProcessed: deletedCount,
        success: false,
        details: { error: error.message }
      };
    }
  }
  
  private async clearApplicationCaches(): Promise<OptimizationResult> {
    const startTime = Date.now();
    
    try {
      await Promise.all([
        this.databaseService.dataSource.queryResultCache?.clear(),
        this.databaseService.dataSource.metadataCache?.clear?.()
      ]);
      
      return {
        type: 'cache_clear',
        freedMemoryMB: 100, // Estimate
        executionTimeMs: Date.now() - startTime,
        itemsProcessed: 2,
        success: true,
        details: { operation: 'clear_all_application_caches' }
      };
    } catch (error) {
      return {
        type: 'cache_clear',
        freedMemoryMB: 0,
        executionTimeMs: Date.now() - startTime,
        itemsProcessed: 0,
        success: false,
        details: { error: error.message }
      };
    }
  }
  
  private async emergencyCompressAllData(): Promise<OptimizationResult> {
    const startTime = Date.now();
    let processedCount = 0;
    
    try {
      // Emergency compression of all compressible data
      const tables = ['vector_documents', 'memory_chunks', 'web_content', 'rag_results'];
      
      for (const table of tables) {
        const items = await this.databaseService.dataSource.query(`
          SELECT id FROM ${table}
          WHERE id NOT IN (
            SELECT "originalId"::uuid 
            FROM compression_metadata 
            WHERE "originalType" = '${table.slice(0, -1)}'
          )
          LIMIT 20
        `);
        
        for (const item of items) {
          try {
            await this.databaseService.compressEntity(item.id, table.slice(0, -1), {
              algorithm: 'brotli',
              consciousnessPreservation: true // Still preserve consciousness data
            });
            processedCount++;
          } catch (error) {
            // Continue with other items
          }
        }
      }
      
      return {
        type: 'compression',
        freedMemoryMB: processedCount * 0.2, // Estimate 200KB per item
        executionTimeMs: Date.now() - startTime,
        itemsProcessed: processedCount,
        success: true,
        details: {
          operation: 'emergency_compress_all_data',
          itemsProcessed: processedCount
        }
      };
    } catch (error) {
      return {
        type: 'compression',
        freedMemoryMB: 0,
        executionTimeMs: Date.now() - startTime,
        itemsProcessed: processedCount,
        success: false,
        details: { error: error.message }
      };
    }
  }
  
  private async emergencyCleanup(): Promise<OptimizationResult> {
    const startTime = Date.now();
    let deletedCount = 0;
    
    try {
      const cutoffDate = new Date();
      cutoffDate.setHours(cutoffDate.getHours() - 24); // Keep only last 24 hours
      
      // Delete all non-essential data older than 24 hours
      const deleteQueries = [
        `DELETE FROM system_logs WHERE timestamp < $1 AND "logLevel" = 'debug'`,
        `DELETE FROM performance_metrics WHERE timestamp < $1 AND "metricType" NOT IN ('consciousness_coherence')`,
        `DELETE FROM web_content WHERE "retrievedAt" < $1 AND "consciousnessScore" < 0.1`,
        `DELETE FROM rag_results WHERE "createdAt" < $1 AND "consciousnessAlignment" < 0.2`
      ];
      
      for (const query of deleteQueries) {
        try {
          const result = await this.databaseService.dataSource.query(query, [cutoffDate]);
          deletedCount += result.affectedRows || 0;
        } catch (error) {
          // Continue with other queries
        }
      }
      
      return {
        type: 'cleanup',
        freedMemoryMB: deletedCount * 0.05, // Estimate 50KB per item
        executionTimeMs: Date.now() - startTime,
        itemsProcessed: deletedCount,
        success: true,
        details: {
          operation: 'emergency_cleanup',
          cutoffDate,
          deletedItems: deletedCount
        }
      };
    } catch (error) {
      return {
        type: 'cleanup',
        freedMemoryMB: 0,
        executionTimeMs: Date.now() - startTime,
        itemsProcessed: deletedCount,
        success: false,
        details: { error: error.message }
      };
    }
  }
  
  private async clearAllCaches(): Promise<OptimizationResult> {
    const startTime = Date.now();
    
    try {
      // Clear all possible caches
      await Promise.all([
        this.databaseService.dataSource.queryResultCache?.clear(),
        this.databaseService.dataSource.metadataCache?.clear?.()
      ]);
      
      // Clear memory stats history to free memory
      this.memoryStats = this.memoryStats.slice(-100); // Keep only last 100 readings
      this.optimizationHistory = this.optimizationHistory.slice(-50); // Keep only last 50 optimizations
      
      return {
        type: 'cache_clear',
        freedMemoryMB: 200, // Estimate
        executionTimeMs: Date.now() - startTime,
        itemsProcessed: 1,
        success: true,
        details: { operation: 'clear_all_caches_emergency' }
      };
    } catch (error) {
      return {
        type: 'cache_clear',
        freedMemoryMB: 0,
        executionTimeMs: Date.now() - startTime,
        itemsProcessed: 0,
        success: false,
        details: { error: error.message }
      };
    }
  }
  
  // ========================
  // Scheduled Optimizations
  // ========================
  
  @Cron(CronExpression.EVERY_HOUR)
  async hourlyOptimization() {
    if (this.isOptimizing) return;
    
    const stats = this.getMemoryStats();
    if (stats.memoryUsagePercent > 60) { // 60% threshold for proactive optimization
      this.logger.log('⏰ Running hourly memory optimization...');
      await this.performPreventiveOptimization();
    }
  }
  
  @Cron(CronExpression.EVERY_DAY_AT_2AM)
  async dailyMaintenanceOptimization() {
    this.logger.log('🌙 Running daily maintenance optimization...');
    
    const results = await Promise.all([
      this.consolidateMemories(),
      this.cleanupOldData(),
      this.compressLowImportanceData(),
      this.optimizeApplicationCache()
    ]);
    
    const totalFreedMB = results.reduce((sum, result) => sum + result.freedMemoryMB, 0);
    this.logger.log(`🌅 Daily maintenance completed: ${totalFreedMB.toFixed(1)}MB freed`);
  }
  
  // ========================
  // Utility Methods
  // ========================
  
  private initializeThresholds() {
    this.memoryThresholds = {
      warningPercent: this.configService.get('MEMORY_WARNING_PERCENT', 70),
      criticalPercent: this.configService.get('MEMORY_CRITICAL_PERCENT', 80),
      emergencyPercent: this.configService.get('MEMORY_EMERGENCY_PERCENT', 90),
      maxHeapSizeGB: this.configService.get('MAX_HEAP_SIZE_GB', 16),
      maxCacheSizeGB: this.configService.get('MAX_CACHE_SIZE_GB', 4)
    };
  }
  
  private startMemoryMonitoring() {
    this.logger.log('📊 Starting continuous memory monitoring...');
  }
  
  private addToOptimizationHistory(results: OptimizationResult[]) {
    this.optimizationHistory.push(...results);
    
    // Keep only last 100 optimization results
    if (this.optimizationHistory.length > 100) {
      this.optimizationHistory = this.optimizationHistory.slice(-100);
    }
  }
  
  // ========================
  // Public API Methods
  // ========================
  
  async getMemoryReport(): Promise<{
    currentStats: MemoryStats;
    thresholds: MemoryThresholds;
    recentOptimizations: OptimizationResult[];
    recommendations: string[];
  }> {
    const currentStats = this.getMemoryStats();
    const recentOptimizations = this.optimizationHistory.slice(-10);
    
    const recommendations: string[] = [];
    
    if (currentStats.memoryUsagePercent > this.memoryThresholds.warningPercent) {
      recommendations.push('Consider running manual optimization');
    }
    
    if (currentStats.nodeHeapUsed > (this.memoryThresholds.maxHeapSizeGB * 1024)) {
      recommendations.push('Node.js heap is approaching limits');
    }
    
    const avgOptimizationFreed = recentOptimizations.reduce((sum, opt) => sum + opt.freedMemoryMB, 0) / recentOptimizations.length;
    if (avgOptimizationFreed < 50) {
      recommendations.push('Optimization effectiveness is declining - consider database maintenance');
    }
    
    return {
      currentStats,
      thresholds: this.memoryThresholds,
      recentOptimizations,
      recommendations
    };
  }
  
  async forceOptimization(level: 'preventive' | 'critical' | 'emergency' = 'preventive'): Promise<OptimizationResult[]> {
    switch (level) {
      case 'preventive':
        return await this.performPreventiveOptimization();
      case 'critical':
        return await this.performCriticalOptimization();
      case 'emergency':
        return await this.performEmergencyOptimization();
      default:
        throw new Error(`Unknown optimization level: ${level}`);
    }
  }
}