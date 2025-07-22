/**
 * Memory Management GraphQL Resolver for Shvayambhu LLM System
 * 
 * Provides GraphQL interface for memory optimization, health monitoring,
 * and system diagnostics with consciousness-aware memory management.
 */

import { Resolver, Query, Mutation, Args, Subscription, Context, Info } from '@nestjs/graphql';
import { UseGuards, Logger } from '@nestjs/common';
import { PubSub } from 'graphql-subscriptions';
import { GraphQLJSONObject } from 'graphql-type-json';
import { GraphQLResolveInfo } from 'graphql';

// Import services
import { MemoryOptimizerService, MemoryStats, OptimizationResult } from './memory-optimizer.service';
import { MemoryHealthService, SystemHealth, HealthAlert, MemoryPrediction } from './memory-health.service';

// Import guards and types
import { AuthGuard } from '../auth/auth.guard';

// GraphQL Types
import { ObjectType, Field, Float, Int, InputType } from '@nestjs/graphql';

@ObjectType()
class MemoryStatsType {
  @Field(() => Float)
  totalMemory: number;

  @Field(() => Float)
  usedMemory: number;

  @Field(() => Float)
  freeMemory: number;

  @Field(() => Float)
  memoryUsagePercent: number;

  @Field(() => Float)
  nodeHeapUsed: number;

  @Field(() => Float)
  nodeHeapTotal: number;

  @Field(() => Float)
  nodeExternal: number;

  @Field(() => Float)
  nodeRss: number;

  @Field()
  timestamp: Date;
}

@ObjectType()
class OptimizationResultType {
  @Field()
  type: string;

  @Field(() => Float)
  freedMemoryMB: number;

  @Field(() => Int)
  executionTimeMs: number;

  @Field(() => Int)
  itemsProcessed: number;

  @Field()
  success: boolean;

  @Field(() => GraphQLJSONObject)
  details: Record<string, any>;
}

@ObjectType()
class MemoryReportType {
  @Field(() => MemoryStatsType)
  currentStats: MemoryStatsType;

  @Field(() => GraphQLJSONObject)
  thresholds: Record<string, any>;

  @Field(() => [OptimizationResultType])
  recentOptimizations: OptimizationResultType[];

  @Field(() => [String])
  recommendations: string[];
}

@ObjectType()
class SystemHealthType {
  @Field()
  overall: string;

  @Field(() => GraphQLJSONObject)
  memory: Record<string, any>;

  @Field(() => GraphQLJSONObject)
  performance: Record<string, any>;

  @Field(() => GraphQLJSONObject)
  optimization: Record<string, any>;

  @Field(() => GraphQLJSONObject)
  alerts: Record<string, any>;
}

@ObjectType()
class HealthAlertType {
  @Field()
  id: string;

  @Field()
  type: string;

  @Field()
  severity: string;

  @Field()
  message: string;

  @Field(() => MemoryStatsType)
  metrics: MemoryStatsType;

  @Field()
  timestamp: Date;

  @Field()
  resolved: boolean;

  @Field({ nullable: true })
  resolvedAt?: Date;

  @Field(() => [String])
  actions: string[];
}

@ObjectType()
class MemoryPredictionType {
  @Field(() => Float)
  predictedUsagePercent: number;

  @Field({ nullable: true })
  timeToExhaustion?: Date;

  @Field(() => Float)
  confidence: number;

  @Field(() => [String])
  recommendations: string[];

  @Field(() => GraphQLJSONObject)
  triggerPoints: Record<string, Date>;
}

@ObjectType()
class DiagnosticsResultType {
  @Field(() => GraphQLJSONObject)
  memoryHealth: Record<string, any>;

  @Field(() => GraphQLJSONObject)
  performanceHealth: Record<string, any>;

  @Field(() => GraphQLJSONObject)
  systemHealth: Record<string, any>;

  @Field(() => [String])
  recommendations: string[];
}

@InputType()
class OptimizationInput {
  @Field({ defaultValue: 'preventive' })
  level: string;

  @Field({ defaultValue: false })
  force: boolean;
}

@Resolver()
export class MemoryResolver {
  private readonly logger = new Logger(MemoryResolver.name);
  private readonly pubSub = new PubSub();

  constructor(
    private readonly memoryOptimizer: MemoryOptimizerService,
    private readonly memoryHealth: MemoryHealthService
  ) {}

  // ========================
  // Queries
  // ========================

  @Query(() => MemoryStatsType)
  @UseGuards(AuthGuard)
  async getMemoryStats(
    @Context() context?: any,
    @Info() info?: GraphQLResolveInfo
  ): Promise<MemoryStatsType> {
    this.logger.debug('Fetching current memory statistics');
    
    const stats = this.memoryOptimizer.getMemoryStats();
    
    return {
      totalMemory: stats.totalMemory,
      usedMemory: stats.usedMemory,
      freeMemory: stats.freeMemory,
      memoryUsagePercent: stats.memoryUsagePercent,
      nodeHeapUsed: stats.nodeHeapUsed,
      nodeHeapTotal: stats.nodeHeapTotal,
      nodeExternal: stats.nodeExternal,
      nodeRss: stats.nodeRss,
      timestamp: stats.timestamp
    };
  }

  @Query(() => MemoryReportType)
  @UseGuards(AuthGuard)
  async getMemoryReport(): Promise<MemoryReportType> {
    this.logger.debug('Generating memory report');
    
    const report = await this.memoryOptimizer.getMemoryReport();
    
    return {
      currentStats: {
        totalMemory: report.currentStats.totalMemory,
        usedMemory: report.currentStats.usedMemory,
        freeMemory: report.currentStats.freeMemory,
        memoryUsagePercent: report.currentStats.memoryUsagePercent,
        nodeHeapUsed: report.currentStats.nodeHeapUsed,
        nodeHeapTotal: report.currentStats.nodeHeapTotal,
        nodeExternal: report.currentStats.nodeExternal,
        nodeRss: report.currentStats.nodeRss,
        timestamp: report.currentStats.timestamp
      },
      thresholds: report.thresholds,
      recentOptimizations: report.recentOptimizations.map(opt => ({
        type: opt.type,
        freedMemoryMB: opt.freedMemoryMB,
        executionTimeMs: opt.executionTimeMs,
        itemsProcessed: opt.itemsProcessed,
        success: opt.success,
        details: opt.details
      })),
      recommendations: report.recommendations
    };
  }

  @Query(() => SystemHealthType)
  @UseGuards(AuthGuard)
  async getSystemHealth(): Promise<SystemHealthType> {
    this.logger.debug('Fetching system health status');
    
    const health = await this.memoryHealth.getSystemHealth();
    
    return {
      overall: health.overall,
      memory: health.memory,
      performance: health.performance,
      optimization: health.optimization,
      alerts: health.alerts
    };
  }

  @Query(() => MemoryPredictionType, { nullable: true })
  @UseGuards(AuthGuard)
  async getMemoryPrediction(): Promise<MemoryPredictionType | null> {
    this.logger.debug('Generating memory usage prediction');
    
    const prediction = await this.memoryHealth.getMemoryPrediction();
    
    if (!prediction) {
      return null;
    }
    
    return {
      predictedUsagePercent: prediction.predictedUsagePercent,
      timeToExhaustion: prediction.timeToExhaustion,
      confidence: prediction.confidence,
      recommendations: prediction.recommendations,
      triggerPoints: prediction.triggerPoints
    };
  }

  @Query(() => [HealthAlertType])
  @UseGuards(AuthGuard)
  async getActiveAlerts(): Promise<HealthAlertType[]> {
    this.logger.debug('Fetching active health alerts');
    
    const alerts = await this.memoryHealth.getActiveAlerts();
    
    return alerts.map(alert => ({
      id: alert.id,
      type: alert.type,
      severity: alert.severity,
      message: alert.message,
      metrics: {
        totalMemory: alert.metrics.totalMemory,
        usedMemory: alert.metrics.usedMemory,
        freeMemory: alert.metrics.freeMemory,
        memoryUsagePercent: alert.metrics.memoryUsagePercent,
        nodeHeapUsed: alert.metrics.nodeHeapUsed,
        nodeHeapTotal: alert.metrics.nodeHeapTotal,
        nodeExternal: alert.metrics.nodeExternal,
        nodeRss: alert.metrics.nodeRss,
        timestamp: alert.metrics.timestamp
      },
      timestamp: alert.timestamp,
      resolved: alert.resolved,
      resolvedAt: alert.resolvedAt,
      actions: alert.actions
    }));
  }

  @Query(() => [HealthAlertType])
  @UseGuards(AuthGuard)
  async getAlertHistory(
    @Args('limit', { defaultValue: 50 }) limit: number
  ): Promise<HealthAlertType[]> {
    this.logger.debug(`Fetching alert history (limit: ${limit})`);
    
    const history = await this.memoryHealth.getAlertHistory(limit);
    
    return history.map(alert => ({
      id: alert.id,
      type: alert.type,
      severity: alert.severity,
      message: alert.message,
      metrics: {
        totalMemory: alert.metrics.totalMemory,
        usedMemory: alert.metrics.usedMemory,
        freeMemory: alert.metrics.freeMemory,
        memoryUsagePercent: alert.metrics.memoryUsagePercent,
        nodeHeapUsed: alert.metrics.nodeHeapUsed,
        nodeHeapTotal: alert.metrics.nodeHeapTotal,
        nodeExternal: alert.metrics.nodeExternal,
        nodeRss: alert.metrics.nodeRss,
        timestamp: alert.metrics.timestamp
      },
      timestamp: alert.timestamp,
      resolved: alert.resolved,
      resolvedAt: alert.resolvedAt,
      actions: alert.actions
    }));
  }

  @Query(() => [SystemHealthType])
  @UseGuards(AuthGuard)
  async getHealthTrends(
    @Args('hours', { defaultValue: 24 }) hours: number
  ): Promise<SystemHealthType[]> {
    this.logger.debug(`Fetching health trends (${hours} hours)`);
    
    const trends = await this.memoryHealth.getHealthTrends(hours);
    
    return trends.map(health => ({
      overall: health.overall,
      memory: health.memory,
      performance: health.performance,
      optimization: health.optimization,
      alerts: health.alerts
    }));
  }

  @Query(() => DiagnosticsResultType)
  @UseGuards(AuthGuard)
  async runHealthDiagnostics(): Promise<DiagnosticsResultType> {
    this.logger.log('Running comprehensive health diagnostics');
    
    const diagnostics = await this.memoryHealth.runHealthDiagnostics();
    
    return {
      memoryHealth: diagnostics.memoryHealth,
      performanceHealth: diagnostics.performanceHealth,
      systemHealth: diagnostics.systemHealth,
      recommendations: diagnostics.recommendations
    };
  }

  // ========================
  // Mutations
  // ========================

  @Mutation(() => [OptimizationResultType])
  @UseGuards(AuthGuard)
  async runMemoryOptimization(
    @Args('input', { nullable: true }) input?: OptimizationInput
  ): Promise<OptimizationResultType[]> {
    const { level = 'preventive', force = false } = input || {};
    
    this.logger.log(`Running ${level} memory optimization (force: ${force})`);
    
    const results = await this.memoryOptimizer.forceOptimization(level as any);
    
    // Publish optimization event
    this.pubSub.publish('MEMORY_OPTIMIZATION_COMPLETED', {
      memoryOptimizationCompleted: {
        level,
        results,
        timestamp: new Date(),
        totalFreedMB: results.reduce((sum, r) => sum + r.freedMemoryMB, 0)
      }
    });
    
    return results.map(result => ({
      type: result.type,
      freedMemoryMB: result.freedMemoryMB,
      executionTimeMs: result.executionTimeMs,
      itemsProcessed: result.itemsProcessed,
      success: result.success,
      details: result.details
    }));
  }

  @Mutation(() => Boolean)
  @UseGuards(AuthGuard)
  async resolveAlert(
    @Args('alertId') alertId: string
  ): Promise<boolean> {
    this.logger.log(`Resolving alert: ${alertId}`);
    
    try {
      await this.memoryHealth.resolveAlert(alertId);
      
      // Publish alert resolution event
      this.pubSub.publish('ALERT_RESOLVED', {
        alertResolved: {
          alertId,
          timestamp: new Date()
        }
      });
      
      return true;
    } catch (error) {
      this.logger.error(`Failed to resolve alert ${alertId}:`, error);
      return false;
    }
  }

  @Mutation(() => GraphQLJSONObject)
  @UseGuards(AuthGuard)
  async emergencyMemoryCleanup(): Promise<Record<string, any>> {
    this.logger.warn('🆘 Emergency memory cleanup requested');
    
    try {
      const results = await this.memoryOptimizer.forceOptimization('emergency');
      const totalFreed = results.reduce((sum, r) => sum + r.freedMemoryMB, 0);
      
      // Publish emergency cleanup event
      this.pubSub.publish('EMERGENCY_CLEANUP_COMPLETED', {
        emergencyCleanupCompleted: {
          results,
          totalFreedMB: totalFreed,
          timestamp: new Date()
        }
      });
      
      return {
        success: true,
        totalFreedMB: totalFreed,
        operationsCompleted: results.length,
        results,
        timestamp: new Date()
      };
    } catch (error) {
      this.logger.error('Emergency cleanup failed:', error);
      
      return {
        success: false,
        error: error.message,
        timestamp: new Date()
      };
    }
  }

  @Mutation(() => GraphQLJSONObject)
  @UseGuards(AuthGuard)
  async forceGarbageCollection(): Promise<Record<string, any>> {
    this.logger.warn('🗑️ Force garbage collection requested');
    
    const beforeHeap = process.memoryUsage().heapUsed / (1024 * 1024);
    
    if (global.gc) {
      // Run multiple GC cycles
      for (let i = 0; i < 3; i++) {
        global.gc();
      }
      
      const afterHeap = process.memoryUsage().heapUsed / (1024 * 1024);
      const freedMB = beforeHeap - afterHeap;
      
      // Publish GC event
      this.pubSub.publish('GARBAGE_COLLECTION_COMPLETED', {
        garbageCollectionCompleted: {
          beforeHeapMB: beforeHeap,
          afterHeapMB: afterHeap,
          freedMB,
          timestamp: new Date()
        }
      });
      
      return {
        success: true,
        beforeHeapMB: beforeHeap,
        afterHeapMB: afterHeap,
        freedMB,
        timestamp: new Date()
      };
    } else {
      return {
        success: false,
        message: 'Garbage collection not available (--expose-gc flag required)',
        timestamp: new Date()
      };
    }
  }

  // ========================
  // Subscriptions
  // ========================

  @Subscription(() => GraphQLJSONObject, {
    name: 'memoryOptimizationCompleted',
    filter: (payload, variables) => {
      // Optional filtering by optimization level
      return !variables.level || payload.memoryOptimizationCompleted.level === variables.level;
    }
  })
  memoryOptimizationCompleted(
    @Args('level', { nullable: true }) level?: string
  ) {
    return this.pubSub.asyncIterator('MEMORY_OPTIMIZATION_COMPLETED');
  }

  @Subscription(() => GraphQLJSONObject, { name: 'alertCreated' })
  alertCreated() {
    return this.pubSub.asyncIterator('HEALTH_ALERT_CREATED');
  }

  @Subscription(() => GraphQLJSONObject, { name: 'alertResolved' })
  alertResolved() {
    return this.pubSub.asyncIterator('ALERT_RESOLVED');
  }

  @Subscription(() => GraphQLJSONObject, { name: 'emergencyCleanupCompleted' })
  emergencyCleanupCompleted() {
    return this.pubSub.asyncIterator('EMERGENCY_CLEANUP_COMPLETED');
  }

  @Subscription(() => GraphQLJSONObject, { name: 'garbageCollectionCompleted' })
  garbageCollectionCompleted() {
    return this.pubSub.asyncIterator('GARBAGE_COLLECTION_COMPLETED');
  }

  @Subscription(() => MemoryStatsType, {
    name: 'memoryStatsUpdated',
    filter: (payload, variables) => {
      // Only send updates if memory usage crosses certain thresholds
      return payload.memoryStatsUpdated.memoryUsagePercent >= (variables.threshold || 70);
    }
  })
  memoryStatsUpdated(
    @Args('threshold', { defaultValue: 70 }) threshold: number
  ) {
    return this.pubSub.asyncIterator('MEMORY_STATS_UPDATED');
  }

  @Subscription(() => SystemHealthType, { name: 'systemHealthUpdated' })
  systemHealthUpdated() {
    return this.pubSub.asyncIterator('SYSTEM_HEALTH_UPDATED');
  }
}