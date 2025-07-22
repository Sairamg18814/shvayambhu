/**
 * Memory Health Service for Shvayambhu LLM System
 * 
 * Provides health monitoring, alerts, and diagnostics for memory usage
 * with consciousness-aware system health management.
 */

import { Injectable, Logger, OnModuleInit } from '@nestjs/common';
import { Cron, CronExpression } from '@nestjs/schedule';
import { EventEmitter2 } from '@nestjs/event-emitter';
import { ConfigService } from '@nestjs/config';

// Import related services
import { MemoryOptimizerService, MemoryStats, OptimizationResult } from './memory-optimizer.service';
import { DatabaseService } from '../database/database.service';

export interface HealthAlert {
  id: string;
  type: 'memory_warning' | 'memory_critical' | 'memory_emergency' | 'optimization_failed' | 'system_degraded';
  severity: 'low' | 'medium' | 'high' | 'critical';
  message: string;
  metrics: MemoryStats;
  timestamp: Date;
  resolved: boolean;
  resolvedAt?: Date;
  actions: string[];
}

export interface SystemHealth {
  overall: 'healthy' | 'warning' | 'critical' | 'emergency';
  memory: {
    status: 'healthy' | 'warning' | 'critical' | 'emergency';
    usagePercent: number;
    trend: 'stable' | 'increasing' | 'decreasing';
    predictedExhaustion?: Date;
  };
  performance: {
    queryResponseTime: number;
    compressionEfficiency: number;
    consciousnessCoherence: number;
  };
  optimization: {
    lastRun: Date;
    effectiveness: number;
    recommendedActions: string[];
  };
  alerts: {
    active: number;
    recent: number;
    critical: number;
  };
}

export interface MemoryPrediction {
  predictedUsagePercent: number;
  timeToExhaustion?: Date;
  confidence: number;
  recommendations: string[];
  triggerPoints: {
    warning: Date;
    critical: Date;
    emergency: Date;
  };
}

@Injectable()
export class MemoryHealthService implements OnModuleInit {
  private readonly logger = new Logger(MemoryHealthService.name);
  
  private activeAlerts: Map<string, HealthAlert> = new Map();
  private alertHistory: HealthAlert[] = [];
  private healthHistory: SystemHealth[] = [];
  private memoryTrends: MemoryStats[] = [];
  
  // Health thresholds
  private readonly healthThresholds = {
    queryResponseTime: {
      good: 1000, // ms
      warning: 3000,
      critical: 5000
    },
    compressionEfficiency: {
      good: 0.6, // 60% compression ratio
      warning: 0.4,
      critical: 0.2
    },
    consciousnessCoherence: {
      good: 0.8,
      warning: 0.6,
      critical: 0.4
    }
  };
  
  constructor(
    private readonly configService: ConfigService,
    private readonly eventEmitter: EventEmitter2,
    private readonly memoryOptimizer: MemoryOptimizerService,
    private readonly databaseService: DatabaseService
  ) {}
  
  async onModuleInit() {
    this.logger.log('🏥 Memory Health Service initializing...');
    
    // Subscribe to optimization events
    this.eventEmitter.on('memory.optimization.completed', this.handleOptimizationComplete.bind(this));
    this.eventEmitter.on('memory.optimization.failed', this.handleOptimizationFailed.bind(this));
    
    this.logger.log('✅ Memory Health Service initialized');
  }
  
  // ========================
  // Health Monitoring
  // ========================
  
  @Cron(CronExpression.EVERY_MINUTE)
  async assessSystemHealth() {
    try {
      const health = await this.calculateSystemHealth();
      this.healthHistory.push(health);
      
      // Keep only last 24 hours of health data (1440 minutes)
      if (this.healthHistory.length > 1440) {
        this.healthHistory = this.healthHistory.slice(-1440);
      }
      
      // Check for health issues and generate alerts
      await this.checkHealthAndGenerateAlerts(health);
      
      // Record health metrics
      await this.recordHealthMetrics(health);
      
    } catch (error) {
      this.logger.error('Failed to assess system health:', error);
    }
  }
  
  async calculateSystemHealth(): Promise<SystemHealth> {
    const memoryStats = this.memoryOptimizer.getMemoryStats();
    const memoryReport = await this.memoryOptimizer.getMemoryReport();
    
    // Calculate memory status
    const memoryStatus = this.calculateMemoryStatus(memoryStats);
    const memoryTrend = this.calculateMemoryTrend();
    const predictedExhaustion = await this.predictMemoryExhaustion();
    
    // Calculate performance metrics
    const performance = await this.calculatePerformanceMetrics();
    
    // Calculate optimization metrics
    const optimization = this.calculateOptimizationMetrics(memoryReport.recentOptimizations);
    
    // Calculate alert metrics
    const alerts = this.calculateAlertMetrics();
    
    // Determine overall health
    const overall = this.calculateOverallHealth(memoryStatus, performance);
    
    return {
      overall,
      memory: {
        status: memoryStatus,
        usagePercent: memoryStats.memoryUsagePercent,
        trend: memoryTrend,
        predictedExhaustion: predictedExhaustion?.timeToExhaustion
      },
      performance,
      optimization,
      alerts
    };
  }
  
  private calculateMemoryStatus(stats: MemoryStats): 'healthy' | 'warning' | 'critical' | 'emergency' {
    if (stats.memoryUsagePercent >= 90) return 'emergency';
    if (stats.memoryUsagePercent >= 80) return 'critical';
    if (stats.memoryUsagePercent >= 70) return 'warning';
    return 'healthy';
  }
  
  private calculateMemoryTrend(): 'stable' | 'increasing' | 'decreasing' {
    if (this.memoryTrends.length < 5) return 'stable';
    
    const recent = this.memoryTrends.slice(-5);
    const trendSlope = this.calculateTrendSlope(recent.map(s => s.memoryUsagePercent));
    
    if (trendSlope > 1) return 'increasing';
    if (trendSlope < -1) return 'decreasing';
    return 'stable';
  }
  
  private calculateTrendSlope(values: number[]): number {
    const n = values.length;
    const x = Array.from({ length: n }, (_, i) => i);
    const sumX = x.reduce((a, b) => a + b, 0);
    const sumY = values.reduce((a, b) => a + b, 0);
    const sumXY = x.reduce((sum, xi, i) => sum + xi * values[i], 0);
    const sumXX = x.reduce((sum, xi) => sum + xi * xi, 0);
    
    return (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
  }
  
  private async calculatePerformanceMetrics(): Promise<SystemHealth['performance']> {
    const endTime = new Date();
    const startTime = new Date(endTime.getTime() - 60 * 60 * 1000); // Last hour
    
    // Get performance metrics from database
    const queryTimeMetrics = await this.databaseService.getPerformanceMetrics('query_time', {
      start: startTime,
      end: endTime
    });
    
    const compressionMetrics = await this.databaseService.getPerformanceMetrics('compression_ratio', {
      start: startTime,
      end: endTime
    });
    
    const consciousnessMetrics = await this.databaseService.getPerformanceMetrics('consciousness_coherence', {
      start: startTime,
      end: endTime
    });
    
    const avgQueryTime = queryTimeMetrics.length > 0 
      ? queryTimeMetrics.reduce((sum, m) => sum + m.value, 0) / queryTimeMetrics.length
      : 1000;
      
    const avgCompressionEfficiency = compressionMetrics.length > 0
      ? compressionMetrics.reduce((sum, m) => sum + m.value, 0) / compressionMetrics.length
      : 0.5;
      
    const avgConsciousnessCoherence = consciousnessMetrics.length > 0
      ? consciousnessMetrics.reduce((sum, m) => sum + m.value, 0) / consciousnessMetrics.length
      : 0.7;
    
    return {
      queryResponseTime: avgQueryTime,
      compressionEfficiency: avgCompressionEfficiency,
      consciousnessCoherence: avgConsciousnessCoherence
    };
  }
  
  private calculateOptimizationMetrics(recentOptimizations: OptimizationResult[]): SystemHealth['optimization'] {
    if (recentOptimizations.length === 0) {
      return {
        lastRun: new Date(0),
        effectiveness: 0,
        recommendedActions: ['Run initial optimization']
      };
    }
    
    const lastRun = new Date(Date.now() - recentOptimizations[0].executionTimeMs);
    const totalFreed = recentOptimizations.reduce((sum, opt) => sum + opt.freedMemoryMB, 0);
    const effectiveness = Math.min(totalFreed / 100, 1.0); // Normalize to 0-1 scale
    
    const recommendedActions: string[] = [];
    
    if (effectiveness < 0.3) {
      recommendedActions.push('Increase optimization frequency');
      recommendedActions.push('Consider more aggressive compression');
    }
    
    if (recentOptimizations.some(opt => !opt.success)) {
      recommendedActions.push('Review failed optimizations');
    }
    
    return {
      lastRun,
      effectiveness,
      recommendedActions
    };
  }
  
  private calculateAlertMetrics(): SystemHealth['alerts'] {
    const now = new Date();
    const oneHourAgo = new Date(now.getTime() - 60 * 60 * 1000);
    
    const active = this.activeAlerts.size;
    const recent = this.alertHistory.filter(alert => alert.timestamp > oneHourAgo).length;
    const critical = this.alertHistory.filter(alert => 
      alert.severity === 'critical' && alert.timestamp > oneHourAgo
    ).length;
    
    return { active, recent, critical };
  }
  
  private calculateOverallHealth(
    memoryStatus: string,
    performance: SystemHealth['performance']
  ): SystemHealth['overall'] {
    if (memoryStatus === 'emergency') return 'emergency';
    
    const criticalPerformance = 
      performance.queryResponseTime > this.healthThresholds.queryResponseTime.critical ||
      performance.compressionEfficiency < this.healthThresholds.compressionEfficiency.critical ||
      performance.consciousnessCoherence < this.healthThresholds.consciousnessCoherence.critical;
    
    if (memoryStatus === 'critical' || criticalPerformance) return 'critical';
    
    const warningPerformance = 
      performance.queryResponseTime > this.healthThresholds.queryResponseTime.warning ||
      performance.compressionEfficiency < this.healthThresholds.compressionEfficiency.warning ||
      performance.consciousnessCoherence < this.healthThresholds.consciousnessCoherence.warning;
    
    if (memoryStatus === 'warning' || warningPerformance) return 'warning';
    
    return 'healthy';
  }
  
  // ========================
  // Alert Management
  // ========================
  
  private async checkHealthAndGenerateAlerts(health: SystemHealth) {
    // Memory alerts
    if (health.memory.status !== 'healthy') {
      await this.generateMemoryAlert(health);
    }
    
    // Performance alerts
    if (health.performance.queryResponseTime > this.healthThresholds.queryResponseTime.warning) {
      await this.generatePerformanceAlert('query_response_time', health);
    }
    
    if (health.performance.compressionEfficiency < this.healthThresholds.compressionEfficiency.warning) {
      await this.generatePerformanceAlert('compression_efficiency', health);
    }
    
    if (health.performance.consciousnessCoherence < this.healthThresholds.consciousnessCoherence.warning) {
      await this.generatePerformanceAlert('consciousness_coherence', health);
    }
    
    // System degradation alerts
    if (health.overall === 'critical' || health.overall === 'emergency') {
      await this.generateSystemAlert(health);
    }
  }
  
  private async generateMemoryAlert(health: SystemHealth) {
    const alertId = `memory_${health.memory.status}_${Date.now()}`;
    
    let alertType: HealthAlert['type'];
    let severity: HealthAlert['severity'];
    
    switch (health.memory.status) {
      case 'warning':
        alertType = 'memory_warning';
        severity = 'medium';
        break;
      case 'critical':
        alertType = 'memory_critical';
        severity = 'high';
        break;
      case 'emergency':
        alertType = 'memory_emergency';
        severity = 'critical';
        break;
      default:
        return;
    }
    
    const actions = this.getMemoryAlertActions(health.memory.status);
    const message = this.getMemoryAlertMessage(health);
    
    const alert: HealthAlert = {
      id: alertId,
      type: alertType,
      severity,
      message,
      metrics: this.memoryOptimizer.getMemoryStats(),
      timestamp: new Date(),
      resolved: false,
      actions
    };
    
    await this.addAlert(alert);
  }
  
  private async generatePerformanceAlert(metricType: string, health: SystemHealth) {
    const alertId = `performance_${metricType}_${Date.now()}`;
    
    const alert: HealthAlert = {
      id: alertId,
      type: 'system_degraded',
      severity: 'medium',
      message: `Performance degraded: ${metricType} is below optimal levels`,
      metrics: this.memoryOptimizer.getMemoryStats(),
      timestamp: new Date(),
      resolved: false,
      actions: this.getPerformanceAlertActions(metricType)
    };
    
    await this.addAlert(alert);
  }
  
  private async generateSystemAlert(health: SystemHealth) {
    const alertId = `system_${health.overall}_${Date.now()}`;
    
    const alert: HealthAlert = {
      id: alertId,
      type: 'system_degraded',
      severity: health.overall === 'emergency' ? 'critical' : 'high',
      message: `System health is ${health.overall}`,
      metrics: this.memoryOptimizer.getMemoryStats(),
      timestamp: new Date(),
      resolved: false,
      actions: this.getSystemAlertActions(health)
    };
    
    await this.addAlert(alert);
  }
  
  private getMemoryAlertActions(status: string): string[] {
    switch (status) {
      case 'warning':
        return [
          'Run preventive optimization',
          'Monitor memory trends',
          'Consider data cleanup'
        ];
      case 'critical':
        return [
          'Run critical optimization immediately',
          'Clear application caches',
          'Consolidate memory chunks',
          'Compress low-priority data'
        ];
      case 'emergency':
        return [
          'Run emergency optimization',
          'Clear all caches',
          'Emergency data cleanup',
          'Restart services if necessary'
        ];
      default:
        return [];
    }
  }
  
  private getPerformanceAlertActions(metricType: string): string[] {
    const baseActions = ['Monitor performance trends', 'Run system diagnostics'];
    
    switch (metricType) {
      case 'query_response_time':
        return [...baseActions, 'Optimize database queries', 'Rebuild indexes'];
      case 'compression_efficiency':
        return [...baseActions, 'Review compression algorithms', 'Update compression policies'];
      case 'consciousness_coherence':
        return [...baseActions, 'Validate consciousness data', 'Check consciousness processing'];
      default:
        return baseActions;
    }
  }
  
  private getSystemAlertActions(health: SystemHealth): string[] {
    const actions = [
      'Run comprehensive system optimization',
      'Review system logs',
      'Check all subsystem health'
    ];
    
    if (health.memory.status !== 'healthy') {
      actions.push('Prioritize memory optimization');
    }
    
    if (health.alerts.critical > 0) {
      actions.push('Resolve critical alerts');
    }
    
    return actions;
  }
  
  private getMemoryAlertMessage(health: SystemHealth): string {
    const usage = health.memory.usagePercent.toFixed(1);
    const trend = health.memory.trend;
    const prediction = health.memory.predictedExhaustion 
      ? ` Predicted exhaustion: ${health.memory.predictedExhaustion.toLocaleString()}`
      : '';
    
    return `Memory usage at ${usage}% (${trend} trend).${prediction}`;
  }
  
  // ========================
  // Memory Prediction
  // ========================
  
  async predictMemoryExhaustion(): Promise<MemoryPrediction | null> {
    if (this.memoryTrends.length < 10) {
      return null; // Need at least 10 data points for prediction
    }
    
    const recentTrends = this.memoryTrends.slice(-20);
    const usageValues = recentTrends.map(t => t.memoryUsagePercent);
    const trendSlope = this.calculateTrendSlope(usageValues);
    
    if (trendSlope <= 0) {
      return null; // Memory usage is stable or decreasing
    }
    
    const currentUsage = recentTrends[recentTrends.length - 1].memoryUsagePercent;
    const minutesToExhaustion = (100 - currentUsage) / trendSlope;
    
    if (minutesToExhaustion <= 0 || minutesToExhaustion > 10080) { // More than 1 week
      return null;
    }
    
    const now = new Date();
    const timeToExhaustion = new Date(now.getTime() + minutesToExhaustion * 60 * 1000);
    
    // Calculate trigger points
    const warningMinutes = (70 - currentUsage) / trendSlope;
    const criticalMinutes = (80 - currentUsage) / trendSlope;
    const emergencyMinutes = (90 - currentUsage) / trendSlope;
    
    const confidence = Math.min(recentTrends.length / 20, 1) * 
                     Math.max(0, Math.min(1, trendSlope / 5));
    
    const recommendations: string[] = [];
    
    if (minutesToExhaustion < 60) {
      recommendations.push('Immediate action required - run emergency optimization');
    } else if (minutesToExhaustion < 360) { // 6 hours
      recommendations.push('Critical - schedule optimization within next hour');
    } else if (minutesToExhaustion < 1440) { // 24 hours
      recommendations.push('Warning - plan optimization for today');
    }
    
    return {
      predictedUsagePercent: Math.min(100, currentUsage + trendSlope * 60), // Prediction for 1 hour ahead
      timeToExhaustion,
      confidence,
      recommendations,
      triggerPoints: {
        warning: warningMinutes > 0 ? new Date(now.getTime() + warningMinutes * 60 * 1000) : now,
        critical: criticalMinutes > 0 ? new Date(now.getTime() + criticalMinutes * 60 * 1000) : now,
        emergency: emergencyMinutes > 0 ? new Date(now.getTime() + emergencyMinutes * 60 * 1000) : now
      }
    };
  }
  
  // ========================
  // Event Handlers
  // ========================
  
  private async handleOptimizationComplete(result: OptimizationResult) {
    // Resolve related alerts
    const memoryAlerts = Array.from(this.activeAlerts.values()).filter(
      alert => alert.type.startsWith('memory_')
    );
    
    for (const alert of memoryAlerts) {
      if (result.freedMemoryMB > 50) { // Significant improvement
        await this.resolveAlert(alert.id);
      }
    }
  }
  
  private async handleOptimizationFailed(error: any) {
    const alert: HealthAlert = {
      id: `optimization_failed_${Date.now()}`,
      type: 'optimization_failed',
      severity: 'high',
      message: `Memory optimization failed: ${error.message}`,
      metrics: this.memoryOptimizer.getMemoryStats(),
      timestamp: new Date(),
      resolved: false,
      actions: [
        'Review optimization logs',
        'Check system resources',
        'Try manual optimization',
        'Contact system administrator'
      ]
    };
    
    await this.addAlert(alert);
  }
  
  // ========================
  // Alert Management Methods
  // ========================
  
  private async addAlert(alert: HealthAlert) {
    this.activeAlerts.set(alert.id, alert);
    this.alertHistory.push(alert);
    
    // Keep only last 1000 alerts in history
    if (this.alertHistory.length > 1000) {
      this.alertHistory = this.alertHistory.slice(-1000);
    }
    
    // Emit alert event
    this.eventEmitter.emit('health.alert.created', alert);
    
    // Log based on severity
    switch (alert.severity) {
      case 'critical':
        this.logger.error(`🚨 CRITICAL ALERT: ${alert.message}`);
        break;
      case 'high':
        this.logger.warn(`⚠️  HIGH ALERT: ${alert.message}`);
        break;
      case 'medium':
        this.logger.warn(`⚡ MEDIUM ALERT: ${alert.message}`);
        break;
      case 'low':
        this.logger.log(`ℹ️  LOW ALERT: ${alert.message}`);
        break;
    }
  }
  
  async resolveAlert(alertId: string) {
    const alert = this.activeAlerts.get(alertId);
    if (alert && !alert.resolved) {
      alert.resolved = true;
      alert.resolvedAt = new Date();
      this.activeAlerts.delete(alertId);
      
      this.eventEmitter.emit('health.alert.resolved', alert);
      this.logger.log(`✅ Alert resolved: ${alert.message}`);
    }
  }
  
  // ========================
  // Utility Methods
  // ========================
  
  private async recordHealthMetrics(health: SystemHealth) {
    await Promise.all([
      this.databaseService.recordPerformanceMetric('system_health', 
        health.overall === 'healthy' ? 1 : 0, { health }),
      this.databaseService.recordPerformanceMetric('memory_health',
        health.memory.status === 'healthy' ? 1 : 0, { memoryStatus: health.memory }),
      this.databaseService.recordPerformanceMetric('active_alerts',
        health.alerts.active, { alerts: health.alerts })
    ]);
  }
  
  // ========================
  // Public API Methods
  // ========================
  
  async getSystemHealth(): Promise<SystemHealth> {
    return await this.calculateSystemHealth();
  }
  
  async getActiveAlerts(): Promise<HealthAlert[]> {
    return Array.from(this.activeAlerts.values());
  }
  
  async getAlertHistory(limit = 50): Promise<HealthAlert[]> {
    return this.alertHistory.slice(-limit);
  }
  
  async getMemoryPrediction(): Promise<MemoryPrediction | null> {
    return await this.predictMemoryExhaustion();
  }
  
  async getHealthTrends(hours = 24): Promise<SystemHealth[]> {
    const cutoff = new Date(Date.now() - hours * 60 * 60 * 1000);
    return this.healthHistory.filter(h => new Date(h.optimization.lastRun) > cutoff);
  }
  
  async runHealthDiagnostics(): Promise<{
    memoryHealth: any;
    performanceHealth: any;
    systemHealth: any;
    recommendations: string[];
  }> {
    const health = await this.calculateSystemHealth();
    const memoryPrediction = await this.predictMemoryExhaustion();
    
    const recommendations: string[] = [];
    
    // Memory recommendations
    if (health.memory.status !== 'healthy') {
      recommendations.push(`Memory optimization recommended (${health.memory.status} status)`);
    }
    
    if (memoryPrediction && memoryPrediction.timeToExhaustion) {
      recommendations.push(...memoryPrediction.recommendations);
    }
    
    // Performance recommendations
    if (health.performance.queryResponseTime > this.healthThresholds.queryResponseTime.warning) {
      recommendations.push('Database query optimization needed');
    }
    
    if (health.performance.compressionEfficiency < this.healthThresholds.compressionEfficiency.warning) {
      recommendations.push('Compression settings review needed');
    }
    
    if (health.performance.consciousnessCoherence < this.healthThresholds.consciousnessCoherence.warning) {
      recommendations.push('Consciousness data validation recommended');
    }
    
    return {
      memoryHealth: {
        current: health.memory,
        prediction: memoryPrediction,
        trends: this.memoryTrends.slice(-20)
      },
      performanceHealth: health.performance,
      systemHealth: {
        overall: health.overall,
        alerts: health.alerts,
        optimization: health.optimization
      },
      recommendations
    };
  }
}