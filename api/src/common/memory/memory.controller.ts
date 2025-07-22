/**
 * Memory Management Controller for Shvayambhu LLM System
 * 
 * Provides REST API endpoints for memory optimization, health monitoring,
 * and system diagnostics.
 */

import { Controller, Get, Post, Query, Body, HttpException, HttpStatus, UseGuards, Logger } from '@nestjs/common';
import { ApiTags, ApiOperation, ApiResponse, ApiQuery, ApiBody } from '@nestjs/swagger';

// Import services
import { MemoryOptimizerService, OptimizationResult } from './memory-optimizer.service';
import { MemoryHealthService, SystemHealth, HealthAlert, MemoryPrediction } from './memory-health.service';

// Import guards
import { AuthGuard } from '../auth/auth.guard';

@ApiTags('Memory Management')
@Controller('api/memory')
@UseGuards(AuthGuard)
export class MemoryController {
  private readonly logger = new Logger(MemoryController.name);
  
  constructor(
    private readonly memoryOptimizer: MemoryOptimizerService,
    private readonly memoryHealth: MemoryHealthService
  ) {}
  
  // ========================
  // Memory Statistics
  // ========================
  
  @Get('stats')
  @ApiOperation({ summary: 'Get current memory statistics' })
  @ApiResponse({ status: 200, description: 'Memory statistics retrieved successfully' })
  async getMemoryStats() {
    try {
      const stats = this.memoryOptimizer.getMemoryStats();
      const report = await this.memoryOptimizer.getMemoryReport();
      
      return {
        success: true,
        data: {
          current: stats,
          report,
          timestamp: new Date()
        }
      };
    } catch (error) {
      this.logger.error('Failed to get memory stats:', error);
      throw new HttpException('Failed to retrieve memory statistics', HttpStatus.INTERNAL_SERVER_ERROR);
    }
  }
  
  @Get('health')
  @ApiOperation({ summary: 'Get system health status' })
  @ApiResponse({ status: 200, description: 'System health retrieved successfully' })
  async getSystemHealth(): Promise<{ success: boolean; data: SystemHealth }> {
    try {
      const health = await this.memoryHealth.getSystemHealth();
      
      return {
        success: true,
        data: health
      };
    } catch (error) {
      this.logger.error('Failed to get system health:', error);
      throw new HttpException('Failed to retrieve system health', HttpStatus.INTERNAL_SERVER_ERROR);
    }
  }
  
  @Get('prediction')
  @ApiOperation({ summary: 'Get memory usage prediction' })
  @ApiResponse({ status: 200, description: 'Memory prediction retrieved successfully' })
  async getMemoryPrediction(): Promise<{ success: boolean; data: MemoryPrediction | null }> {
    try {
      const prediction = await this.memoryHealth.getMemoryPrediction();
      
      return {
        success: true,
        data: prediction
      };
    } catch (error) {
      this.logger.error('Failed to get memory prediction:', error);
      throw new HttpException('Failed to retrieve memory prediction', HttpStatus.INTERNAL_SERVER_ERROR);
    }
  }
  
  // ========================
  // Optimization Operations
  // ========================
  
  @Post('optimize')
  @ApiOperation({ summary: 'Run memory optimization' })
  @ApiBody({
    schema: {
      type: 'object',
      properties: {
        level: {
          type: 'string',
          enum: ['preventive', 'critical', 'emergency'],
          default: 'preventive'
        },
        force: {
          type: 'boolean',
          default: false
        }
      }
    }
  })
  @ApiResponse({ status: 200, description: 'Optimization completed successfully' })
  @ApiResponse({ status: 429, description: 'Optimization already in progress' })
  async runOptimization(@Body() body: { level?: 'preventive' | 'critical' | 'emergency'; force?: boolean }): Promise<{
    success: boolean;
    data: OptimizationResult[];
    message: string;
  }> {
    try {
      const { level = 'preventive', force = false } = body;
      
      // Check if optimization is already running
      const currentStats = this.memoryOptimizer.getMemoryStats();
      if (currentStats && !force) {
        // You might want to add a property to track if optimization is running
        // For now, we'll proceed
      }
      
      this.logger.log(`Running ${level} optimization (force: ${force})`);
      
      const results = await this.memoryOptimizer.forceOptimization(level);
      const totalFreed = results.reduce((sum, result) => sum + result.freedMemoryMB, 0);
      
      return {
        success: true,
        data: results,
        message: `Optimization completed: ${totalFreed.toFixed(1)}MB freed across ${results.length} operations`
      };
    } catch (error) {
      this.logger.error('Optimization failed:', error);
      throw new HttpException(`Optimization failed: ${error.message}`, HttpStatus.INTERNAL_SERVER_ERROR);
    }
  }
  
  @Get('optimization-history')
  @ApiOperation({ summary: 'Get optimization history' })
  @ApiQuery({ name: 'limit', required: false, type: Number, description: 'Number of records to return' })
  @ApiResponse({ status: 200, description: 'Optimization history retrieved successfully' })
  async getOptimizationHistory(@Query('limit') limit: string = '20'): Promise<{
    success: boolean;
    data: OptimizationResult[];
  }> {
    try {
      const report = await this.memoryOptimizer.getMemoryReport();
      const history = report.recentOptimizations.slice(-parseInt(limit));
      
      return {
        success: true,
        data: history
      };
    } catch (error) {
      this.logger.error('Failed to get optimization history:', error);
      throw new HttpException('Failed to retrieve optimization history', HttpStatus.INTERNAL_SERVER_ERROR);
    }
  }
  
  // ========================
  // Alert Management
  // ========================
  
  @Get('alerts')
  @ApiOperation({ summary: 'Get active alerts' })
  @ApiResponse({ status: 200, description: 'Active alerts retrieved successfully' })
  async getActiveAlerts(): Promise<{ success: boolean; data: HealthAlert[] }> {
    try {
      const alerts = await this.memoryHealth.getActiveAlerts();
      
      return {
        success: true,
        data: alerts
      };
    } catch (error) {
      this.logger.error('Failed to get active alerts:', error);
      throw new HttpException('Failed to retrieve active alerts', HttpStatus.INTERNAL_SERVER_ERROR);
    }
  }
  
  @Get('alerts/history')
  @ApiOperation({ summary: 'Get alert history' })
  @ApiQuery({ name: 'limit', required: false, type: Number, description: 'Number of records to return' })
  @ApiResponse({ status: 200, description: 'Alert history retrieved successfully' })
  async getAlertHistory(@Query('limit') limit: string = '50'): Promise<{
    success: boolean;
    data: HealthAlert[];
  }> {
    try {
      const history = await this.memoryHealth.getAlertHistory(parseInt(limit));
      
      return {
        success: true,
        data: history
      };
    } catch (error) {
      this.logger.error('Failed to get alert history:', error);
      throw new HttpException('Failed to retrieve alert history', HttpStatus.INTERNAL_SERVER_ERROR);
    }
  }
  
  @Post('alerts/:alertId/resolve')
  @ApiOperation({ summary: 'Resolve an alert' })
  @ApiResponse({ status: 200, description: 'Alert resolved successfully' })
  @ApiResponse({ status: 404, description: 'Alert not found' })
  async resolveAlert(@Query('alertId') alertId: string): Promise<{
    success: boolean;
    message: string;
  }> {
    try {
      await this.memoryHealth.resolveAlert(alertId);
      
      return {
        success: true,
        message: `Alert ${alertId} resolved successfully`
      };
    } catch (error) {
      this.logger.error(`Failed to resolve alert ${alertId}:`, error);
      throw new HttpException('Failed to resolve alert', HttpStatus.INTERNAL_SERVER_ERROR);
    }
  }
  
  // ========================
  // Health Monitoring
  // ========================
  
  @Get('health/trends')
  @ApiOperation({ summary: 'Get health trends' })
  @ApiQuery({ name: 'hours', required: false, type: Number, description: 'Number of hours to look back' })
  @ApiResponse({ status: 200, description: 'Health trends retrieved successfully' })
  async getHealthTrends(@Query('hours') hours: string = '24'): Promise<{
    success: boolean;
    data: SystemHealth[];
  }> {
    try {
      const trends = await this.memoryHealth.getHealthTrends(parseInt(hours));
      
      return {
        success: true,
        data: trends
      };
    } catch (error) {
      this.logger.error('Failed to get health trends:', error);
      throw new HttpException('Failed to retrieve health trends', HttpStatus.INTERNAL_SERVER_ERROR);
    }
  }
  
  @Get('diagnostics')
  @ApiOperation({ summary: 'Run comprehensive system diagnostics' })
  @ApiResponse({ status: 200, description: 'Diagnostics completed successfully' })
  async runDiagnostics(): Promise<{
    success: boolean;
    data: any;
    timestamp: Date;
  }> {
    try {
      this.logger.log('Running comprehensive system diagnostics...');
      
      const diagnostics = await this.memoryHealth.runHealthDiagnostics();
      
      return {
        success: true,
        data: diagnostics,
        timestamp: new Date()
      };
    } catch (error) {
      this.logger.error('Failed to run diagnostics:', error);
      throw new HttpException('Failed to run system diagnostics', HttpStatus.INTERNAL_SERVER_ERROR);
    }
  }
  
  // ========================
  // Configuration
  // ========================
  
  @Get('thresholds')
  @ApiOperation({ summary: 'Get memory thresholds configuration' })
  @ApiResponse({ status: 200, description: 'Thresholds retrieved successfully' })
  async getMemoryThresholds(): Promise<{ success: boolean; data: any }> {
    try {
      const report = await this.memoryOptimizer.getMemoryReport();
      
      return {
        success: true,
        data: {
          thresholds: report.thresholds,
          recommendations: report.recommendations
        }
      };
    } catch (error) {
      this.logger.error('Failed to get memory thresholds:', error);
      throw new HttpException('Failed to retrieve memory thresholds', HttpStatus.INTERNAL_SERVER_ERROR);
    }
  }
  
  // ========================
  // Emergency Operations
  // ========================
  
  @Post('emergency/clear-caches')
  @ApiOperation({ summary: 'Emergency cache clearing' })
  @ApiResponse({ status: 200, description: 'Caches cleared successfully' })
  async emergencyClearCaches(): Promise<{ success: boolean; message: string }> {
    try {
      this.logger.warn('🆘 Emergency cache clearing requested');
      
      // This would trigger emergency cache clearing
      const results = await this.memoryOptimizer.forceOptimization('emergency');
      const cacheResults = results.filter(r => r.type === 'cache_clear');
      const totalFreed = cacheResults.reduce((sum, r) => sum + r.freedMemoryMB, 0);
      
      return {
        success: true,
        message: `Emergency cache clearing completed: ${totalFreed.toFixed(1)}MB freed`
      };
    } catch (error) {
      this.logger.error('Emergency cache clearing failed:', error);
      throw new HttpException('Emergency cache clearing failed', HttpStatus.INTERNAL_SERVER_ERROR);
    }
  }
  
  @Post('emergency/force-gc')
  @ApiOperation({ summary: 'Force garbage collection' })
  @ApiResponse({ status: 200, description: 'Garbage collection completed' })
  async forceGarbageCollection(): Promise<{ success: boolean; data: any; message: string }> {
    try {
      this.logger.warn('🗑️ Force garbage collection requested');
      
      const beforeHeap = process.memoryUsage().heapUsed / (1024 * 1024);
      
      if (global.gc) {
        // Run multiple GC cycles for maximum effect
        for (let i = 0; i < 3; i++) {
          global.gc();
        }
        
        const afterHeap = process.memoryUsage().heapUsed / (1024 * 1024);
        const freedMB = beforeHeap - afterHeap;
        
        return {
          success: true,
          data: {
            beforeHeapMB: beforeHeap,
            afterHeapMB: afterHeap,
            freedMB
          },
          message: `Garbage collection completed: ${freedMB.toFixed(1)}MB freed`
        };
      } else {
        return {
          success: false,
          data: null,
          message: 'Garbage collection not available (--expose-gc flag required)'
        };
      }
    } catch (error) {
      this.logger.error('Force garbage collection failed:', error);
      throw new HttpException('Force garbage collection failed', HttpStatus.INTERNAL_SERVER_ERROR);
    }
  }
}