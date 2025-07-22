import { Injectable, Logger } from '@nestjs/common';
import { Subject, BehaviorSubject } from 'rxjs';

export interface BackpressureMetrics {
  bufferSize: number;
  droppedMessages: number;
  processingRate: number;
  queueDepth: number;
  isThrottled: boolean;
}

export interface BackpressureConfig {
  maxBufferSize: number;
  highWaterMark: number;
  lowWaterMark: number;
  dropPolicy: 'tail' | 'head' | 'none';
  throttleMs: number;
}

@Injectable()
export class BackpressureManager {
  private logger = new Logger('BackpressureManager');
  private buffers = new Map<string, any[]>();
  private metrics = new Map<string, BackpressureMetrics>();
  private throttleStates = new Map<string, boolean>();
  private metricsSubject = new BehaviorSubject<Map<string, BackpressureMetrics>>(new Map());

  private config: BackpressureConfig = {
    maxBufferSize: 1000,
    highWaterMark: 800,
    lowWaterMark: 200,
    dropPolicy: 'tail',
    throttleMs: 100,
  };

  /**
   * Create a buffer for a client
   */
  createBuffer(clientId: string): void {
    if (!this.buffers.has(clientId)) {
      this.buffers.set(clientId, []);
      this.metrics.set(clientId, {
        bufferSize: 0,
        droppedMessages: 0,
        processingRate: 0,
        queueDepth: 0,
        isThrottled: false,
      });
      this.throttleStates.set(clientId, false);
      this.updateMetrics();
    }
  }

  /**
   * Add item to buffer with backpressure handling
   */
  addToBuffer(clientId: string, item: any): boolean {
    const buffer = this.buffers.get(clientId);
    const metrics = this.metrics.get(clientId);
    
    if (!buffer || !metrics) {
      this.createBuffer(clientId);
      return this.addToBuffer(clientId, item);
    }

    // Check if buffer is full
    if (buffer.length >= this.config.maxBufferSize) {
      metrics.droppedMessages++;
      
      // Apply drop policy
      switch (this.config.dropPolicy) {
        case 'tail':
          // Drop new message
          this.logger.warn(`Dropping message for client ${clientId} - buffer full`);
          this.updateMetrics();
          return false;
        case 'head':
          // Drop oldest message
          buffer.shift();
          buffer.push(item);
          break;
        case 'none':
          // Block until space available
          return false;
      }
    } else {
      buffer.push(item);
    }

    metrics.bufferSize = buffer.length;
    metrics.queueDepth = buffer.length;

    // Check high water mark
    if (buffer.length >= this.config.highWaterMark && !this.throttleStates.get(clientId)) {
      this.throttleClient(clientId, true);
    }

    this.updateMetrics();
    return true;
  }

  /**
   * Get items from buffer
   */
  getFromBuffer(clientId: string, count: number = 1): any[] {
    const buffer = this.buffers.get(clientId);
    const metrics = this.metrics.get(clientId);
    
    if (!buffer || !metrics) {
      return [];
    }

    const items = buffer.splice(0, count);
    metrics.bufferSize = buffer.length;
    metrics.queueDepth = buffer.length;

    // Check low water mark
    if (buffer.length <= this.config.lowWaterMark && this.throttleStates.get(clientId)) {
      this.throttleClient(clientId, false);
    }

    // Update processing rate
    this.updateProcessingRate(clientId, items.length);
    this.updateMetrics();

    return items;
  }

  /**
   * Check if client should be throttled
   */
  shouldThrottle(clientId: string): boolean {
    return this.throttleStates.get(clientId) || false;
  }

  /**
   * Clear buffer for a client
   */
  clearBuffer(clientId: string): void {
    this.buffers.delete(clientId);
    this.metrics.delete(clientId);
    this.throttleStates.delete(clientId);
    this.updateMetrics();
  }

  /**
   * Get metrics for a client
   */
  getMetrics(clientId?: string): BackpressureMetrics | Map<string, BackpressureMetrics> {
    if (clientId) {
      return this.metrics.get(clientId) || this.createDefaultMetrics();
    }
    return new Map(this.metrics);
  }

  /**
   * Subscribe to metrics updates
   */
  subscribeToMetrics() {
    return this.metricsSubject.asObservable();
  }

  /**
   * Update configuration
   */
  updateConfig(config: Partial<BackpressureConfig>): void {
    this.config = { ...this.config, ...config };
    this.logger.log(`Backpressure config updated: ${JSON.stringify(this.config)}`);
  }

  private throttleClient(clientId: string, throttle: boolean): void {
    this.throttleStates.set(clientId, throttle);
    const metrics = this.metrics.get(clientId);
    if (metrics) {
      metrics.isThrottled = throttle;
    }
    
    this.logger.log(`Client ${clientId} throttled: ${throttle}`);
  }

  private updateProcessingRate(clientId: string, itemsProcessed: number): void {
    const metrics = this.metrics.get(clientId);
    if (metrics) {
      // Simple exponential moving average
      const alpha = 0.3;
      metrics.processingRate = alpha * itemsProcessed + (1 - alpha) * metrics.processingRate;
    }
  }

  private updateMetrics(): void {
    this.metricsSubject.next(new Map(this.metrics));
  }

  private createDefaultMetrics(): BackpressureMetrics {
    return {
      bufferSize: 0,
      droppedMessages: 0,
      processingRate: 0,
      queueDepth: 0,
      isThrottled: false,
    };
  }

  /**
   * Get system-wide backpressure status
   */
  getSystemStatus(): any {
    const allMetrics = Array.from(this.metrics.values());
    
    return {
      totalClients: this.metrics.size,
      throttledClients: Array.from(this.throttleStates.values()).filter(t => t).length,
      totalBufferSize: allMetrics.reduce((sum, m) => sum + m.bufferSize, 0),
      totalDropped: allMetrics.reduce((sum, m) => sum + m.droppedMessages, 0),
      avgProcessingRate: allMetrics.length > 0
        ? allMetrics.reduce((sum, m) => sum + m.processingRate, 0) / allMetrics.length
        : 0,
      config: this.config,
    };
  }
}
