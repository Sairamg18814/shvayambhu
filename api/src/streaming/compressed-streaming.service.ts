/**
 * Compressed Streaming Service
 * 
 * Service for handling compressed WebSocket data transmission
 */

import { Injectable, Logger } from '@nestjs/common';
import { Socket } from 'socket.io';
import * as zlib from 'zlib';
import { CompressionService, CompressionType } from '../common/compression/compression.service';

export interface CompressionStats {
  originalBytes: number;
  compressedBytes: number;
  compressionRatio: number;
  algorithm: string;
  processingTimeMs: number;
}

export interface CompressedMessage {
  type: 'token' | 'consciousness' | 'metadata' | 'system';
  compressed: boolean;
  algorithm?: string;
  originalSize?: number;
  compressedSize?: number;
  data: string; // Base64 encoded compressed data or raw data
  timestamp: number;
}

@Injectable()
export class CompressedStreamingService {
  private readonly logger = new Logger(CompressedStreamingService.name);
  private compressionStats = new Map<string, CompressionStats[]>();

  constructor(private compressionService: CompressionService) {}

  /**
   * Send compressed message to WebSocket client
   */
  async sendCompressed(
    client: Socket,
    eventName: string,
    data: any,
    forceCompression = false
  ): Promise<void> {
    const startTime = Date.now();
    const serializedData = JSON.stringify(data);
    const originalSize = Buffer.byteLength(serializedData, 'utf8');

    try {
      // Determine if compression is beneficial
      const shouldCompress = forceCompression || this.shouldCompress(serializedData, eventName);

      if (shouldCompress) {
        // Choose optimal compression method
        const compressionResult = await this.compressionService.chooseOptimalCompression(
          serializedData,
          1.2 // Minimum compression ratio
        );

        const compressedMessage: CompressedMessage = {
          type: this.getMessageType(eventName),
          compressed: compressionResult.type !== CompressionType.NONE,
          algorithm: compressionResult.type,
          originalSize: originalSize,
          compressedSize: compressionResult.compressedSize,
          data: compressionResult.data.toString('base64'),
          timestamp: Date.now()
        };

        // Send compressed message
        client.emit('compressed_' + eventName, compressedMessage);

        // Track compression stats
        this.recordCompressionStats(client.id, {
          originalBytes: originalSize,
          compressedBytes: compressionResult.compressedSize,
          compressionRatio: compressionResult.compressionRatio,
          algorithm: compressionResult.type,
          processingTimeMs: Date.now() - startTime
        });

        this.logger.debug(
          `Compressed ${eventName}: ${originalSize} -> ${compressionResult.compressedSize} bytes ` +
          `(${compressionResult.compressionRatio.toFixed(2)}x reduction) using ${compressionResult.type}`
        );
      } else {
        // Send uncompressed for small messages
        const uncompressedMessage: CompressedMessage = {
          type: this.getMessageType(eventName),
          compressed: false,
          data: serializedData,
          timestamp: Date.now()
        };

        client.emit('compressed_' + eventName, uncompressedMessage);
      }
    } catch (error) {
      this.logger.error(`Compression error for ${eventName}: ${error.message}`);
      // Fallback to uncompressed
      client.emit(eventName, data);
    }
  }

  /**
   * Send compressed consciousness state updates
   */
  async sendConsciousnessUpdate(
    client: Socket,
    consciousnessData: any
  ): Promise<void> {
    await this.sendCompressed(client, 'consciousnessUpdate', consciousnessData, true);
  }

  /**
   * Send compressed token stream
   */
  async sendTokenStream(
    client: Socket,
    tokens: any[],
    batchSize = 10
  ): Promise<void> {
    // Batch tokens for better compression
    const batches = this.batchTokens(tokens, batchSize);
    
    for (const batch of batches) {
      await this.sendCompressed(client, 'tokenBatch', {
        tokens: batch,
        batchSize: batch.length,
        isFinalBatch: batch === batches[batches.length - 1]
      }, true);
    }
  }

  /**
   * Send compressed metadata updates
   */
  async sendMetadataUpdate(
    client: Socket,
    metadata: any
  ): Promise<void> {
    // Metadata often contains repetitive information - good for compression
    await this.sendCompressed(client, 'metadataUpdate', metadata, true);
  }

  /**
   * Handle client compression capabilities negotiation
   */
  negotiateCompression(client: Socket, capabilities: {
    supportedAlgorithms: string[];
    maxChunkSize: number;
    preferredCompression: string;
  }): void {
    // Store client capabilities for future compression decisions
    client.data.compressionCapabilities = capabilities;
    
    // Respond with server compression capabilities
    client.emit('compressionNegotiated', {
      supportedAlgorithms: [
        CompressionType.GZIP,
        CompressionType.DEFLATE,
        CompressionType.BROTLI,
        CompressionType.CONSCIOUSNESS
      ],
      optimalChunkSize: 8192,
      defaultAlgorithm: CompressionType.GZIP,
      consciousnessCompressionSupported: true
    });

    this.logger.log(`Compression negotiated for client ${client.id}: ${JSON.stringify(capabilities)}`);
  }

  /**
   * Get compression statistics for a client
   */
  getCompressionStats(clientId: string): {
    totalMessages: number;
    averageCompressionRatio: number;
    totalBytesSaved: number;
    compressionDistribution: Record<string, number>;
    averageProcessingTime: number;
  } {
    const stats = this.compressionStats.get(clientId) || [];
    
    if (stats.length === 0) {
      return {
        totalMessages: 0,
        averageCompressionRatio: 1,
        totalBytesSaved: 0,
        compressionDistribution: {},
        averageProcessingTime: 0
      };
    }

    const totalMessages = stats.length;
    const totalCompressionRatio = stats.reduce((sum, stat) => sum + stat.compressionRatio, 0);
    const averageCompressionRatio = totalCompressionRatio / totalMessages;
    
    const totalBytesSaved = stats.reduce(
      (sum, stat) => sum + (stat.originalBytes - stat.compressedBytes),
      0
    );

    const compressionDistribution: Record<string, number> = {};
    stats.forEach(stat => {
      compressionDistribution[stat.algorithm] = (compressionDistribution[stat.algorithm] || 0) + 1;
    });

    const averageProcessingTime = stats.reduce((sum, stat) => sum + stat.processingTimeMs, 0) / totalMessages;

    return {
      totalMessages,
      averageCompressionRatio,
      totalBytesSaved,
      compressionDistribution,
      averageProcessingTime
    };
  }

  /**
   * Clean up compression stats for disconnected client
   */
  cleanupClient(clientId: string): void {
    this.compressionStats.delete(clientId);
    this.logger.debug(`Cleaned up compression stats for client: ${clientId}`);
  }

  /**
   * Create a compressed WebSocket namespace for high-volume streaming
   */
  createCompressedNamespace(): any {
    return {
      compression: true,
      perMessageDeflate: {
        threshold: 1024, // Only compress messages > 1KB
        zlibDeflateOptions: {
          level: zlib.constants.Z_BEST_COMPRESSION,
          windowBits: 15,
          memLevel: 8,
        },
        zlibInflateOptions: {
          windowBits: 15,
          memLevel: 8,
        },
        serverMaxNoContextTakeover: false,
        clientMaxNoContextTakeover: false,
        serverMaxWindowBits: 15,
        clientMaxWindowBits: 15,
      }
    };
  }

  private shouldCompress(data: string, eventType: string): boolean {
    const dataSize = Buffer.byteLength(data, 'utf8');
    
    // Always compress large messages
    if (dataSize > 2048) return true;
    
    // Compress consciousness-related messages (often repetitive)
    if (eventType.includes('consciousness') || eventType.includes('metadata')) {
      return dataSize > 512;
    }
    
    // Compress token batches
    if (eventType.includes('token') || eventType.includes('batch')) {
      return dataSize > 256;
    }
    
    // Don't compress small messages
    return dataSize > 1024;
  }

  private getMessageType(eventName: string): 'token' | 'consciousness' | 'metadata' | 'system' {
    if (eventName.includes('token')) return 'token';
    if (eventName.includes('consciousness')) return 'consciousness';
    if (eventName.includes('metadata')) return 'metadata';
    return 'system';
  }

  private batchTokens(tokens: any[], batchSize: number): any[][] {
    const batches: any[][] = [];
    for (let i = 0; i < tokens.length; i += batchSize) {
      batches.push(tokens.slice(i, i + batchSize));
    }
    return batches;
  }

  private recordCompressionStats(clientId: string, stats: CompressionStats): void {
    if (!this.compressionStats.has(clientId)) {
      this.compressionStats.set(clientId, []);
    }
    
    const clientStats = this.compressionStats.get(clientId)!;
    clientStats.push(stats);
    
    // Keep only last 1000 stats per client
    if (clientStats.length > 1000) {
      clientStats.splice(0, clientStats.length - 500);
    }
  }

  /**
   * Real-time compression performance monitoring
   */
  getGlobalCompressionMetrics(): {
    totalClients: number;
    totalMessagesCompressed: number;
    globalAverageCompressionRatio: number;
    totalBytesSavedGlobal: number;
    topPerformingAlgorithms: Array<{ algorithm: string; usage: number; avgRatio: number }>;
  } {
    const allStats: CompressionStats[] = [];
    let totalClients = 0;

    this.compressionStats.forEach((clientStats) => {
      totalClients++;
      allStats.push(...clientStats);
    });

    if (allStats.length === 0) {
      return {
        totalClients: 0,
        totalMessagesCompressed: 0,
        globalAverageCompressionRatio: 1,
        totalBytesSavedGlobal: 0,
        topPerformingAlgorithms: []
      };
    }

    const totalMessagesCompressed = allStats.length;
    const globalAverageCompressionRatio = allStats.reduce((sum, stat) => sum + stat.compressionRatio, 0) / totalMessagesCompressed;
    const totalBytesSavedGlobal = allStats.reduce((sum, stat) => sum + (stat.originalBytes - stat.compressedBytes), 0);

    // Calculate algorithm performance
    const algorithmStats: Record<string, { count: number; totalRatio: number }> = {};
    allStats.forEach(stat => {
      if (!algorithmStats[stat.algorithm]) {
        algorithmStats[stat.algorithm] = { count: 0, totalRatio: 0 };
      }
      algorithmStats[stat.algorithm].count++;
      algorithmStats[stat.algorithm].totalRatio += stat.compressionRatio;
    });

    const topPerformingAlgorithms = Object.entries(algorithmStats)
      .map(([algorithm, stats]) => ({
        algorithm,
        usage: stats.count,
        avgRatio: stats.totalRatio / stats.count
      }))
      .sort((a, b) => b.avgRatio - a.avgRatio);

    return {
      totalClients,
      totalMessagesCompressed,
      globalAverageCompressionRatio,
      totalBytesSavedGlobal,
      topPerformingAlgorithms
    };
  }
}