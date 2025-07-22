import { Injectable, NestMiddleware, Logger } from '@nestjs/common';
import { Request, Response, NextFunction } from 'express';
import * as compression from 'compression';
import * as zlib from 'zlib';
import { CompressionService } from '../compression/compression.service';

export interface CompressionStats {
  originalSize: number;
  compressedSize: number;
  compressionRatio: number;
  algorithm: string;
  processingTime: number;
}

@Injectable()
export class GraphQLCompressionMiddleware implements NestMiddleware {
  private readonly logger = new Logger(GraphQLCompressionMiddleware.name);
  private compressionStats: CompressionStats[] = [];

  constructor(private readonly compressionService: CompressionService) {}

  use(req: Request, res: Response, next: NextFunction) {
    const startTime = Date.now();

    // Enhanced compression for GraphQL responses
    const compressionOptions = {
      level: zlib.constants.Z_BEST_COMPRESSION,
      chunkSize: 16 * 1024, // 16KB chunks
      windowBits: 15,
      memLevel: 8,
      strategy: zlib.constants.Z_DEFAULT_STRATEGY,
      filter: this.shouldCompress.bind(this),
      threshold: 1024, // Only compress responses > 1KB
    };

    // Apply standard compression middleware
    const compressionHandler = compression(compressionOptions);
    
    // Intercept response to collect stats
    const originalSend = res.send;
    const originalJson = res.json;

    res.send = (body: any) => {
      this.logCompressionStats(req, res, body, startTime);
      return originalSend.call(res, body);
    };

    res.json = (body: any) => {
      this.logCompressionStats(req, res, JSON.stringify(body), startTime);
      return originalJson.call(res, body);
    };

    compressionHandler(req, res, next);
  }

  private shouldCompress(req: Request, res: Response): boolean {
    // Always compress GraphQL responses
    if (req.path === '/graphql') {
      return true;
    }

    // Compress consciousness state updates
    if (req.path.includes('/consciousness') || req.path.includes('/streaming')) {
      return true;
    }

    // Compress large JSON responses
    const contentType = res.getHeader('content-type') as string;
    if (contentType && contentType.includes('application/json')) {
      return true;
    }

    // Default compression filter
    return compression.filter(req, res);
  }

  private logCompressionStats(
    req: Request,
    res: Response,
    body: any,
    startTime: number,
  ): void {
    try {
      const originalSize = Buffer.isBuffer(body) ? body.length : Buffer.byteLength(body, 'utf8');
      const contentEncoding = res.getHeader('content-encoding') as string;
      const processingTime = Date.now() - startTime;

      if (contentEncoding && originalSize > 0) {
        // Estimate compressed size (actual size would need response interception)
        const estimatedCompressedSize = this.estimateCompressedSize(body, contentEncoding);
        const compressionRatio = originalSize / estimatedCompressedSize;

        const stats: CompressionStats = {
          originalSize,
          compressedSize: estimatedCompressedSize,
          compressionRatio,
          algorithm: contentEncoding,
          processingTime,
        };

        this.compressionStats.push(stats);

        // Keep only last 1000 stats
        if (this.compressionStats.length > 1000) {
          this.compressionStats = this.compressionStats.slice(-500);
        }

        // Log significant compressions
        if (compressionRatio > 2) {
          this.logger.log(
            `High compression achieved: ${originalSize} -> ${estimatedCompressedSize} bytes ` +
            `(${compressionRatio.toFixed(2)}x reduction) using ${contentEncoding}`,
          );
        }
      }
    } catch (error) {
      this.logger.error('Error logging compression stats:', error);
    }
  }

  private estimateCompressedSize(body: any, encoding: string): number {
    // Simple estimation - in production, would use actual compressed response size
    const originalSize = Buffer.isBuffer(body) ? body.length : Buffer.byteLength(body, 'utf8');
    
    switch (encoding) {
      case 'gzip':
        return Math.floor(originalSize * 0.3); // ~70% compression typical
      case 'deflate':
        return Math.floor(originalSize * 0.35); // ~65% compression typical
      case 'br':
        return Math.floor(originalSize * 0.25); // ~75% compression typical
      default:
        return originalSize;
    }
  }

  getCompressionStats(): {
    totalRequests: number;
    averageCompressionRatio: number;
    totalBytesSaved: number;
    averageProcessingTime: number;
    algorithmDistribution: Record<string, number>;
  } {
    if (this.compressionStats.length === 0) {
      return {
        totalRequests: 0,
        averageCompressionRatio: 1,
        totalBytesSaved: 0,
        averageProcessingTime: 0,
        algorithmDistribution: {},
      };
    }

    const totalRequests = this.compressionStats.length;
    const totalCompressionRatio = this.compressionStats.reduce((sum, stat) => sum + stat.compressionRatio, 0);
    const averageCompressionRatio = totalCompressionRatio / totalRequests;
    
    const totalBytesSaved = this.compressionStats.reduce(
      (sum, stat) => sum + (stat.originalSize - stat.compressedSize),
      0,
    );

    const totalProcessingTime = this.compressionStats.reduce((sum, stat) => sum + stat.processingTime, 0);
    const averageProcessingTime = totalProcessingTime / totalRequests;

    const algorithmDistribution: Record<string, number> = {};
    this.compressionStats.forEach(stat => {
      algorithmDistribution[stat.algorithm] = (algorithmDistribution[stat.algorithm] || 0) + 1;
    });

    return {
      totalRequests,
      averageCompressionRatio,
      totalBytesSaved,
      averageProcessingTime,
      algorithmDistribution,
    };
  }
}

@Injectable()
export class ConsciousnessCompressionMiddleware implements NestMiddleware {
  private readonly logger = new Logger(ConsciousnessCompressionMiddleware.name);

  constructor(private readonly compressionService: CompressionService) {}

  async use(req: Request, res: Response, next: NextFunction) {
    // Special handling for consciousness state data
    if (this.isConsciousnessRequest(req)) {
      const originalSend = res.send;
      const originalJson = res.json;

      res.send = async (body: any) => {
        const compressedBody = await this.compressConsciousnessData(body);
        res.setHeader('X-Consciousness-Compression', 'active');
        res.setHeader('X-Original-Size', Buffer.byteLength(body, 'utf8').toString());
        res.setHeader('X-Compressed-Size', Buffer.byteLength(compressedBody, 'utf8').toString());
        return originalSend.call(res, compressedBody);
      };

      res.json = async (body: any) => {
        const jsonBody = JSON.stringify(body);
        const compressedBody = await this.compressConsciousnessData(jsonBody);
        res.setHeader('X-Consciousness-Compression', 'active');
        res.setHeader('X-Original-Size', Buffer.byteLength(jsonBody, 'utf8').toString());
        res.setHeader('X-Compressed-Size', Buffer.byteLength(compressedBody, 'utf8').toString());
        return originalJson.call(res, JSON.parse(compressedBody));
      };
    }

    next();
  }

  private isConsciousnessRequest(req: Request): boolean {
    return (
      req.path.includes('/consciousness') ||
      req.path.includes('/self-model') ||
      req.path.includes('/qualia') ||
      req.path.includes('/stream') ||
      req.path.includes('/metacognition') ||
      req.path.includes('/existential') ||
      req.body?.operationName?.toLowerCase().includes('consciousness')
    );
  }

  private async compressConsciousnessData(data: any): Promise<string> {
    try {
      const dataStr = typeof data === 'string' ? data : JSON.stringify(data);
      
      // Use specialized consciousness compression
      const compressed = await this.compressionService.compressConsciousnessState(dataStr);
      
      return compressed;
    } catch (error) {
      this.logger.error('Error compressing consciousness data:', error);
      return typeof data === 'string' ? data : JSON.stringify(data);
    }
  }
}