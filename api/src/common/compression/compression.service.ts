import { Injectable, Logger } from '@nestjs/common';
import * as zlib from 'zlib';
import { promisify } from 'util';

const gzip = promisify(zlib.gzip);
const gunzip = promisify(zlib.gunzip);
const deflate = promisify(zlib.deflate);
const inflate = promisify(zlib.inflate);
const brotliCompress = promisify(zlib.brotliCompress);
const brotliDecompress = promisify(zlib.brotliDecompress);

export enum CompressionType {
  GZIP = 'gzip',
  DEFLATE = 'deflate',
  BROTLI = 'brotli',
  CONSCIOUSNESS = 'consciousness',
  NONE = 'none',
}

export interface CompressionResult {
  data: Buffer;
  originalSize: number;
  compressedSize: number;
  compressionRatio: number;
  type: CompressionType;
}

@Injectable()
export class CompressionService {
  private readonly logger = new Logger(CompressionService.name);
  async compress(
    data: string | Buffer,
    type: CompressionType = CompressionType.GZIP,
  ): Promise<CompressionResult> {
    const buffer = Buffer.isBuffer(data) ? data : Buffer.from(data, 'utf-8');
    const originalSize = buffer.length;

    let compressed: Buffer;
    switch (type) {
      case CompressionType.GZIP:
        compressed = await gzip(buffer, { level: 9 });
        break;
      case CompressionType.DEFLATE:
        compressed = await deflate(buffer, { level: 9 });
        break;
      case CompressionType.BROTLI:
        compressed = await brotliCompress(buffer, {
          params: {
            [zlib.constants.BROTLI_PARAM_QUALITY]: 11,
            [zlib.constants.BROTLI_PARAM_SIZE_HINT]: buffer.length,
          },
        });
        break;
      case CompressionType.CONSCIOUSNESS:
        compressed = await this.compressConsciousnessData(buffer);
        break;
      case CompressionType.NONE:
        compressed = buffer;
        break;
      default:
        throw new Error(`Unknown compression type: ${type}`);
    }

    return {
      data: compressed,
      originalSize,
      compressedSize: compressed.length,
      compressionRatio: originalSize / compressed.length,
      type,
    };
  }

  async decompress(
    data: Buffer,
    type: CompressionType = CompressionType.GZIP,
  ): Promise<Buffer> {
    switch (type) {
      case CompressionType.GZIP:
        return await gunzip(data);
      case CompressionType.DEFLATE:
        return await inflate(data);
      case CompressionType.NONE:
        return data;
      default:
        throw new Error(`Unknown compression type: ${type}`);
    }
  }

  /**
   * Compress JSON data with optimal settings
   */
  async compressJson(data: any): Promise<CompressionResult> {
    const json = JSON.stringify(data);
    return this.compress(json, CompressionType.GZIP);
  }

  /**
   * Decompress JSON data
   */
  async decompressJson<T = any>(
    data: Buffer,
    type: CompressionType = CompressionType.GZIP,
  ): Promise<T> {
    const decompressed = await this.decompress(data, type);
    return JSON.parse(decompressed.toString('utf-8'));
  }

  /**
   * Calculate compression ratio for different types
   */
  async analyzeCompression(
    data: string | Buffer,
  ): Promise<Record<CompressionType, CompressionResult>> {
    const results: Record<CompressionType, CompressionResult> = {} as any;

    for (const type of Object.values(CompressionType)) {
      results[type] = await this.compress(data, type);
    }

    return results;
  }

  /**
   * Choose optimal compression based on data characteristics
   */
  async chooseOptimalCompression(
    data: string | Buffer,
    minRatio: number = 1.2,
  ): Promise<CompressionResult> {
    const analysis = await this.analyzeCompression(data);
    
    // Skip compression if data is too small
    const originalSize = Buffer.isBuffer(data) ? data.length : Buffer.byteLength(data, 'utf-8');
    if (originalSize < 1024) {
      return analysis[CompressionType.NONE];
    }

    // Choose best compression ratio
    let best = analysis[CompressionType.NONE];
    for (const result of Object.values(analysis)) {
      if (result.compressionRatio > minRatio && result.compressionRatio > best.compressionRatio) {
        best = result;
      }
    }

    return best;
  }

  /**
   * Specialized compression for consciousness state data
   */
  async compressConsciousnessState(data: string): Promise<string> {
    try {
      const parsed = JSON.parse(data);
      
      // Apply consciousness-specific compression techniques
      const compressed = this.applyConsciousnessCompression(parsed);
      
      // Use brotli for final compression
      const buffer = Buffer.from(JSON.stringify(compressed), 'utf-8');
      const finalCompressed = await brotliCompress(buffer, {
        params: {
          [zlib.constants.BROTLI_PARAM_QUALITY]: 11,
        },
      });
      
      return finalCompressed.toString('base64');
    } catch (error) {
      this.logger.error('Error in consciousness compression:', error);
      return data;
    }
  }

  /**
   * Decompress consciousness state data
   */
  async decompressConsciousnessState(compressedData: string): Promise<string> {
    try {
      // Decode from base64 and decompress with brotli
      const buffer = Buffer.from(compressedData, 'base64');
      const decompressed = await brotliDecompress(buffer);
      const jsonStr = decompressed.toString('utf-8');
      
      // Parse and expand consciousness data
      const compressed = JSON.parse(jsonStr);
      const expanded = this.expandConsciousnessData(compressed);
      
      return JSON.stringify(expanded);
    } catch (error) {
      this.logger.error('Error in consciousness decompression:', error);
      return compressedData;
    }
  }

  /**
   * Apply consciousness-specific compression techniques
   */
  private applyConsciousnessCompression(data: any): any {
    if (!data || typeof data !== 'object') {
      return data;
    }

    const compressed: any = {};

    for (const [key, value] of Object.entries(data)) {
      switch (key) {
        case 'stream_history':
        case 'experience_history':
          // Compress repetitive historical data
          compressed[key] = this.compressHistoricalData(value as any[]);
          break;
          
        case 'qualia':
        case 'thoughts':
          // Compress thought and qualia patterns
          compressed[key] = this.compressPatternData(value);
          break;
          
        case 'metadata':
        case 'context':
          // Remove redundant metadata
          compressed[key] = this.compressMetadata(value);
          break;
          
        case 'timestamp':
        case 'created_at':
        case 'updated_at':
          // Compress timestamps to relative format
          compressed[key] = this.compressTimestamp(value);
          break;
          
        default:
          // Recursively compress nested objects
          if (Array.isArray(value)) {
            compressed[key] = value.map(item => 
              typeof item === 'object' ? this.applyConsciousnessCompression(item) : item
            );
          } else if (typeof value === 'object' && value !== null) {
            compressed[key] = this.applyConsciousnessCompression(value);
          } else {
            compressed[key] = value;
          }
      }
    }

    return compressed;
  }

  /**
   * Expand compressed consciousness data
   */
  private expandConsciousnessData(compressed: any): any {
    if (!compressed || typeof compressed !== 'object') {
      return compressed;
    }

    const expanded: any = {};

    for (const [key, value] of Object.entries(compressed)) {
      switch (key) {
        case 'stream_history':
        case 'experience_history':
          expanded[key] = this.expandHistoricalData(value as any);
          break;
          
        case 'timestamp':
        case 'created_at':
        case 'updated_at':
          expanded[key] = this.expandTimestamp(value);
          break;
          
        default:
          if (Array.isArray(value)) {
            expanded[key] = value.map(item => 
              typeof item === 'object' ? this.expandConsciousnessData(item) : item
            );
          } else if (typeof value === 'object' && value !== null) {
            expanded[key] = this.expandConsciousnessData(value);
          } else {
            expanded[key] = value;
          }
      }
    }

    return expanded;
  }

  /**
   * Compress historical data by removing duplicates and patterns
   */
  private compressHistoricalData(history: any[]): any {
    if (!Array.isArray(history) || history.length === 0) {
      return history;
    }

    // Keep only recent entries and unique patterns
    const recent = history.slice(-100); // Last 100 entries
    const unique = this.removeDuplicatePatterns(recent);
    
    return {
      compressed: true,
      count: history.length,
      unique_entries: unique.length,
      data: unique,
    };
  }

  /**
   * Expand compressed historical data
   */
  private expandHistoricalData(compressed: any): any[] {
    if (!compressed || !compressed.compressed) {
      return compressed;
    }
    
    return compressed.data || [];
  }

  /**
   * Remove duplicate patterns from data array
   */
  private removeDuplicatePatterns(data: any[]): any[] {
    const seen = new Set();
    const unique: any[] = [];
    
    for (const item of data) {
      const key = this.generatePatternKey(item);
      if (!seen.has(key)) {
        seen.add(key);
        unique.push(item);
      }
    }
    
    return unique;
  }

  /**
   * Generate pattern key for deduplication
   */
  private generatePatternKey(item: any): string {
    if (typeof item === 'string') {
      return item;
    }
    
    if (typeof item === 'object' && item !== null) {
      // Create key from important fields
      const keyFields = ['type', 'category', 'content', 'name'];
      const keyParts = keyFields
        .filter(field => field in item)
        .map(field => `${field}:${item[field]}`)
        .join('|');
      
      return keyParts || JSON.stringify(item);
    }
    
    return String(item);
  }

  /**
   * Compress pattern data (qualia, thoughts, etc.)
   */
  private compressPatternData(data: any): any {
    if (Array.isArray(data)) {
      return this.removeDuplicatePatterns(data);
    }
    return data;
  }

  /**
   * Compress metadata by removing redundant fields
   */
  private compressMetadata(metadata: any): any {
    if (!metadata || typeof metadata !== 'object') {
      return metadata;
    }
    
    const compressed: any = {};
    const importantFields = [
      'id', 'type', 'category', 'priority', 'confidence',
      'intensity', 'quality', 'source'
    ];
    
    for (const field of importantFields) {
      if (field in metadata) {
        compressed[field] = metadata[field];
      }
    }
    
    return compressed;
  }

  /**
   * Compress timestamp to relative format
   */
  private compressTimestamp(timestamp: any): any {
    if (typeof timestamp === 'string') {
      try {
        const date = new Date(timestamp);
        const now = new Date();
        const diffMs = now.getTime() - date.getTime();
        
        // Store as milliseconds ago (more compact than full ISO string)
        return { rel: diffMs, base: now.getTime() };
      } catch {
        return timestamp;
      }
    }
    return timestamp;
  }

  /**
   * Expand relative timestamp back to absolute
   */
  private expandTimestamp(compressed: any): any {
    if (compressed && typeof compressed === 'object' && 'rel' in compressed && 'base' in compressed) {
      const absoluteTime = compressed.base - compressed.rel;
      return new Date(absoluteTime).toISOString();
    }
    return compressed;
  }

  /**
   * Compress consciousness data with specialized algorithm
   */
  private async compressConsciousnessData(buffer: Buffer): Promise<Buffer> {
    try {
      const data = buffer.toString('utf-8');
      const compressed = await this.compressConsciousnessState(data);
      return Buffer.from(compressed, 'utf-8');
    } catch (error) {
      this.logger.error('Error in consciousness data compression:', error);
      return buffer;
    }
  }
}
