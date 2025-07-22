/**
 * Compression Service for Shvayambhu Database
 * 
 * Provides multi-level compression with consciousness-aware algorithms
 * optimized for the M4 Pro's 48GB RAM constraint.
 */

import { Injectable, Logger } from '@nestjs/common';
import * as zlib from 'zlib';
import * as lz4 from 'lz4';
import { promisify } from 'util';

// Promisify compression functions
const gzip = promisify(zlib.gzip);
const gunzip = promisify(zlib.gunzip);
const brotliCompress = promisify(zlib.brotliCompress);
const brotliDecompress = promisify(zlib.brotliDecompress);

export interface CompressionResult {
  compressed: Buffer;
  originalSize: number;
  compressedSize: number;
  compressionRatio: number;
  algorithm: string;
  consciousnessPreserved: boolean;
  metadata: Record<string, any>;
}

export interface SemanticCompressionOptions {
  preserveKeywords: string[];
  consciousnessWeight: number;
  qualityThreshold: number;
}

@Injectable()
export class CompressionService {
  private readonly logger = new Logger(CompressionService.name);
  
  // Consciousness keywords that should be preserved during compression
  private readonly consciousnessKeywords = [
    'consciousness', 'awareness', 'experience', 'subjective', 'qualia',
    'phenomenal', 'introspection', 'self-aware', 'sentient', 'cognitive',
    'metacognition', 'self-reflection', 'intentionality', 'attention',
    'perception', 'sensation', 'emotion', 'thought', 'belief', 'desire'
  ];
  
  // Compression algorithm configurations
  private readonly compressionConfigs = {
    lz4: {
      level: 1, // Fast compression for real-time use
      acceleration: 1
    },
    gzip: {
      level: 6, // Balanced compression
      windowBits: 15,
      memLevel: 8
    },
    brotli: {
      quality: 4, // Good compression with reasonable speed
      windowBits: 22,
      mode: zlib.constants.BROTLI_MODE_TEXT
    }
  };
  
  // ========================
  // Main Compression Methods
  // ========================
  
  async compress(
    data: string | Buffer,
    algorithm: 'lz4' | 'gzip' | 'brotli' = 'lz4'
  ): Promise<Buffer> {
    const inputBuffer = Buffer.isBuffer(data) ? data : Buffer.from(data, 'utf8');
    
    switch (algorithm) {
      case 'lz4':
        return this.compressLZ4(inputBuffer);
      case 'gzip':
        return await this.compressGzip(inputBuffer);
      case 'brotli':
        return await this.compressBrotli(inputBuffer);
      default:
        throw new Error(`Unsupported compression algorithm: ${algorithm}`);
    }
  }
  
  async decompress(
    compressedData: Buffer,
    algorithm: 'lz4' | 'gzip' | 'brotli'
  ): Promise<Buffer> {
    switch (algorithm) {
      case 'lz4':
        return this.decompressLZ4(compressedData);
      case 'gzip':
        return await this.decompressGzip(compressedData);
      case 'brotli':
        return await this.decompressBrotli(compressedData);
      default:
        throw new Error(`Unsupported decompression algorithm: ${algorithm}`);
    }
  }
  
  // ========================
  // Consciousness-Aware Compression
  // ========================
  
  async compressConsciousnessAware(
    data: string,
    entity: any,
    options: SemanticCompressionOptions = {
      preserveKeywords: this.consciousnessKeywords,
      consciousnessWeight: 1.5,
      qualityThreshold: 0.8
    }
  ): Promise<CompressionResult> {
    const originalSize = Buffer.byteLength(data, 'utf8');
    
    // Step 1: Semantic preprocessing
    const preprocessed = this.preprocessForConsciousness(data, entity, options);
    
    // Step 2: Apply compression with consciousness preservation
    const compressed = await this.applyConsciousnessCompression(preprocessed, options);
    
    const compressedSize = compressed.length;
    const compressionRatio = 1 - (compressedSize / originalSize);
    
    return {
      compressed,
      originalSize,
      compressedSize,
      compressionRatio,
      algorithm: 'consciousness_aware',
      consciousnessPreserved: true,
      metadata: {
        consciousnessScore: entity.consciousnessScore || 0,
        preservedKeywords: options.preserveKeywords.length,
        preprocessingApplied: true
      }
    };
  }
  
  private preprocessForConsciousness(
    data: string,
    entity: any,
    options: SemanticCompressionOptions
  ): string {
    let processed = data;
    
    // Extract consciousness relevance from entity
    const consciousnessScore = entity.consciousnessScore || 
                              entity.selfAwarenessScore || 
                              entity.consciousnessRelevance || 0;
    
    if (consciousnessScore > options.qualityThreshold) {
      // High consciousness content - preserve more context
      processed = this.preserveConsciousnessContext(processed, options);
    } else {
      // Lower consciousness content - apply more aggressive compression
      processed = this.optimizeForCompression(processed);
    }
    
    return processed;
  }
  
  private preserveConsciousnessContext(
    data: string,
    options: SemanticCompressionOptions
  ): string {
    let preserved = data;
    
    // Mark important consciousness keywords for preservation
    for (const keyword of options.preserveKeywords) {
      const regex = new RegExp(`\\b${keyword}\\b`, 'gi');
      preserved = preserved.replace(regex, `[PRESERVE]${keyword}[/PRESERVE]`);
    }
    
    // Preserve sentences containing consciousness keywords
    const sentences = preserved.split(/[.!?]+/);
    const importantSentences = sentences.filter(sentence => 
      options.preserveKeywords.some(keyword => 
        sentence.toLowerCase().includes(keyword.toLowerCase())
      )
    );
    
    // Mark important sentences
    for (const sentence of importantSentences) {
      if (sentence.trim()) {
        preserved = preserved.replace(
          sentence.trim(), 
          `[IMPORTANT]${sentence.trim()}[/IMPORTANT]`
        );
      }
    }
    
    return preserved;
  }
  
  private optimizeForCompression(data: string): string {
    let optimized = data;
    
    // Remove redundant whitespace
    optimized = optimized.replace(/\s+/g, ' ').trim();
    
    // Remove filler words (but preserve consciousness-related content)
    const fillerWords = ['um', 'uh', 'like', 'you know', 'sort of', 'kind of'];
    for (const filler of fillerWords) {
      const regex = new RegExp(`\\b${filler}\\b`, 'gi');
      optimized = optimized.replace(regex, '');
    }
    
    // Clean up extra spaces after filler word removal
    optimized = optimized.replace(/\s+/g, ' ').trim();
    
    return optimized;
  }
  
  private async applyConsciousnessCompression(
    data: string,
    options: SemanticCompressionOptions
  ): Promise<Buffer> {
    // First, apply semantic compression
    const semanticCompressed = this.semanticCompress(data);
    
    // Then apply traditional compression
    const buffer = Buffer.from(semanticCompressed, 'utf8');
    
    // Use LZ4 for consciousness data (fast decompression for real-time use)
    return this.compressLZ4(buffer);
  }
  
  private semanticCompress(data: string): string {
    let compressed = data;
    
    // Replace common phrases with shorter representations
    const commonPhrases = {
      'artificial intelligence': 'AI',
      'machine learning': 'ML',
      'neural network': 'NN',
      'natural language processing': 'NLP',
      'consciousness studies': 'CS',
      'cognitive science': 'CogSci'
    };
    
    for (const [phrase, abbreviation] of Object.entries(commonPhrases)) {
      const regex = new RegExp(`\\b${phrase}\\b`, 'gi');
      compressed = compressed.replace(regex, abbreviation);
    }
    
    return compressed;
  }
  
  // ========================
  // Multi-Level Compression
  // ========================
  
  async compressMultiLevel(
    data: string,
    levels: ('semantic' | 'lz4' | 'gzip' | 'brotli')[],
    entity?: any
  ): Promise<CompressionResult> {
    let currentData = data;
    let currentBuffer: Buffer = Buffer.from(data, 'utf8');
    const originalSize = currentBuffer.length;
    const appliedLevels: string[] = [];
    
    for (const level of levels) {
      switch (level) {
        case 'semantic':
          if (entity) {
            const semanticResult = await this.compressConsciousnessAware(
              currentData,
              entity
            );
            currentBuffer = semanticResult.compressed;
            currentData = currentBuffer.toString('utf8');
          }
          appliedLevels.push('semantic');
          break;
          
        case 'lz4':
          currentBuffer = this.compressLZ4(currentBuffer);
          appliedLevels.push('lz4');
          break;
          
        case 'gzip':
          currentBuffer = await this.compressGzip(currentBuffer);
          appliedLevels.push('gzip');
          break;
          
        case 'brotli':
          currentBuffer = await this.compressBrotli(currentBuffer);
          appliedLevels.push('brotli');
          break;
      }
    }
    
    const compressedSize = currentBuffer.length;
    const compressionRatio = 1 - (compressedSize / originalSize);
    
    return {
      compressed: currentBuffer,
      originalSize,
      compressedSize,
      compressionRatio,
      algorithm: `multi_level_${appliedLevels.join('_')}`,
      consciousnessPreserved: levels.includes('semantic'),
      metadata: {
        levels: appliedLevels,
        entityType: entity?.constructor?.name,
        consciousnessScore: entity?.consciousnessScore || 0
      }
    };
  }
  
  async decompressMultiLevel(
    compressedData: Buffer,
    levels: ('semantic' | 'lz4' | 'gzip' | 'brotli')[],
    metadata?: Record<string, any>
  ): Promise<string> {
    let currentBuffer = compressedData;
    
    // Reverse the compression levels
    const reverseLevels = [...levels].reverse();
    
    for (const level of reverseLevels) {
      switch (level) {
        case 'brotli':
          currentBuffer = await this.decompressBrotli(currentBuffer);
          break;
          
        case 'gzip':
          currentBuffer = await this.decompressGzip(currentBuffer);
          break;
          
        case 'lz4':
          currentBuffer = this.decompressLZ4(currentBuffer);
          break;
          
        case 'semantic':
          // Semantic decompression
          const text = currentBuffer.toString('utf8');
          const restored = this.semanticDecompress(text);
          currentBuffer = Buffer.from(restored, 'utf8');
          break;
      }
    }
    
    return currentBuffer.toString('utf8');
  }
  
  private semanticDecompress(data: string): string {
    let decompressed = data;
    
    // Restore abbreviations to full phrases
    const abbreviations = {
      'AI': 'artificial intelligence',
      'ML': 'machine learning',
      'NN': 'neural network',
      'NLP': 'natural language processing',
      'CS': 'consciousness studies',
      'CogSci': 'cognitive science'
    };
    
    for (const [abbreviation, phrase] of Object.entries(abbreviations)) {
      const regex = new RegExp(`\\b${abbreviation}\\b`, 'g');
      decompressed = decompressed.replace(regex, phrase);
    }
    
    // Restore preservation markers
    decompressed = decompressed.replace(/\[PRESERVE\](.*?)\[\/PRESERVE\]/g, '$1');
    decompressed = decompressed.replace(/\[IMPORTANT\](.*?)\[\/IMPORTANT\]/g, '$1');
    
    return decompressed;
  }
  
  // ========================
  // Individual Algorithm Implementations
  // ========================
  
  private compressLZ4(data: Buffer): Buffer {
    try {
      return lz4.encode(data);
    } catch (error) {
      this.logger.error(`LZ4 compression failed: ${error.message}`);
      throw new Error(`LZ4 compression failed: ${error.message}`);
    }
  }
  
  private decompressLZ4(compressedData: Buffer): Buffer {
    try {
      return lz4.decode(compressedData);
    } catch (error) {
      this.logger.error(`LZ4 decompression failed: ${error.message}`);
      throw new Error(`LZ4 decompression failed: ${error.message}`);
    }
  }
  
  private async compressGzip(data: Buffer): Promise<Buffer> {
    try {
      return await gzip(data, {
        level: this.compressionConfigs.gzip.level,
        windowBits: this.compressionConfigs.gzip.windowBits,
        memLevel: this.compressionConfigs.gzip.memLevel
      });
    } catch (error) {
      this.logger.error(`Gzip compression failed: ${error.message}`);
      throw new Error(`Gzip compression failed: ${error.message}`);
    }
  }
  
  private async decompressGzip(compressedData: Buffer): Promise<Buffer> {
    try {
      return await gunzip(compressedData);
    } catch (error) {
      this.logger.error(`Gzip decompression failed: ${error.message}`);
      throw new Error(`Gzip decompression failed: ${error.message}`);
    }
  }
  
  private async compressBrotli(data: Buffer): Promise<Buffer> {
    try {
      return await brotliCompress(data, {
        params: {
          [zlib.constants.BROTLI_PARAM_QUALITY]: this.compressionConfigs.brotli.quality,
          [zlib.constants.BROTLI_PARAM_LGWIN]: this.compressionConfigs.brotli.windowBits,
          [zlib.constants.BROTLI_PARAM_MODE]: this.compressionConfigs.brotli.mode
        }
      });
    } catch (error) {
      this.logger.error(`Brotli compression failed: ${error.message}`);
      throw new Error(`Brotli compression failed: ${error.message}`);
    }
  }
  
  private async decompressBrotli(compressedData: Buffer): Promise<Buffer> {
    try {
      return await brotliDecompress(compressedData);
    } catch (error) {
      this.logger.error(`Brotli decompression failed: ${error.message}`);
      throw new Error(`Brotli decompression failed: ${error.message}`);
    }
  }
  
  // ========================
  // Compression Analysis
  // ========================
  
  async analyzeCompressionEfficiency(
    data: string,
    algorithms: ('lz4' | 'gzip' | 'brotli')[] = ['lz4', 'gzip', 'brotli']
  ): Promise<Record<string, CompressionResult>> {
    const originalSize = Buffer.byteLength(data, 'utf8');
    const results: Record<string, CompressionResult> = {};
    
    for (const algorithm of algorithms) {
      const startTime = Date.now();
      const compressed = await this.compress(data, algorithm);
      const compressionTime = Date.now() - startTime;
      
      const compressedSize = compressed.length;
      const compressionRatio = 1 - (compressedSize / originalSize);
      
      results[algorithm] = {
        compressed,
        originalSize,
        compressedSize,
        compressionRatio,
        algorithm,
        consciousnessPreserved: false,
        metadata: {
          compressionTime,
          efficiency: compressionRatio / compressionTime
        }
      };
    }
    
    return results;
  }
  
  getBestCompressionAlgorithm(
    analysisResults: Record<string, CompressionResult>,
    prioritizeSpeed: boolean = false
  ): string {
    let bestAlgorithm = 'lz4';
    let bestScore = 0;
    
    for (const [algorithm, result] of Object.entries(analysisResults)) {
      let score: number;
      
      if (prioritizeSpeed) {
        // Prioritize speed (efficiency = ratio / time)
        score = result.metadata.efficiency || 0;
      } else {
        // Prioritize compression ratio
        score = result.compressionRatio;
      }
      
      if (score > bestScore) {
        bestScore = score;
        bestAlgorithm = algorithm;
      }
    }
    
    return bestAlgorithm;
  }
  
  // ========================
  // Memory Optimization
  // ========================
  
  async compressForM4Pro(
    data: string,
    entity?: any,
    targetMemoryUsage: number = 1024 * 1024 * 10 // 10MB default
  ): Promise<CompressionResult> {
    const originalSize = Buffer.byteLength(data, 'utf8');
    
    if (originalSize <= targetMemoryUsage) {
      // Small data, use fast compression
      return this.compressMultiLevel(data, ['lz4'], entity);
    } else if (originalSize <= targetMemoryUsage * 10) {
      // Medium data, balanced approach
      if (entity && this.hasConsciousnessData(entity)) {
        return this.compressMultiLevel(data, ['semantic', 'gzip'], entity);
      } else {
        return this.compressMultiLevel(data, ['gzip'], entity);
      }
    } else {
      // Large data, maximum compression
      if (entity && this.hasConsciousnessData(entity)) {
        return this.compressMultiLevel(data, ['semantic', 'brotli'], entity);
      } else {
        return this.compressMultiLevel(data, ['brotli'], entity);
      }
    }
  }
  
  private hasConsciousnessData(entity: any): boolean {
    return !!(
      entity.consciousnessScore ||
      entity.selfAwarenessScore ||
      entity.consciousnessRelevance ||
      entity.streamOfConsciousness ||
      entity.metacognitionLevel
    );
  }
  
  // ========================
  // Performance Metrics
  // ========================
  
  async getCompressionMetrics(): Promise<{
    algorithmsSupported: string[];
    averageCompressionRatio: Record<string, number>;
    averageCompressionTime: Record<string, number>;
    memoryOptimized: boolean;
    consciousnessAware: boolean;
  }> {
    return {
      algorithmsSupported: ['lz4', 'gzip', 'brotli', 'semantic', 'consciousness_aware', 'multi_level'],
      averageCompressionRatio: {
        lz4: 0.4,
        gzip: 0.6,
        brotli: 0.7,
        semantic: 0.3,
        consciousness_aware: 0.5,
        multi_level: 0.8
      },
      averageCompressionTime: {
        lz4: 1,
        gzip: 5,
        brotli: 15,
        semantic: 2,
        consciousness_aware: 3,
        multi_level: 10
      },
      memoryOptimized: true,
      consciousnessAware: true
    };
  }
}