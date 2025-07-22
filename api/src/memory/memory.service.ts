import { Injectable, NotFoundException } from '@nestjs/common';
import { InjectRepository } from '@nestjs/typeorm';
import { Repository } from 'typeorm';
import { CompressedMemory, MemoryType } from '../common/entities/compressed-memory.entity';
import { CompressionService, CompressionType } from '../common/compression/compression.service';
import * as crypto from 'crypto';

export interface StoreMemoryInput {
  type: MemoryType;
  data: any;
  category?: string;
  metadata?: Record<string, any>;
  consciousnessLevel?: number;
  emotionalState?: Record<string, any>;
  thoughtSummary?: string;
}

export interface RetrieveMemoryOptions {
  type?: MemoryType;
  category?: string;
  limit?: number;
  offset?: number;
  includeMetadata?: boolean;
}

@Injectable()
export class MemoryService {
  constructor(
    @InjectRepository(CompressedMemory)
    private memoryRepository: Repository<CompressedMemory>,
    private compressionService: CompressionService,
  ) {}

  /**
   * Store a memory with automatic compression
   */
  async storeMemory(input: StoreMemoryInput): Promise<CompressedMemory> {
    // Convert data to JSON string for compression
    const dataStr = JSON.stringify(input.data);
    
    // Choose optimal compression
    const compressed = await this.compressionService.chooseOptimalCompression(dataStr);
    
    // Calculate checksum
    const checksum = crypto
      .createHash('sha256')
      .update(dataStr)
      .digest('hex');
    
    // Create memory entity
    const memory = this.memoryRepository.create({
      type: input.type,
      category: input.category,
      compressedData: compressed.data,
      originalSize: compressed.originalSize,
      compressedSize: compressed.compressedSize,
      compressionRatio: compressed.compressionRatio,
      compressionType: compressed.type,
      metadata: input.metadata,
      checksum,
      consciousnessLevel: input.consciousnessLevel,
      emotionalState: input.emotionalState,
      thoughtSummary: input.thoughtSummary,
    });
    
    return await this.memoryRepository.save(memory);
  }

  /**
   * Retrieve and decompress memories
   */
  async retrieveMemories(options: RetrieveMemoryOptions = {}): Promise<any[]> {
    const query = this.memoryRepository.createQueryBuilder('memory');
    
    if (options.type) {
      query.andWhere('memory.type = :type', { type: options.type });
    }
    
    if (options.category) {
      query.andWhere('memory.category = :category', { category: options.category });
    }
    
    query
      .orderBy('memory.createdAt', 'DESC')
      .limit(options.limit || 10)
      .offset(options.offset || 0);
    
    const memories = await query.getMany();
    
    // Decompress and return data
    const results = [];
    for (const memory of memories) {
      // Update access statistics
      await this.memoryRepository.update(memory.id, {
        accessCount: memory.accessCount + 1,
        lastAccessedAt: new Date(),
      });
      
      // Decompress data
      const decompressed = await this.compressionService.decompressJson(
        memory.compressedData,
        memory.compressionType,
      );
      
      if (options.includeMetadata) {
        results.push({
          id: memory.id,
          type: memory.type,
          category: memory.category,
          data: decompressed,
          metadata: memory.metadata,
          consciousnessLevel: memory.consciousnessLevel,
          emotionalState: memory.emotionalState,
          thoughtSummary: memory.thoughtSummary,
          createdAt: memory.createdAt,
          compressionRatio: memory.compressionRatio,
        });
      } else {
        results.push(decompressed);
      }
    }
    
    return results;
  }

  /**
   * Get memory by ID
   */
  async getMemory(id: string): Promise<any> {
    const memory = await this.memoryRepository.findOne({ where: { id } });
    
    if (!memory) {
      throw new NotFoundException(`Memory with ID ${id} not found`);
    }
    
    // Update access statistics
    await this.memoryRepository.update(memory.id, {
      accessCount: memory.accessCount + 1,
      lastAccessedAt: new Date(),
    });
    
    // Decompress and return
    return await this.compressionService.decompressJson(
      memory.compressedData,
      memory.compressionType,
    );
  }

  /**
   * Get compression statistics
   */
  async getCompressionStats(): Promise<any> {
    const stats = await this.memoryRepository
      .createQueryBuilder('memory')
      .select('memory.type', 'type')
      .addSelect('COUNT(*)', 'count')
      .addSelect('SUM(memory.originalSize)', 'totalOriginalSize')
      .addSelect('SUM(memory.compressedSize)', 'totalCompressedSize')
      .addSelect('AVG(memory.compressionRatio)', 'avgCompressionRatio')
      .groupBy('memory.type')
      .getRawMany();
    
    const totalStats = await this.memoryRepository
      .createQueryBuilder('memory')
      .select('COUNT(*)', 'totalCount')
      .addSelect('SUM(memory.originalSize)', 'totalOriginalSize')
      .addSelect('SUM(memory.compressedSize)', 'totalCompressedSize')
      .getRawOne();
    
    return {
      byType: stats,
      total: {
        ...totalStats,
        compressionRatio: totalStats.totalOriginalSize / totalStats.totalCompressedSize,
        savedBytes: totalStats.totalOriginalSize - totalStats.totalCompressedSize,
        savedPercentage: ((totalStats.totalOriginalSize - totalStats.totalCompressedSize) / totalStats.totalOriginalSize) * 100,
      },
    };
  }

  /**
   * Consolidate old memories to save space
   */
  async consolidateMemories(
    olderThanDays: number = 30,
    minAccessCount: number = 2,
  ): Promise<number> {
    const cutoffDate = new Date();
    cutoffDate.setDate(cutoffDate.getDate() - olderThanDays);
    
    // Find memories to consolidate
    const memories = await this.memoryRepository
      .createQueryBuilder('memory')
      .where('memory.createdAt < :cutoffDate', { cutoffDate })
      .andWhere('memory.accessCount < :minAccessCount', { minAccessCount })
      .andWhere('memory.type != :type', { type: MemoryType.MODEL_WEIGHTS })
      .getMany();
    
    if (memories.length === 0) {
      return 0;
    }
    
    // Group by type and category for consolidation
    const groups = new Map<string, CompressedMemory[]>();
    for (const memory of memories) {
      const key = `${memory.type}:${memory.category || 'default'}`;
      if (!groups.has(key)) {
        groups.set(key, []);
      }
      groups.get(key)!.push(memory);
    }
    
    let consolidatedCount = 0;
    
    // Consolidate each group
    for (const [key, groupMemories] of groups) {
      if (groupMemories.length < 2) continue;
      
      const [type, category] = key.split(':');
      
      // Decompress all memories in group
      const allData = [];
      for (const memory of groupMemories) {
        const data = await this.compressionService.decompressJson(
          memory.compressedData,
          memory.compressionType,
        );
        allData.push({
          data,
          metadata: memory.metadata,
          createdAt: memory.createdAt,
        });
      }
      
      // Create consolidated memory
      await this.storeMemory({
        type: type as MemoryType,
        category: category === 'default' ? undefined : category,
        data: {
          consolidated: true,
          count: allData.length,
          dateRange: {
            from: groupMemories[groupMemories.length - 1].createdAt,
            to: groupMemories[0].createdAt,
          },
          memories: allData,
        },
        metadata: {
          consolidatedAt: new Date(),
          originalCount: allData.length,
        },
      });
      
      // Delete original memories
      await this.memoryRepository.remove(groupMemories);
      consolidatedCount += groupMemories.length;
    }
    
    return consolidatedCount;
  }

  /**
   * Clean up duplicate memories based on checksum
   */
  async deduplicateMemories(): Promise<number> {
    const duplicates = await this.memoryRepository
      .createQueryBuilder('memory')
      .select('memory.checksum', 'checksum')
      .addSelect('COUNT(*)', 'count')
      .groupBy('memory.checksum')
      .having('COUNT(*) > 1')
      .getRawMany();
    
    let removedCount = 0;
    
    for (const dup of duplicates) {
      // Keep the newest, remove others
      const memories = await this.memoryRepository.find({
        where: { checksum: dup.checksum },
        order: { createdAt: 'DESC' },
      });
      
      if (memories.length > 1) {
        const toRemove = memories.slice(1);
        await this.memoryRepository.remove(toRemove);
        removedCount += toRemove.length;
      }
    }
    
    return removedCount;
  }
}
