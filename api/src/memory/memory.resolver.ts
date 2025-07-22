import { Resolver, Query, Mutation, Args, Int } from '@nestjs/graphql';
import { MemoryService, StoreMemoryInput, RetrieveMemoryOptions } from './memory.service';
import { MemoryType } from '../common/entities/compressed-memory.entity';

@Resolver()
export class MemoryResolver {
  constructor(private memoryService: MemoryService) {}

  @Query(() => String)
  async hello(): Promise<string> {
    return 'Shvayambhu GraphQL API with Compression';
  }

  @Mutation(() => String)
  async storeMemory(
    @Args('type') type: string,
    @Args('data') data: string,
    @Args('category', { nullable: true }) category?: string,
    @Args('metadata', { nullable: true }) metadata?: string,
  ): Promise<string> {
    const memory = await this.memoryService.storeMemory({
      type: type as MemoryType,
      data: JSON.parse(data),
      category,
      metadata: metadata ? JSON.parse(metadata) : undefined,
    });
    
    return memory.id;
  }

  @Query(() => String)
  async retrieveMemories(
    @Args('type', { nullable: true }) type?: string,
    @Args('category', { nullable: true }) category?: string,
    @Args('limit', { type: () => Int, nullable: true }) limit?: number,
    @Args('offset', { type: () => Int, nullable: true }) offset?: number,
  ): Promise<string> {
    const memories = await this.memoryService.retrieveMemories({
      type: type as MemoryType,
      category,
      limit,
      offset,
      includeMetadata: true,
    });
    
    return JSON.stringify(memories);
  }

  @Query(() => String)
  async getMemory(@Args('id') id: string): Promise<string> {
    const memory = await this.memoryService.getMemory(id);
    return JSON.stringify(memory);
  }

  @Query(() => String)
  async compressionStats(): Promise<string> {
    const stats = await this.memoryService.getCompressionStats();
    return JSON.stringify(stats);
  }

  @Mutation(() => Int)
  async consolidateMemories(
    @Args('olderThanDays', { type: () => Int, nullable: true }) olderThanDays?: number,
    @Args('minAccessCount', { type: () => Int, nullable: true }) minAccessCount?: number,
  ): Promise<number> {
    return await this.memoryService.consolidateMemories(olderThanDays, minAccessCount);
  }

  @Mutation(() => Int)
  async deduplicateMemories(): Promise<number> {
    return await this.memoryService.deduplicateMemories();
  }
}
