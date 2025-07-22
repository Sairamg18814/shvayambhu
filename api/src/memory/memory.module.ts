import { Module } from '@nestjs/common';
import { TypeOrmModule } from '@nestjs/typeorm';
import { MemoryService } from './memory.service';
import { MemoryResolver } from './memory.resolver';
import { CompressedMemory } from '../common/entities/compressed-memory.entity';
import { CompressionService } from '../common/compression/compression.service';

@Module({
  imports: [TypeOrmModule.forFeature([CompressedMemory])],
  providers: [MemoryService, MemoryResolver, CompressionService],
  exports: [MemoryService],
})
export class MemoryModule {}
