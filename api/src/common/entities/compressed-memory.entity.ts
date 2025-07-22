import { Entity, Column, PrimaryGeneratedColumn, CreateDateColumn, UpdateDateColumn, Index } from 'typeorm';
import { CompressionType } from '../compression/compression.service';

export enum MemoryType {
  TRAINING_DATA = 'training_data',
  MODEL_WEIGHTS = 'model_weights',
  CONSCIOUSNESS_STATE = 'consciousness_state',
  EXPERIENCE = 'experience',
  KNOWLEDGE = 'knowledge',
  CONVERSATION = 'conversation',
  WEB_CACHE = 'web_cache',
}

@Entity('compressed_memories')
@Index(['type', 'category'])
@Index(['createdAt'])
export class CompressedMemory {
  @PrimaryGeneratedColumn('uuid')
  id: string;

  @Column({ type: 'varchar', length: 50 })
  type: MemoryType;

  @Column({ type: 'varchar', length: 100, nullable: true })
  category?: string;

  @Column({ type: 'blob' })
  compressedData: Buffer;

  @Column({ type: 'int' })
  originalSize: number;

  @Column({ type: 'int' })
  compressedSize: number;

  @Column({ type: 'float' })
  compressionRatio: number;

  @Column({ type: 'varchar', length: 20 })
  compressionType: CompressionType;

  @Column({ type: 'json', nullable: true })
  metadata?: Record<string, any>;

  @Column({ type: 'text', nullable: true })
  checksum?: string;

  @Column({ type: 'int', default: 0 })
  accessCount: number;

  @CreateDateColumn()
  createdAt: Date;

  @UpdateDateColumn()
  updatedAt: Date;

  @Column({ type: 'datetime', nullable: true })
  lastAccessedAt?: Date;

  // Consciousness-specific fields
  @Column({ type: 'float', nullable: true })
  consciousnessLevel?: number;

  @Column({ type: 'json', nullable: true })
  emotionalState?: Record<string, any>;

  @Column({ type: 'text', nullable: true })
  thoughtSummary?: string;
}
