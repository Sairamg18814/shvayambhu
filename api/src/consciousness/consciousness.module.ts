/**
 * Consciousness Module
 * 
 * Module for consciousness state management system
 */

import { Module } from '@nestjs/common';
import { TypeOrmModule } from '@nestjs/typeorm';
import {
  ConsciousnessState,
  AttentionFocus,
  QualiaState,
  StreamOfConsciousnessEntry,
  SelfModel,
  ExistentialState,
  ConsciousnessSnapshot,
  MetacognitiveReflection
} from '../common/schemas/consciousness-state.schema';
import { ConsciousnessService } from './consciousness.service';
import { ConsciousnessResolver } from './consciousness.resolver';
import { CompressionService } from '../common/compression/compression.service';

@Module({
  imports: [
    TypeOrmModule.forFeature([
      ConsciousnessState,
      AttentionFocus,
      QualiaState,
      StreamOfConsciousnessEntry,
      SelfModel,
      ExistentialState,
      ConsciousnessSnapshot,
      MetacognitiveReflection
    ])
  ],
  providers: [
    ConsciousnessService,
    ConsciousnessResolver,
    CompressionService
  ],
  exports: [ConsciousnessService]
})
export class ConsciousnessModule {}