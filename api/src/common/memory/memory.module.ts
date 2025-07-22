/**
 * Memory Module for Shvayambhu LLM System
 * 
 * Provides comprehensive memory management, optimization, and health monitoring
 * specifically designed for M4 Pro 48GB memory constraints.
 */

import { Module, Global } from '@nestjs/common';
import { ConfigModule } from '@nestjs/config';
import { ScheduleModule } from '@nestjs/schedule';
import { EventEmitterModule } from '@nestjs/event-emitter';

// Import database module for dependencies
import { DatabaseModule } from '../database/database.module';

// Import memory management services
import { MemoryOptimizerService } from './memory-optimizer.service';
import { MemoryHealthService } from './memory-health.service';

// Import controllers and resolvers
import { MemoryController } from './memory.controller';
import { MemoryResolver } from './memory.resolver';

@Global()
@Module({
  imports: [
    ConfigModule,
    ScheduleModule.forRoot(),
    EventEmitterModule.forRoot({
      // Configure event emitter
      wildcard: true,
      delimiter: '.',
      newListener: false,
      removeListener: false,
      maxListeners: 20,
      verboseMemoryLeak: true,
      ignoreErrors: false
    }),
    DatabaseModule
  ],
  providers: [
    MemoryOptimizerService,
    MemoryHealthService,
    MemoryResolver
  ],
  controllers: [
    MemoryController
  ],
  exports: [
    MemoryOptimizerService,
    MemoryHealthService
  ]
})
export class MemoryModule {
  constructor(
    private readonly memoryOptimizer: MemoryOptimizerService,
    private readonly memoryHealth: MemoryHealthService
  ) {}
}