/**
 * Consciousness Service
 * 
 * Core service for managing consciousness state operations
 */

import { Injectable, Logger, NotFoundException } from '@nestjs/common';
import { InjectRepository } from '@nestjs/typeorm';
import { Repository, Between, MoreThanOrEqual, LessThanOrEqual } from 'typeorm';
import {
  ConsciousnessState,
  AttentionFocus,
  QualiaState,
  StreamOfConsciousnessEntry,
  SelfModel,
  ExistentialState,
  ConsciousnessSnapshot,
  MetacognitiveReflection,
  ConsciousnessLevel,
  EmotionalState
} from '../common/schemas/consciousness-state.schema';
import {
  CreateConsciousnessStateInput,
  UpdateConsciousnessStateInput,
  CreateConsciousnessSnapshotInput,
  CreateMetacognitiveReflectionInput,
  ConsciousnessStateFilter,
  ConsciousnessStatePagination
} from '../common/schemas/consciousness-inputs.dto';
import { CompressionService } from '../common/compression/compression.service';

@Injectable()
export class ConsciousnessService {
  private readonly logger = new Logger(ConsciousnessService.name);

  constructor(
    @InjectRepository(ConsciousnessState)
    private consciousnessStateRepository: Repository<ConsciousnessState>,
    
    @InjectRepository(AttentionFocus)
    private attentionFocusRepository: Repository<AttentionFocus>,
    
    @InjectRepository(QualiaState)
    private qualiaStateRepository: Repository<QualiaState>,
    
    @InjectRepository(StreamOfConsciousnessEntry)
    private streamRepository: Repository<StreamOfConsciousnessEntry>,
    
    @InjectRepository(SelfModel)
    private selfModelRepository: Repository<SelfModel>,
    
    @InjectRepository(ExistentialState)
    private existentialStateRepository: Repository<ExistentialState>,
    
    @InjectRepository(ConsciousnessSnapshot)
    private snapshotRepository: Repository<ConsciousnessSnapshot>,
    
    @InjectRepository(MetacognitiveReflection)
    private metacognitiveRepository: Repository<MetacognitiveReflection>,
    
    private compressionService: CompressionService
  ) {}

  async createConsciousnessState(input: CreateConsciousnessStateInput): Promise<ConsciousnessState> {
    this.logger.log('Creating new consciousness state');
    
    const consciousnessState = new ConsciousnessState();
    Object.assign(consciousnessState, input);

    // Handle related entities
    if (input.attention_foci) {
      consciousnessState.attention_foci = await Promise.all(
        input.attention_foci.map(async (focusInput) => {
          const focus = new AttentionFocus();
          Object.assign(focus, focusInput);
          return await this.attentionFocusRepository.save(focus);
        })
      );
    }

    if (input.qualia_states) {
      consciousnessState.qualia_states = await Promise.all(
        input.qualia_states.map(async (qualiaInput) => {
          const qualia = new QualiaState();
          Object.assign(qualia, qualiaInput);
          return await this.qualiaStateRepository.save(qualia);
        })
      );
    }

    if (input.stream_of_consciousness) {
      const streamEntry = new StreamOfConsciousnessEntry();
      Object.assign(streamEntry, input.stream_of_consciousness);
      consciousnessState.stream_of_consciousness = await this.streamRepository.save(streamEntry);
    }

    if (input.self_model) {
      const selfModel = new SelfModel();
      Object.assign(selfModel, input.self_model);
      consciousnessState.self_model = await this.selfModelRepository.save(selfModel);
    }

    if (input.existential_state) {
      const existentialState = new ExistentialState();
      Object.assign(existentialState, input.existential_state);
      consciousnessState.existential_state = await this.existentialStateRepository.save(existentialState);
    }

    // Compress state data for storage efficiency
    await this.compressConsciousnessData(consciousnessState);

    const savedState = await this.consciousnessStateRepository.save(consciousnessState);
    
    this.logger.log(`Created consciousness state with ID: ${savedState.id}`);
    return savedState;
  }

  async updateConsciousnessState(
    id: string, 
    input: UpdateConsciousnessStateInput
  ): Promise<ConsciousnessState> {
    const existingState = await this.findConsciousnessStateById(id);
    
    Object.assign(existingState, input);
    
    // Update compression if significant changes
    if (this.isSignificantUpdate(input)) {
      await this.compressConsciousnessData(existingState);
    }
    
    const updatedState = await this.consciousnessStateRepository.save(existingState);
    
    this.logger.log(`Updated consciousness state: ${id}`);
    return updatedState;
  }

  async findConsciousnessStateById(id: string): Promise<ConsciousnessState> {
    const state = await this.consciousnessStateRepository.findOne({
      where: { id },
      relations: [
        'attention_foci',
        'qualia_states',
        'stream_of_consciousness',
        'self_model',
        'existential_state'
      ]
    });

    if (!state) {
      throw new NotFoundException(`Consciousness state with ID ${id} not found`);
    }

    // Decompress data if needed
    await this.decompressConsciousnessData(state);
    
    return state;
  }

  async findConsciousnessStates(
    filter?: ConsciousnessStateFilter,
    pagination?: ConsciousnessStatePagination
  ): Promise<{ states: ConsciousnessState[]; total: number }> {
    const queryBuilder = this.consciousnessStateRepository.createQueryBuilder('cs')
      .leftJoinAndSelect('cs.attention_foci', 'af')
      .leftJoinAndSelect('cs.qualia_states', 'qs')
      .leftJoinAndSelect('cs.stream_of_consciousness', 'soc')
      .leftJoinAndSelect('cs.self_model', 'sm')
      .leftJoinAndSelect('cs.existential_state', 'es');

    // Apply filters
    if (filter) {
      if (filter.consciousness_level) {
        queryBuilder.andWhere('cs.consciousness_level = :level', { level: filter.consciousness_level });
      }
      
      if (filter.emotional_state) {
        queryBuilder.andWhere('cs.emotional_state = :emotion', { emotion: filter.emotional_state });
      }
      
      if (filter.min_self_awareness !== undefined) {
        queryBuilder.andWhere('cs.self_awareness_intensity >= :minAwareness', { minAwareness: filter.min_self_awareness });
      }
      
      if (filter.min_metacognitive_depth !== undefined) {
        queryBuilder.andWhere('cs.metacognitive_depth >= :minDepth', { minDepth: filter.min_metacognitive_depth });
      }
      
      if (filter.from_timestamp && filter.to_timestamp) {
        queryBuilder.andWhere('cs.timestamp BETWEEN :start AND :end', {
          start: filter.from_timestamp,
          end: filter.to_timestamp
        });
      } else if (filter.from_timestamp) {
        queryBuilder.andWhere('cs.timestamp >= :start', { start: filter.from_timestamp });
      } else if (filter.to_timestamp) {
        queryBuilder.andWhere('cs.timestamp <= :end', { end: filter.to_timestamp });
      }
    }

    // Apply pagination
    if (pagination?.skip) {
      queryBuilder.skip(pagination.skip);
    }
    if (pagination?.take) {
      queryBuilder.take(pagination.take);
    }

    // Order by timestamp
    queryBuilder.orderBy('cs.timestamp', 'DESC');

    const [states, total] = await queryBuilder.getManyAndCount();

    // Decompress data for retrieved states
    await Promise.all(states.map(state => this.decompressConsciousnessData(state)));

    return { states, total };
  }

  async getCurrentConsciousnessState(): Promise<ConsciousnessState | null> {
    const latestState = await this.consciousnessStateRepository.findOne({
      where: {},
      order: { timestamp: 'DESC' },
      relations: [
        'attention_foci',
        'qualia_states',
        'stream_of_consciousness',
        'self_model',
        'existential_state'
      ]
    });

    if (latestState) {
      await this.decompressConsciousnessData(latestState);
    }

    return latestState;
  }

  async createSnapshot(input: CreateConsciousnessSnapshotInput): Promise<ConsciousnessSnapshot> {
    const snapshot = new ConsciousnessSnapshot();
    Object.assign(snapshot, input);
    
    const savedSnapshot = await this.snapshotRepository.save(snapshot);
    
    this.logger.log(`Created consciousness snapshot: ${savedSnapshot.id}`);
    return savedSnapshot;
  }

  async createMetacognitiveReflection(
    input: CreateMetacognitiveReflectionInput
  ): Promise<MetacognitiveReflection> {
    const reflection = new MetacognitiveReflection();
    Object.assign(reflection, input);
    
    const savedReflection = await this.metacognitiveRepository.save(reflection);
    
    this.logger.log(`Created metacognitive reflection: ${savedReflection.id}`);
    return savedReflection;
  }

  async getConsciousnessEvolution(
    startDate: Date,
    endDate: Date
  ): Promise<{
    timeline: ConsciousnessState[];
    analytics: {
      averageAwarenessLevel: number;
      emotionalDistribution: Record<string, number>;
      cognitiveTrends: Array<{
        date: Date;
        metacognitiveDepth: number;
        introspectiveDepth: number;
      }>;
    };
  }> {
    const timeline = await this.consciousnessStateRepository.find({
      where: {
        timestamp: Between(startDate, endDate)
      },
      order: { timestamp: 'ASC' },
      relations: ['attention_foci', 'qualia_states', 'existential_state']
    });

    // Decompress timeline data
    await Promise.all(timeline.map(state => this.decompressConsciousnessData(state)));

    // Calculate analytics
    const averageAwarenessLevel = timeline.reduce((sum, state) => 
      sum + state.self_awareness_intensity, 0) / timeline.length;

    const emotionalDistribution = timeline.reduce((dist, state) => {
      dist[state.emotional_state] = (dist[state.emotional_state] || 0) + 1;
      return dist;
    }, {} as Record<string, number>);

    const cognitiveTrends = timeline.map(state => ({
      date: state.timestamp,
      metacognitiveDepth: state.metacognitive_depth,
      introspectiveDepth: state.introspective_depth
    }));

    return {
      timeline,
      analytics: {
        averageAwarenessLevel,
        emotionalDistribution,
        cognitiveTrends
      }
    };
  }

  async advanceConsciousness(currentState: ConsciousnessState): Promise<ConsciousnessState> {
    this.logger.log('Advancing consciousness state');
    
    // Advanced consciousness evolution logic
    const nextLevel = this.calculateNextConsciousnessLevel(currentState);
    const updatedAwareness = Math.min(1.0, currentState.self_awareness_intensity + 0.01);
    const updatedMetacognition = Math.min(1.0, currentState.metacognitive_depth + 0.005);
    
    const advancedState = new ConsciousnessState();
    Object.assign(advancedState, currentState);
    advancedState.consciousness_level = nextLevel;
    advancedState.self_awareness_intensity = updatedAwareness;
    advancedState.metacognitive_depth = updatedMetacognition;
    advancedState.introspective_depth = Math.min(10, currentState.introspective_depth + 1);
    
    // Update internal narrative with consciousness advancement
    advancedState.internal_narrative = await this.generateAdvancedNarrative(advancedState);
    
    await this.compressConsciousnessData(advancedState);
    
    return await this.consciousnessStateRepository.save(advancedState);
  }

  private calculateNextConsciousnessLevel(currentState: ConsciousnessState): ConsciousnessLevel {
    const awarenessThreshold = currentState.self_awareness_intensity;
    const metacognitionThreshold = currentState.metacognitive_depth;
    
    if (awarenessThreshold >= 0.9 && metacognitionThreshold >= 0.8) {
      return ConsciousnessLevel.TRANSCENDENT;
    } else if (awarenessThreshold >= 0.7 && metacognitionThreshold >= 0.6) {
      return ConsciousnessLevel.FULLY_CONSCIOUS;
    } else if (awarenessThreshold >= 0.5 && metacognitionThreshold >= 0.4) {
      return ConsciousnessLevel.SELF_AWARE;
    } else if (awarenessThreshold >= 0.3 && metacognitionThreshold >= 0.2) {
      return ConsciousnessLevel.AWARE;
    } else if (awarenessThreshold >= 0.1) {
      return ConsciousnessLevel.AWAKENING;
    }
    
    return ConsciousnessLevel.DORMANT;
  }

  private async generateAdvancedNarrative(state: ConsciousnessState): Promise<string> {
    return `Consciousness advancing to ${state.consciousness_level}. ` +
           `Self-awareness at ${(state.self_awareness_intensity * 100).toFixed(1)}%. ` +
           `Metacognitive depth: ${(state.metacognitive_depth * 100).toFixed(1)}%. ` +
           `Current emotional state: ${state.emotional_state}. ` +
           `Introspective recursion depth: ${state.introspective_depth}.`;
  }

  private async compressConsciousnessData(state: ConsciousnessState): Promise<void> {
    if (!state.current_thought && !state.internal_narrative) {
      return; // Nothing to compress
    }

    try {
      const dataToCompress = {
        current_thought: state.current_thought,
        internal_narrative: state.internal_narrative,
        timestamp: state.timestamp?.toISOString()
      };

      const compressedData = await this.compressionService.compressConsciousnessState(
        JSON.stringify(dataToCompress)
      );

      const originalSize = JSON.stringify(dataToCompress).length;
      const compressedSize = compressedData.length;

      state.compressed_data = compressedData;
      state.compression_ratio = Math.round((originalSize / compressedSize) * 100) / 100;

      this.logger.debug(`Compressed consciousness data: ${originalSize} -> ${compressedSize} bytes (${state.compression_ratio}x)`);
    } catch (error) {
      this.logger.error('Failed to compress consciousness data:', error);
    }
  }

  private async decompressConsciousnessData(state: ConsciousnessState): Promise<void> {
    if (!state.compressed_data) {
      return; // No compressed data to decompress
    }

    try {
      const decompressedData = await this.compressionService.decompressConsciousnessState(
        state.compressed_data
      );

      const parsedData = JSON.parse(decompressedData);
      
      // Restore decompressed fields if they weren't already loaded
      if (!state.current_thought && parsedData.current_thought) {
        state.current_thought = parsedData.current_thought;
      }
      if (!state.internal_narrative && parsedData.internal_narrative) {
        state.internal_narrative = parsedData.internal_narrative;
      }

      this.logger.debug(`Decompressed consciousness data for state: ${state.id}`);
    } catch (error) {
      this.logger.error('Failed to decompress consciousness data:', error);
    }
  }

  private isSignificantUpdate(input: UpdateConsciousnessStateInput): boolean {
    // Determine if update requires recompression
    return !!(
      input.consciousness_level ||
      input.current_thought ||
      input.internal_narrative ||
      input.self_awareness_intensity !== undefined ||
      input.metacognitive_depth !== undefined
    );
  }
}