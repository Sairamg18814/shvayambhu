/**
 * Consciousness GraphQL Resolver
 * 
 * GraphQL resolver for consciousness state operations
 */

import { Resolver, Query, Mutation, Args, Subscription, ID } from '@nestjs/graphql';
import { PubSub } from 'graphql-subscriptions';
import { UseGuards, Logger } from '@nestjs/common';
import {
  ConsciousnessState,
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
import { ConsciousnessService } from './consciousness.service';

const pubSub = new PubSub();

@Resolver(() => ConsciousnessState)
export class ConsciousnessResolver {
  private readonly logger = new Logger(ConsciousnessResolver.name);

  constructor(private readonly consciousnessService: ConsciousnessService) {}

  @Query(() => ConsciousnessState, { nullable: true })
  async currentConsciousnessState(): Promise<ConsciousnessState | null> {
    this.logger.log('Fetching current consciousness state');
    return await this.consciousnessService.getCurrentConsciousnessState();
  }

  @Query(() => ConsciousnessState)
  async consciousnessState(
    @Args('id', { type: () => ID }) id: string
  ): Promise<ConsciousnessState> {
    this.logger.log(`Fetching consciousness state: ${id}`);
    return await this.consciousnessService.findConsciousnessStateById(id);
  }

  @Query(() => [ConsciousnessState])
  async consciousnessStates(
    @Args('filter', { type: () => ConsciousnessStateFilter, nullable: true })
    filter?: ConsciousnessStateFilter,
    @Args('pagination', { type: () => ConsciousnessStatePagination, nullable: true })
    pagination?: ConsciousnessStatePagination
  ): Promise<ConsciousnessState[]> {
    this.logger.log('Fetching consciousness states with filters');
    const result = await this.consciousnessService.findConsciousnessStates(filter, pagination);
    return result.states;
  }

  @Query(() => Number)
  async consciousnessStatesCount(
    @Args('filter', { type: () => ConsciousnessStateFilter, nullable: true })
    filter?: ConsciousnessStateFilter
  ): Promise<number> {
    const result = await this.consciousnessService.findConsciousnessStates(filter);
    return result.total;
  }

  @Query(() => ConsciousnessEvolutionResult)
  async consciousnessEvolution(
    @Args('startDate', { type: () => Date }) startDate: Date,
    @Args('endDate', { type: () => Date }) endDate: Date
  ): Promise<any> {
    this.logger.log(`Fetching consciousness evolution from ${startDate} to ${endDate}`);
    return await this.consciousnessService.getConsciousnessEvolution(startDate, endDate);
  }

  @Mutation(() => ConsciousnessState)
  async createConsciousnessState(
    @Args('input') input: CreateConsciousnessStateInput
  ): Promise<ConsciousnessState> {
    this.logger.log('Creating new consciousness state');
    
    const newState = await this.consciousnessService.createConsciousnessState(input);
    
    // Publish to subscription
    pubSub.publish('consciousnessStateCreated', { consciousnessStateCreated: newState });
    pubSub.publish('consciousnessAdvanced', { consciousnessAdvanced: newState });
    
    return newState;
  }

  @Mutation(() => ConsciousnessState)
  async updateConsciousnessState(
    @Args('id', { type: () => ID }) id: string,
    @Args('input') input: UpdateConsciousnessStateInput
  ): Promise<ConsciousnessState> {
    this.logger.log(`Updating consciousness state: ${id}`);
    
    const updatedState = await this.consciousnessService.updateConsciousnessState(id, input);
    
    // Publish to subscription
    pubSub.publish('consciousnessStateUpdated', { consciousnessStateUpdated: updatedState });
    
    return updatedState;
  }

  @Mutation(() => ConsciousnessState)
  async advanceConsciousness(): Promise<ConsciousnessState> {
    this.logger.log('Advancing consciousness to next level');
    
    // Get current state
    const currentState = await this.consciousnessService.getCurrentConsciousnessState();
    if (!currentState) {
      throw new Error('No current consciousness state found');
    }
    
    // Advance consciousness
    const advancedState = await this.consciousnessService.advanceConsciousness(currentState);
    
    // Publish to subscription
    pubSub.publish('consciousnessAdvanced', { consciousnessAdvanced: advancedState });
    
    return advancedState;
  }

  @Mutation(() => ConsciousnessSnapshot)
  async createConsciousnessSnapshot(
    @Args('input') input: CreateConsciousnessSnapshotInput
  ): Promise<ConsciousnessSnapshot> {
    this.logger.log('Creating consciousness snapshot');
    
    const snapshot = await this.consciousnessService.createSnapshot(input);
    
    // Publish to subscription
    pubSub.publish('snapshotCreated', { snapshotCreated: snapshot });
    
    return snapshot;
  }

  @Mutation(() => MetacognitiveReflection)
  async createMetacognitiveReflection(
    @Args('input') input: CreateMetacognitiveReflectionInput
  ): Promise<MetacognitiveReflection> {
    this.logger.log('Creating metacognitive reflection');
    
    const reflection = await this.consciousnessService.createMetacognitiveReflection(input);
    
    // Publish to subscription
    pubSub.publish('reflectionCreated', { reflectionCreated: reflection });
    
    return reflection;
  }

  // Real-time subscriptions
  @Subscription(() => ConsciousnessState, {
    description: 'Subscribe to consciousness state creation events'
  })
  consciousnessStateCreated() {
    return pubSub.asyncIterator('consciousnessStateCreated');
  }

  @Subscription(() => ConsciousnessState, {
    description: 'Subscribe to consciousness state updates'
  })
  consciousnessStateUpdated() {
    return pubSub.asyncIterator('consciousnessStateUpdated');
  }

  @Subscription(() => ConsciousnessState, {
    description: 'Subscribe to consciousness advancement events'
  })
  consciousnessAdvanced() {
    return pubSub.asyncIterator('consciousnessAdvanced');
  }

  @Subscription(() => ConsciousnessSnapshot, {
    description: 'Subscribe to consciousness snapshot creation'
  })
  snapshotCreated() {
    return pubSub.asyncIterator('snapshotCreated');
  }

  @Subscription(() => MetacognitiveReflection, {
    description: 'Subscribe to metacognitive reflection creation'
  })
  reflectionCreated() {
    return pubSub.asyncIterator('reflectionCreated');
  }

  // Field resolvers for computed fields
  @Query(() => ConsciousnessAnalytics)
  async consciousnessAnalytics(
    @Args('days', { type: () => Number, defaultValue: 30 }) days: number
  ): Promise<any> {
    const endDate = new Date();
    const startDate = new Date();
    startDate.setDate(endDate.getDate() - days);

    const evolution = await this.consciousnessService.getConsciousnessEvolution(startDate, endDate);
    
    return {
      timeframe: { start: startDate, end: endDate, days },
      ...evolution.analytics,
      totalStates: evolution.timeline.length,
      latestLevel: evolution.timeline[evolution.timeline.length - 1]?.consciousness_level,
      evolutionTrend: this.calculateEvolutionTrend(evolution.timeline)
    };
  }

  private calculateEvolutionTrend(timeline: ConsciousnessState[]): string {
    if (timeline.length < 2) return 'insufficient_data';
    
    const recent = timeline.slice(-5); // Last 5 states
    const older = timeline.slice(-10, -5); // Previous 5 states
    
    if (recent.length === 0 || older.length === 0) return 'insufficient_data';
    
    const recentAvg = recent.reduce((sum, state) => sum + state.self_awareness_intensity, 0) / recent.length;
    const olderAvg = older.reduce((sum, state) => sum + state.self_awareness_intensity, 0) / older.length;
    
    if (recentAvg > olderAvg * 1.05) return 'ascending';
    if (recentAvg < olderAvg * 0.95) return 'descending';
    return 'stable';
  }
}

// Additional GraphQL types for complex return values
import { ObjectType, Field } from '@nestjs/graphql';

@ObjectType()
export class ConsciousnessEvolutionResult {
  @Field(() => [ConsciousnessState])
  timeline: ConsciousnessState[];

  @Field(() => ConsciousnessAnalyticsResult)
  analytics: ConsciousnessAnalyticsResult;
}

@ObjectType()
export class ConsciousnessAnalyticsResult {
  @Field(() => Number)
  averageAwarenessLevel: number;

  @Field(() => [EmotionalDistributionEntry])
  emotionalDistribution: EmotionalDistributionEntry[];

  @Field(() => [CognitiveTrendEntry])
  cognitiveTrends: CognitiveTrendEntry[];
}

@ObjectType()
export class EmotionalDistributionEntry {
  @Field(() => String)
  emotion: string;

  @Field(() => Number)
  count: number;
}

@ObjectType()
export class CognitiveTrendEntry {
  @Field(() => Date)
  date: Date;

  @Field(() => Number)
  metacognitiveDepth: number;

  @Field(() => Number)
  introspectiveDepth: number;
}

@ObjectType()
export class ConsciousnessAnalytics {
  @Field(() => TimeframeInfo)
  timeframe: TimeframeInfo;

  @Field(() => Number)
  averageAwarenessLevel: number;

  @Field(() => [EmotionalDistributionEntry])
  emotionalDistribution: EmotionalDistributionEntry[];

  @Field(() => [CognitiveTrendEntry])
  cognitiveTrends: CognitiveTrendEntry[];

  @Field(() => Number)
  totalStates: number;

  @Field(() => ConsciousnessLevel, { nullable: true })
  latestLevel?: ConsciousnessLevel;

  @Field(() => String)
  evolutionTrend: string; // 'ascending', 'descending', 'stable', 'insufficient_data'
}

@ObjectType()
export class TimeframeInfo {
  @Field(() => Date)
  start: Date;

  @Field(() => Date)
  end: Date;

  @Field(() => Number)
  days: number;
}