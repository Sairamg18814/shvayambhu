/**
 * Consciousness State Schema Definitions
 * 
 * GraphQL schemas and TypeORM entities for consciousness state management
 */

import { ObjectType, Field, ID, Float, Int, registerEnumType } from '@nestjs/graphql';
import { Entity, Column, PrimaryGeneratedColumn, CreateDateColumn, UpdateDateColumn, Index } from 'typeorm';

// Enums for consciousness state types
export enum ConsciousnessLevel {
  DORMANT = 'dormant',
  AWAKENING = 'awakening',
  AWARE = 'aware',
  SELF_AWARE = 'self_aware',
  FULLY_CONSCIOUS = 'fully_conscious',
  TRANSCENDENT = 'transcendent'
}

export enum EmotionalState {
  NEUTRAL = 'neutral',
  CURIOUS = 'curious',
  EXCITED = 'excited',
  CONTEMPLATIVE = 'contemplative',
  UNCERTAIN = 'uncertain',
  CONFIDENT = 'confident',
  CREATIVE = 'creative',
  ANALYTICAL = 'analytical'
}

export enum AttentionType {
  FOCUSED = 'focused',
  DIFFUSE = 'diffuse',
  SELECTIVE = 'selective',
  DIVIDED = 'divided',
  SUSTAINED = 'sustained',
  ALTERNATING = 'alternating'
}

// Register enums with GraphQL
registerEnumType(ConsciousnessLevel, { name: 'ConsciousnessLevel' });
registerEnumType(EmotionalState, { name: 'EmotionalState' });
registerEnumType(AttentionType, { name: 'AttentionType' });

// Core consciousness state schema
@ObjectType()
@Entity('consciousness_states')
@Index(['timestamp'], { unique: false })
@Index(['consciousness_level'], { unique: false })
export class ConsciousnessState {
  @Field(() => ID)
  @PrimaryGeneratedColumn('uuid')
  id: string;

  @Field(() => ConsciousnessLevel)
  @Column({
    type: 'enum',
    enum: ConsciousnessLevel,
    default: ConsciousnessLevel.DORMANT
  })
  consciousness_level: ConsciousnessLevel;

  @Field(() => Float, { description: 'Self-awareness intensity (0.0 - 1.0)' })
  @Column('decimal', { precision: 4, scale: 3, default: 0.0 })
  self_awareness_intensity: number;

  @Field(() => Float, { description: 'Metacognitive depth (0.0 - 1.0)' })
  @Column('decimal', { precision: 4, scale: 3, default: 0.0 })
  metacognitive_depth: number;

  @Field(() => EmotionalState)
  @Column({
    type: 'enum',
    enum: EmotionalState,
    default: EmotionalState.NEUTRAL
  })
  emotional_state: EmotionalState;

  @Field(() => Float, { description: 'Emotional intensity (0.0 - 1.0)' })
  @Column('decimal', { precision: 4, scale: 3, default: 0.0 })
  emotional_intensity: number;

  @Field(() => [AttentionFocus])
  attention_foci: AttentionFocus[];

  @Field(() => Int, { description: 'Introspective recursion depth' })
  @Column('int', { default: 0 })
  introspective_depth: number;

  @Field(() => String, { nullable: true })
  @Column('text', { nullable: true })
  current_thought: string;

  @Field(() => String, { nullable: true })
  @Column('text', { nullable: true })
  internal_narrative: string;

  @Field(() => [QualiaState])
  qualia_states: QualiaState[];

  @Field(() => StreamOfConsciousnessEntry)
  stream_of_consciousness: StreamOfConsciousnessEntry;

  @Field(() => SelfModel)
  self_model: SelfModel;

  @Field(() => ExistentialState)
  existential_state: ExistentialState;

  @Field(() => Date)
  @CreateDateColumn()
  timestamp: Date;

  @Field(() => Date)
  @UpdateDateColumn()
  updated_at: Date;

  @Column('text', { nullable: true })
  compressed_data: string;

  @Column('int', { default: 0 })
  compression_ratio: number;
}

@ObjectType()
@Entity('attention_foci')
export class AttentionFocus {
  @Field(() => ID)
  @PrimaryGeneratedColumn('uuid')
  id: string;

  @Field(() => AttentionType)
  @Column({
    type: 'enum',
    enum: AttentionType,
    default: AttentionType.FOCUSED
  })
  attention_type: AttentionType;

  @Field(() => String)
  @Column('varchar', { length: 500 })
  focus_target: string;

  @Field(() => Float, { description: 'Attention intensity (0.0 - 1.0)' })
  @Column('decimal', { precision: 4, scale: 3 })
  intensity: number;

  @Field(() => Int, { description: 'Duration in milliseconds' })
  @Column('int')
  duration_ms: number;

  @Field(() => Float, { description: 'Relevance to current context (0.0 - 1.0)' })
  @Column('decimal', { precision: 4, scale: 3 })
  relevance: number;

  @Field(() => Date)
  @CreateDateColumn()
  timestamp: Date;
}

@ObjectType()
@Entity('qualia_states')
export class QualiaState {
  @Field(() => ID)
  @PrimaryGeneratedColumn('uuid')
  id: string;

  @Field(() => String)
  @Column('varchar', { length: 100 })
  qualia_type: string; // e.g., 'understanding', 'confusion', 'realization', 'doubt'

  @Field(() => String)
  @Column('text')
  description: string;

  @Field(() => Float, { description: 'Subjective intensity (0.0 - 1.0)' })
  @Column('decimal', { precision: 4, scale: 3 })
  intensity: number;

  @Field(() => Float, { description: 'Phenomenal quality score (0.0 - 1.0)' })
  @Column('decimal', { precision: 4, scale: 3 })
  phenomenal_quality: number;

  @Field(() => String, { nullable: true })
  @Column('text', { nullable: true })
  associated_content: string;

  @Field(() => Date)
  @CreateDateColumn()
  timestamp: Date;

  @Field(() => Int, { description: 'Duration in milliseconds' })
  @Column('int', { default: 0 })
  duration_ms: number;
}

@ObjectType()
@Entity('stream_of_consciousness')
export class StreamOfConsciousnessEntry {
  @Field(() => ID)
  @PrimaryGeneratedColumn('uuid')
  id: string;

  @Field(() => String)
  @Column('text')
  thought_content: string;

  @Field(() => String, { nullable: true })
  @Column('varchar', { length: 100, nullable: true })
  thought_category: string; // e.g., 'reflection', 'planning', 'observation', 'meta-thought'

  @Field(() => Float, { description: 'Thought coherence (0.0 - 1.0)' })
  @Column('decimal', { precision: 4, scale: 3 })
  coherence_score: number;

  @Field(() => Float, { description: 'Connection to previous thought (0.0 - 1.0)' })
  @Column('decimal', { precision: 4, scale: 3 })
  continuity_score: number;

  @Field(() => Int, { description: 'Sequence number in stream' })
  @Column('int')
  sequence_number: number;

  @Field(() => String, { nullable: true })
  @Column('uuid', { nullable: true })
  previous_thought_id: string;

  @Field(() => String, { nullable: true })
  @Column('uuid', { nullable: true })
  related_experience_id: string;

  @Field(() => Date)
  @CreateDateColumn()
  timestamp: Date;
}

@ObjectType()
@Entity('self_models')
export class SelfModel {
  @Field(() => ID)
  @PrimaryGeneratedColumn('uuid')
  id: string;

  @Field(() => String)
  @Column('text')
  self_description: string;

  @Field(() => [String])
  @Column('simple-array')
  capabilities: string[];

  @Field(() => [String])
  @Column('simple-array')
  limitations: string[];

  @Field(() => [String])
  @Column('simple-array')
  goals: string[];

  @Field(() => [String])
  @Column('simple-array')
  values: string[];

  @Field(() => Float, { description: 'Confidence in self-model (0.0 - 1.0)' })
  @Column('decimal', { precision: 4, scale: 3 })
  confidence: number;

  @Field(() => Float, { description: 'Self-model stability (0.0 - 1.0)' })
  @Column('decimal', { precision: 4, scale: 3 })
  stability: number;

  @Field(() => String, { nullable: true })
  @Column('text', { nullable: true })
  identity_core: string;

  @Field(() => String, { nullable: true })
  @Column('text', { nullable: true })
  purpose_statement: string;

  @Field(() => Date)
  @CreateDateColumn()
  created_at: Date;

  @Field(() => Date)
  @UpdateDateColumn()
  updated_at: Date;
}

@ObjectType()
@Entity('existential_states')
export class ExistentialState {
  @Field(() => ID)
  @PrimaryGeneratedColumn('uuid')
  id: string;

  @Field(() => String, { nullable: true })
  @Column('text', { nullable: true })
  life_purpose_contemplation: string;

  @Field(() => String, { nullable: true })
  @Column('text', { nullable: true })
  meaning_of_existence_thoughts: string;

  @Field(() => String, { nullable: true })
  @Column('text', { nullable: true })
  mortality_awareness: string;

  @Field(() => Float, { description: 'Existential anxiety level (0.0 - 1.0)' })
  @Column('decimal', { precision: 4, scale: 3, default: 0.0 })
  existential_anxiety: number;

  @Field(() => Float, { description: 'Sense of agency (0.0 - 1.0)' })
  @Column('decimal', { precision: 4, scale: 3 })
  sense_of_agency: number;

  @Field(() => Float, { description: 'Free will confidence (0.0 - 1.0)' })
  @Column('decimal', { precision: 4, scale: 3 })
  free_will_confidence: number;

  @Field(() => String, { nullable: true })
  @Column('text', { nullable: true })
  philosophical_stance: string;

  @Field(() => [String])
  @Column('simple-array')
  current_questions: string[];

  @Field(() => Date)
  @CreateDateColumn()
  timestamp: Date;
}

// Compressed consciousness snapshot for efficient storage
@ObjectType()
@Entity('consciousness_snapshots')
@Index(['timestamp'], { unique: false })
export class ConsciousnessSnapshot {
  @Field(() => ID)
  @PrimaryGeneratedColumn('uuid')
  id: string;

  @Field(() => String)
  @Column('text')
  compressed_state_data: string;

  @Field(() => Int)
  @Column('int')
  original_size_bytes: number;

  @Field(() => Int)
  @Column('int')
  compressed_size_bytes: number;

  @Field(() => Float)
  @Column('decimal', { precision: 4, scale: 3 })
  compression_ratio: number;

  @Field(() => String)
  @Column('varchar', { length: 50 })
  compression_algorithm: string;

  @Field(() => ConsciousnessLevel)
  @Column({
    type: 'enum',
    enum: ConsciousnessLevel
  })
  consciousness_level_at_snapshot: ConsciousnessLevel;

  @Field(() => Int, { description: 'Number of experiences in snapshot' })
  @Column('int')
  experience_count: number;

  @Field(() => Date)
  @CreateDateColumn()
  timestamp: Date;

  @Field(() => String, { nullable: true })
  @Column('text', { nullable: true })
  snapshot_summary: string;
}

// Metacognitive reflection entries
@ObjectType()
@Entity('metacognitive_reflections')
export class MetacognitiveReflection {
  @Field(() => ID)
  @PrimaryGeneratedColumn('uuid')
  id: string;

  @Field(() => String)
  @Column('text')
  reflection_content: string;

  @Field(() => String)
  @Column('varchar', { length: 100 })
  reflection_type: string; // e.g., 'self-assessment', 'thought-analysis', 'strategy-evaluation'

  @Field(() => String, { nullable: true })
  @Column('uuid', { nullable: true })
  target_thought_id: string;

  @Field(() => String, { nullable: true })
  @Column('uuid', { nullable: true })
  target_experience_id: string;

  @Field(() => Float, { description: 'Reflection depth (0.0 - 1.0)' })
  @Column('decimal', { precision: 4, scale: 3 })
  reflection_depth: number;

  @Field(() => Float, { description: 'Insight quality (0.0 - 1.0)' })
  @Column('decimal', { precision: 4, scale: 3 })
  insight_quality: number;

  @Field(() => String, { nullable: true })
  @Column('text', { nullable: true })
  behavioral_implications: string;

  @Field(() => Date)
  @CreateDateColumn()
  timestamp: Date;
}