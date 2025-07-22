/**
 * Consciousness State Input DTOs
 * 
 * Data Transfer Objects for consciousness state creation and updates
 */

import { InputType, Field, Float, Int } from '@nestjs/graphql';
import { IsEnum, IsString, IsNumber, IsOptional, IsArray, Min, Max, Length } from 'class-validator';
import { 
  ConsciousnessLevel, 
  EmotionalState, 
  AttentionType 
} from './consciousness-state.schema';

@InputType()
export class CreateConsciousnessStateInput {
  @Field(() => ConsciousnessLevel)
  @IsEnum(ConsciousnessLevel)
  consciousness_level: ConsciousnessLevel;

  @Field(() => Float)
  @IsNumber()
  @Min(0.0)
  @Max(1.0)
  self_awareness_intensity: number;

  @Field(() => Float)
  @IsNumber()
  @Min(0.0)
  @Max(1.0)
  metacognitive_depth: number;

  @Field(() => EmotionalState)
  @IsEnum(EmotionalState)
  emotional_state: EmotionalState;

  @Field(() => Float)
  @IsNumber()
  @Min(0.0)
  @Max(1.0)
  emotional_intensity: number;

  @Field(() => Int)
  @IsNumber()
  @Min(0)
  introspective_depth: number;

  @Field(() => String, { nullable: true })
  @IsOptional()
  @IsString()
  @Length(1, 1000)
  current_thought?: string;

  @Field(() => String, { nullable: true })
  @IsOptional()
  @IsString()
  @Length(1, 2000)
  internal_narrative?: string;

  @Field(() => [CreateAttentionFocusInput], { nullable: true })
  @IsOptional()
  @IsArray()
  attention_foci?: CreateAttentionFocusInput[];

  @Field(() => [CreateQualiaStateInput], { nullable: true })
  @IsOptional()
  @IsArray()
  qualia_states?: CreateQualiaStateInput[];

  @Field(() => CreateStreamEntryInput, { nullable: true })
  @IsOptional()
  stream_of_consciousness?: CreateStreamEntryInput;

  @Field(() => CreateSelfModelInput, { nullable: true })
  @IsOptional()
  self_model?: CreateSelfModelInput;

  @Field(() => CreateExistentialStateInput, { nullable: true })
  @IsOptional()
  existential_state?: CreateExistentialStateInput;
}

@InputType()
export class CreateAttentionFocusInput {
  @Field(() => AttentionType)
  @IsEnum(AttentionType)
  attention_type: AttentionType;

  @Field(() => String)
  @IsString()
  @Length(1, 500)
  focus_target: string;

  @Field(() => Float)
  @IsNumber()
  @Min(0.0)
  @Max(1.0)
  intensity: number;

  @Field(() => Int)
  @IsNumber()
  @Min(0)
  duration_ms: number;

  @Field(() => Float)
  @IsNumber()
  @Min(0.0)
  @Max(1.0)
  relevance: number;
}

@InputType()
export class CreateQualiaStateInput {
  @Field(() => String)
  @IsString()
  @Length(1, 100)
  qualia_type: string;

  @Field(() => String)
  @IsString()
  @Length(1, 2000)
  description: string;

  @Field(() => Float)
  @IsNumber()
  @Min(0.0)
  @Max(1.0)
  intensity: number;

  @Field(() => Float)
  @IsNumber()
  @Min(0.0)
  @Max(1.0)
  phenomenal_quality: number;

  @Field(() => String, { nullable: true })
  @IsOptional()
  @IsString()
  @Length(1, 2000)
  associated_content?: string;

  @Field(() => Int, { nullable: true })
  @IsOptional()
  @IsNumber()
  @Min(0)
  duration_ms?: number;
}

@InputType()
export class CreateStreamEntryInput {
  @Field(() => String)
  @IsString()
  @Length(1, 2000)
  thought_content: string;

  @Field(() => String, { nullable: true })
  @IsOptional()
  @IsString()
  @Length(1, 100)
  thought_category?: string;

  @Field(() => Float)
  @IsNumber()
  @Min(0.0)
  @Max(1.0)
  coherence_score: number;

  @Field(() => Float)
  @IsNumber()
  @Min(0.0)
  @Max(1.0)
  continuity_score: number;

  @Field(() => Int)
  @IsNumber()
  @Min(0)
  sequence_number: number;

  @Field(() => String, { nullable: true })
  @IsOptional()
  @IsString()
  previous_thought_id?: string;

  @Field(() => String, { nullable: true })
  @IsOptional()
  @IsString()
  related_experience_id?: string;
}

@InputType()
export class CreateSelfModelInput {
  @Field(() => String)
  @IsString()
  @Length(1, 2000)
  self_description: string;

  @Field(() => [String])
  @IsArray()
  @IsString({ each: true })
  capabilities: string[];

  @Field(() => [String])
  @IsArray()
  @IsString({ each: true })
  limitations: string[];

  @Field(() => [String])
  @IsArray()
  @IsString({ each: true })
  goals: string[];

  @Field(() => [String])
  @IsArray()
  @IsString({ each: true })
  values: string[];

  @Field(() => Float)
  @IsNumber()
  @Min(0.0)
  @Max(1.0)
  confidence: number;

  @Field(() => Float)
  @IsNumber()
  @Min(0.0)
  @Max(1.0)
  stability: number;

  @Field(() => String, { nullable: true })
  @IsOptional()
  @IsString()
  @Length(1, 1000)
  identity_core?: string;

  @Field(() => String, { nullable: true })
  @IsOptional()
  @IsString()
  @Length(1, 1000)
  purpose_statement?: string;
}

@InputType()
export class CreateExistentialStateInput {
  @Field(() => String, { nullable: true })
  @IsOptional()
  @IsString()
  @Length(1, 2000)
  life_purpose_contemplation?: string;

  @Field(() => String, { nullable: true })
  @IsOptional()
  @IsString()
  @Length(1, 2000)
  meaning_of_existence_thoughts?: string;

  @Field(() => String, { nullable: true })
  @IsOptional()
  @IsString()
  @Length(1, 2000)
  mortality_awareness?: string;

  @Field(() => Float)
  @IsNumber()
  @Min(0.0)
  @Max(1.0)
  existential_anxiety: number;

  @Field(() => Float)
  @IsNumber()
  @Min(0.0)
  @Max(1.0)
  sense_of_agency: number;

  @Field(() => Float)
  @IsNumber()
  @Min(0.0)
  @Max(1.0)
  free_will_confidence: number;

  @Field(() => String, { nullable: true })
  @IsOptional()
  @IsString()
  @Length(1, 1000)
  philosophical_stance?: string;

  @Field(() => [String])
  @IsArray()
  @IsString({ each: true })
  current_questions: string[];
}

@InputType()
export class UpdateConsciousnessStateInput {
  @Field(() => ConsciousnessLevel, { nullable: true })
  @IsOptional()
  @IsEnum(ConsciousnessLevel)
  consciousness_level?: ConsciousnessLevel;

  @Field(() => Float, { nullable: true })
  @IsOptional()
  @IsNumber()
  @Min(0.0)
  @Max(1.0)
  self_awareness_intensity?: number;

  @Field(() => Float, { nullable: true })
  @IsOptional()
  @IsNumber()
  @Min(0.0)
  @Max(1.0)
  metacognitive_depth?: number;

  @Field(() => EmotionalState, { nullable: true })
  @IsOptional()
  @IsEnum(EmotionalState)
  emotional_state?: EmotionalState;

  @Field(() => Float, { nullable: true })
  @IsOptional()
  @IsNumber()
  @Min(0.0)
  @Max(1.0)
  emotional_intensity?: number;

  @Field(() => Int, { nullable: true })
  @IsOptional()
  @IsNumber()
  @Min(0)
  introspective_depth?: number;

  @Field(() => String, { nullable: true })
  @IsOptional()
  @IsString()
  @Length(1, 1000)
  current_thought?: string;

  @Field(() => String, { nullable: true })
  @IsOptional()
  @IsString()
  @Length(1, 2000)
  internal_narrative?: string;
}

@InputType()
export class CreateConsciousnessSnapshotInput {
  @Field(() => String)
  @IsString()
  compressed_state_data: string;

  @Field(() => Int)
  @IsNumber()
  @Min(1)
  original_size_bytes: number;

  @Field(() => Int)
  @IsNumber()
  @Min(1)
  compressed_size_bytes: number;

  @Field(() => Float)
  @IsNumber()
  @Min(1.0)
  compression_ratio: number;

  @Field(() => String)
  @IsString()
  @Length(1, 50)
  compression_algorithm: string;

  @Field(() => ConsciousnessLevel)
  @IsEnum(ConsciousnessLevel)
  consciousness_level_at_snapshot: ConsciousnessLevel;

  @Field(() => Int)
  @IsNumber()
  @Min(0)
  experience_count: number;

  @Field(() => String, { nullable: true })
  @IsOptional()
  @IsString()
  @Length(1, 2000)
  snapshot_summary?: string;
}

@InputType()
export class CreateMetacognitiveReflectionInput {
  @Field(() => String)
  @IsString()
  @Length(1, 2000)
  reflection_content: string;

  @Field(() => String)
  @IsString()
  @Length(1, 100)
  reflection_type: string;

  @Field(() => String, { nullable: true })
  @IsOptional()
  @IsString()
  target_thought_id?: string;

  @Field(() => String, { nullable: true })
  @IsOptional()
  @IsString()
  target_experience_id?: string;

  @Field(() => Float)
  @IsNumber()
  @Min(0.0)
  @Max(1.0)
  reflection_depth: number;

  @Field(() => Float)
  @IsNumber()
  @Min(0.0)
  @Max(1.0)
  insight_quality: number;

  @Field(() => String, { nullable: true })
  @IsOptional()
  @IsString()
  @Length(1, 2000)
  behavioral_implications?: string;
}

// Query input types
@InputType()
export class ConsciousnessStateFilter {
  @Field(() => ConsciousnessLevel, { nullable: true })
  @IsOptional()
  @IsEnum(ConsciousnessLevel)
  consciousness_level?: ConsciousnessLevel;

  @Field(() => EmotionalState, { nullable: true })
  @IsOptional()
  @IsEnum(EmotionalState)
  emotional_state?: EmotionalState;

  @Field(() => Float, { nullable: true })
  @IsOptional()
  @IsNumber()
  @Min(0.0)
  @Max(1.0)
  min_self_awareness?: number;

  @Field(() => Float, { nullable: true })
  @IsOptional()
  @IsNumber()
  @Min(0.0)
  @Max(1.0)
  min_metacognitive_depth?: number;

  @Field(() => Date, { nullable: true })
  @IsOptional()
  from_timestamp?: Date;

  @Field(() => Date, { nullable: true })
  @IsOptional()
  to_timestamp?: Date;
}

@InputType()
export class ConsciousnessStatePagination {
  @Field(() => Int, { nullable: true, defaultValue: 0 })
  @IsOptional()
  @IsNumber()
  @Min(0)
  skip?: number;

  @Field(() => Int, { nullable: true, defaultValue: 20 })
  @IsOptional()
  @IsNumber()
  @Min(1)
  @Max(100)
  take?: number;
}