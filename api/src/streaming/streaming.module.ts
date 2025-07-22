import { Module } from '@nestjs/common';
import { StreamingGateway } from './streaming.gateway';
import { StreamingService } from './streaming.service';
import { BackpressureManager } from './backpressure.manager';
import { ConnectionManager } from './connection.manager';
import { CompressedStreamingService } from './compressed-streaming.service';
import { CompressionService } from '../common/compression/compression.service';

@Module({
  providers: [
    StreamingGateway,
    StreamingService,
    BackpressureManager,
    ConnectionManager,
    CompressedStreamingService,
    CompressionService,
  ],
  exports: [StreamingGateway, StreamingService, CompressedStreamingService],
})
export class StreamingModule {}
