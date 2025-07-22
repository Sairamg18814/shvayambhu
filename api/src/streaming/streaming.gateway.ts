import {
  WebSocketGateway,
  WebSocketServer,
  SubscribeMessage,
  OnGatewayConnection,
  OnGatewayDisconnect,
  OnGatewayInit,
  MessageBody,
  ConnectedSocket,
  WsException,
} from '@nestjs/websockets';
import { Server, Socket } from 'socket.io';
import { Logger, UseGuards } from '@nestjs/common';
import { Observable, Subject } from 'rxjs';
import { map, takeUntil } from 'rxjs/operators';
import { CompressedStreamingService } from './compressed-streaming.service';

export interface StreamingOptions {
  temperature?: number;
  maxTokens?: number;
  topP?: number;
  topK?: number;
  stream?: boolean;
  model?: string;
}

export interface GenerationToken {
  token: string;
  timestamp: number;
  metadata?: {
    entropy?: number;
    probability?: number;
    alternatives?: string[];
  };
}

export interface StreamingSession {
  id: string;
  clientId: string;
  startTime: Date;
  options: StreamingOptions;
  tokensGenerated: number;
  active: boolean;
  consciousnessLevel?: number;
}

@WebSocketGateway({
  cors: {
    origin: process.env.ALLOWED_ORIGINS?.split(',') || ['http://localhost:3000'],
    credentials: true,
  },
  namespace: 'streaming',
  transports: ['websocket', 'polling'],
})
export class StreamingGateway
  implements OnGatewayInit, OnGatewayConnection, OnGatewayDisconnect
{
  @WebSocketServer()
  server: Server;

  private logger = new Logger('StreamingGateway');
  private sessions = new Map<string, StreamingSession>();
  private cancelSubjects = new Map<string, Subject<void>>();

  constructor(private compressedStreamingService: CompressedStreamingService) {}

  afterInit(server: Server) {
    this.logger.log('WebSocket Gateway initialized');
  }

  handleConnection(client: Socket) {
    this.logger.log(`Client connected: ${client.id}`);
    
    // Send initial connection acknowledgment with compression support
    this.compressedStreamingService.sendCompressed(client, 'connected', {
      clientId: client.id,
      timestamp: new Date(),
      capabilities: {
        streaming: true,
        consciousness: true,
        multimodal: false,
        maxTokens: 4096,
        compression: true,
        supportedCompressionTypes: ['gzip', 'deflate', 'brotli', 'consciousness'],
      },
    });
  }

  handleDisconnect(client: Socket) {
    this.logger.log(`Client disconnected: ${client.id}`);
    
    // Clean up any active sessions
    const session = this.sessions.get(client.id);
    if (session) {
      this.cancelGeneration(client);
      this.sessions.delete(client.id);
    }
    
    // Clean up compression stats
    this.compressedStreamingService.cleanupClient(client.id);
  }

  @SubscribeMessage('startGeneration')
  async startGeneration(
    @MessageBody() data: { prompt: string; options?: StreamingOptions },
    @ConnectedSocket() client: Socket,
  ): Promise<void> {
    try {
      // Validate input
      if (!data.prompt || data.prompt.length === 0) {
        throw new WsException('Prompt is required');
      }

      // Create session
      const session: StreamingSession = {
        id: `session_${Date.now()}_${client.id}`,
        clientId: client.id,
        startTime: new Date(),
        options: data.options || {},
        tokensGenerated: 0,
        active: true,
        consciousnessLevel: 0.5, // Default consciousness level
      };

      this.sessions.set(client.id, session);

      // Create cancellation subject
      const cancelSubject = new Subject<void>();
      this.cancelSubjects.set(client.id, cancelSubject);

      // Send session started event with compression
      await this.compressedStreamingService.sendCompressed(client, 'generationStarted', {
        sessionId: session.id,
        timestamp: new Date(),
      });

      // Start streaming generation
      this.streamGeneration(data.prompt, session, client, cancelSubject)
        .subscribe({
          next: async (token) => {
            if (session.active) {
              await this.compressedStreamingService.sendCompressed(client, 'token', token);
              session.tokensGenerated++;
            }
          },
          error: (error) => {
            this.logger.error(`Generation error: ${error.message}`);
            client.emit('error', {
              message: error.message,
              code: error.code || 'GENERATION_ERROR',
            });
            this.cleanupSession(client.id);
          },
          complete: async () => {
            if (session.active) {
              await this.compressedStreamingService.sendCompressed(client, 'generationComplete', {
                sessionId: session.id,
                tokensGenerated: session.tokensGenerated,
                duration: Date.now() - session.startTime.getTime(),
              });
            }
            this.cleanupSession(client.id);
          },
        });
    } catch (error) {
      this.logger.error(`Start generation error: ${error.message}`);
      client.emit('error', {
        message: error.message,
        code: 'START_ERROR',
      });
    }
  }

  @SubscribeMessage('cancelGeneration')
  cancelGeneration(@ConnectedSocket() client: Socket): void {
    const session = this.sessions.get(client.id);
    const cancelSubject = this.cancelSubjects.get(client.id);

    if (session && cancelSubject) {
      session.active = false;
      cancelSubject.next();
      cancelSubject.complete();
      
      client.emit('generationCancelled', {
        sessionId: session.id,
        tokensGenerated: session.tokensGenerated,
      });
      
      this.cleanupSession(client.id);
    }
  }

  @SubscribeMessage('updateConsciousness')
  updateConsciousness(
    @MessageBody() data: { level: number },
    @ConnectedSocket() client: Socket,
  ): void {
    const session = this.sessions.get(client.id);
    
    if (session && data.level >= 0 && data.level <= 1) {
      session.consciousnessLevel = data.level;
      
      this.compressedStreamingService.sendConsciousnessUpdate(client, {
        level: data.level,
        timestamp: new Date(),
      });
    }
  }

  @SubscribeMessage('negotiateCompression')
  negotiateCompression(
    @MessageBody() data: {
      supportedAlgorithms: string[];
      maxChunkSize: number;
      preferredCompression: string;
    },
    @ConnectedSocket() client: Socket,
  ): void {
    this.compressedStreamingService.negotiateCompression(client, data);
  }

  @SubscribeMessage('getCompressionStats')
  getCompressionStats(
    @ConnectedSocket() client: Socket,
  ): void {
    const stats = this.compressedStreamingService.getCompressionStats(client.id);
    this.compressedStreamingService.sendCompressed(client, 'compressionStats', stats);
  }

  private streamGeneration(
    prompt: string,
    session: StreamingSession,
    client: Socket,
    cancelSubject: Subject<void>,
  ): Observable<GenerationToken> {
    // This is a mock implementation - replace with actual model integration
    return new Observable<GenerationToken>((subscriber) => {
      const words = prompt.split(' ');
      const responseWords = [
        'This', 'is', 'a', 'streaming', 'response', 'from', 'Shvayambhu.',
        'The', 'conscious', 'AI', 'is', 'processing', 'your', 'request...',
      ];

      let index = 0;
      const interval = setInterval(() => {
        if (index < responseWords.length) {
          const token: GenerationToken = {
            token: responseWords[index] + (index < responseWords.length - 1 ? ' ' : ''),
            timestamp: Date.now(),
            metadata: {
              entropy: Math.random() * 5,
              probability: 0.8 + Math.random() * 0.2,
              alternatives: [],
            },
          };
          subscriber.next(token);
          index++;
        } else {
          clearInterval(interval);
          subscriber.complete();
        }
      }, 100); // 100ms between tokens

      // Handle cancellation
      cancelSubject.subscribe(() => {
        clearInterval(interval);
        subscriber.complete();
      });

      // Cleanup on unsubscribe
      return () => {
        clearInterval(interval);
      };
    }).pipe(
      takeUntil(cancelSubject),
      map((token) => {
        // Apply consciousness level modifications
        if (session.consciousnessLevel && session.consciousnessLevel > 0.7) {
          // Higher consciousness = more thoughtful responses
          token.metadata.entropy *= 1.2;
        }
        return token;
      }),
    );
  }

  private cleanupSession(clientId: string): void {
    this.sessions.delete(clientId);
    const cancelSubject = this.cancelSubjects.get(clientId);
    if (cancelSubject) {
      cancelSubject.complete();
      this.cancelSubjects.delete(clientId);
    }
  }

  // Broadcast to all connected clients
  broadcastSystemMessage(message: string): void {
    this.server.emit('systemMessage', {
      message,
      timestamp: new Date(),
    });
  }

  // Get active sessions count
  getActiveSessions(): number {
    return Array.from(this.sessions.values()).filter(s => s.active).length;
  }

  // Get session metrics
  getSessionMetrics(): any {
    const sessions = Array.from(this.sessions.values());
    const activeSessions = sessions.filter(s => s.active);
    
    return {
      totalSessions: sessions.length,
      activeSessions: activeSessions.length,
      totalTokensGenerated: sessions.reduce((sum, s) => sum + s.tokensGenerated, 0),
      averageConsciousnessLevel: activeSessions.length > 0
        ? activeSessions.reduce((sum, s) => sum + (s.consciousnessLevel || 0), 0) / activeSessions.length
        : 0,
    };
  }
}
