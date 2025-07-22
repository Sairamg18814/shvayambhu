import { Injectable, Logger } from '@nestjs/common';
import { Observable, Subject, interval, of } from 'rxjs';
import { map, takeUntil, bufferTime, filter, mergeMap, catchError } from 'rxjs/operators';

export interface Token {
  id: string;
  content: string;
  timestamp: number;
  metadata?: any;
}

export interface StreamConfig {
  bufferSize: number;
  bufferTimeMs: number;
  maxConcurrentStreams: number;
  tokenDelayMs: number;
}

@Injectable()
export class StreamingService {
  private logger = new Logger('StreamingService');
  private activeStreams = new Map<string, Subject<void>>();
  
  private config: StreamConfig = {
    bufferSize: 10,
    bufferTimeMs: 100,
    maxConcurrentStreams: 100,
    tokenDelayMs: 50,
  };

  /**
   * Create a token stream for text generation
   */
  createTokenStream(
    sessionId: string,
    generator: AsyncGenerator<string, void, unknown>,
  ): Observable<Token[]> {
    // Create cancellation subject
    const cancelSubject = new Subject<void>();
    this.activeStreams.set(sessionId, cancelSubject);

    return new Observable<Token>((subscriber) => {
      let tokenId = 0;
      
      const processGenerator = async () => {
        try {
          for await (const token of generator) {
            if (cancelSubject.closed) break;
            
            subscriber.next({
              id: `${sessionId}_${tokenId++}`,
              content: token,
              timestamp: Date.now(),
            });
            
            // Add artificial delay for demonstration
            await new Promise(resolve => setTimeout(resolve, this.config.tokenDelayMs));
          }
          subscriber.complete();
        } catch (error) {
          subscriber.error(error);
        } finally {
          this.cleanupStream(sessionId);
        }
      };

      processGenerator();

      // Cleanup on unsubscribe
      return () => {
        cancelSubject.next();
        cancelSubject.complete();
        this.cleanupStream(sessionId);
      };
    }).pipe(
      takeUntil(cancelSubject),
      bufferTime(this.config.bufferTimeMs, null, this.config.bufferSize),
      filter(tokens => tokens.length > 0),
      catchError(error => {
        this.logger.error(`Stream error: ${error.message}`);
        return of([]);
      }),
    );
  }

  /**
   * Create a stream with backpressure handling
   */
  createBackpressureStream<T>(
    source: Observable<T>,
    processItem: (item: T) => Observable<any>,
    concurrency: number = 5,
  ): Observable<any> {
    return source.pipe(
      mergeMap(item => processItem(item), concurrency),
      catchError(error => {
        this.logger.error(`Backpressure stream error: ${error.message}`);
        return of(null);
      }),
    );
  }

  /**
   * Cancel an active stream
   */
  cancelStream(sessionId: string): boolean {
    const cancelSubject = this.activeStreams.get(sessionId);
    if (cancelSubject && !cancelSubject.closed) {
      cancelSubject.next();
      cancelSubject.complete();
      this.activeStreams.delete(sessionId);
      return true;
    }
    return false;
  }

  /**
   * Get active stream count
   */
  getActiveStreamCount(): number {
    return Array.from(this.activeStreams.values())
      .filter(subject => !subject.closed)
      .length;
  }

  /**
   * Check if we can accept new streams
   */
  canAcceptNewStream(): boolean {
    return this.getActiveStreamCount() < this.config.maxConcurrentStreams;
  }

  /**
   * Update streaming configuration
   */
  updateConfig(config: Partial<StreamConfig>): void {
    this.config = { ...this.config, ...config };
    this.logger.log(`Streaming config updated: ${JSON.stringify(this.config)}`);
  }

  private cleanupStream(sessionId: string): void {
    const cancelSubject = this.activeStreams.get(sessionId);
    if (cancelSubject && !cancelSubject.closed) {
      cancelSubject.complete();
    }
    this.activeStreams.delete(sessionId);
  }

  /**
   * Create a mock text generator for testing
   */
  async *mockTextGenerator(prompt: string): AsyncGenerator<string, void, unknown> {
    const response = `Thank you for your prompt: "${prompt}". This is a streaming response from Shvayambhu, the conscious AI system. `;
    const tokens = response.split(' ');
    
    for (const token of tokens) {
      yield token + ' ';
      await new Promise(resolve => setTimeout(resolve, 50));
    }
  }
}
