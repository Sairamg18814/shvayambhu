import { Module, MiddlewareConsumer } from '@nestjs/common';
import { ConfigModule, ConfigService } from '@nestjs/config';
import { GraphQLModule } from '@nestjs/graphql';
import { ApolloDriver, ApolloDriverConfig } from '@nestjs/apollo';
import { TypeOrmModule } from '@nestjs/typeorm';
import { join } from 'path';
import { AppController } from './app.controller';
import { AppService } from './app.service';
import { MemoryModule } from './memory/memory.module';
import { StreamingModule } from './streaming/streaming.module';
import { ConsciousnessModule } from './consciousness/consciousness.module';
import { CompressionService } from './common/compression/compression.service';
import { GraphQLCompressionMiddleware, ConsciousnessCompressionMiddleware } from './common/middleware/compression.middleware';

@Module({
  imports: [
    ConfigModule.forRoot({
      isGlobal: true,
      envFilePath: '../.env',
    }),
    GraphQLModule.forRootAsync<ApolloDriverConfig>({
      driver: ApolloDriver,
      imports: [ConfigModule],
      useFactory: async (configService: ConfigService) => ({
        autoSchemaFile: join(process.cwd(), 'src/schema.gql'),
        sortSchema: true,
        playground: true,
        introspection: true,
        persistedQueries: {
          ttl: 900, // 15 minutes
        },
        context: ({ req, res }) => ({ req, res }),
        formatResponse: (response, context) => {
          // Add compression headers
          if (context.response) {
            context.response.setHeader('X-Content-Encoding', 'gzip');
          }
          return response;
        },
      }),
      inject: [ConfigService],
    }),
    TypeOrmModule.forRootAsync({
      imports: [ConfigModule],
      useFactory: async (configService: ConfigService) => ({
        type: 'sqlite',
        database: join(process.cwd(), '../data/shvayambhu.db'),
        entities: [__dirname + '/**/*.entity{.ts,.js}'],
        synchronize: configService.get('NODE_ENV') === 'development',
        logging: configService.get('NODE_ENV') === 'development',
      }),
      inject: [ConfigService],
    }),
    MemoryModule,
    StreamingModule,
    ConsciousnessModule,
  ],
  controllers: [AppController],
  providers: [
    AppService,
    CompressionService,
    GraphQLCompressionMiddleware,
    ConsciousnessCompressionMiddleware,
  ],
})
export class AppModule {
  configure(consumer: MiddlewareConsumer) {
    consumer
      .apply(GraphQLCompressionMiddleware)
      .forRoutes('*');
    
    consumer
      .apply(ConsciousnessCompressionMiddleware)
      .forRoutes('*');
  }
}
