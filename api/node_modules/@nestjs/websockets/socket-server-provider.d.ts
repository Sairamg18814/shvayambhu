import { ApplicationConfig } from '@nestjs/core/application-config';
import { GatewayMetadata } from './interfaces/gateway-metadata.interface';
import { ServerAndEventStreamsHost } from './interfaces/server-and-event-streams-host.interface';
import { SocketsContainer } from './sockets-container';
export declare class SocketServerProvider {
    private readonly socketsContainer;
    private readonly applicationConfig;
    constructor(socketsContainer: SocketsContainer, applicationConfig: ApplicationConfig);
    scanForSocketServer<T extends GatewayMetadata = any>(options: T, port: number): ServerAndEventStreamsHost;
    private createSocketServer;
    private decorateWithNamespace;
    private getServerOfNamespace;
    private validateNamespace;
}
