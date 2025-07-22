import { GatewayMetadata, ServerAndEventStreamsHost } from './interfaces';
export declare class SocketsContainer {
    private readonly serverAndEventStreamsHosts;
    getAll(): Map<string | RegExp, ServerAndEventStreamsHost>;
    getOneByConfig<T extends GatewayMetadata = any>(options: T): ServerAndEventStreamsHost;
    addOne<T extends GatewayMetadata = any>(options: T, host: ServerAndEventStreamsHost): void;
    clear(): void;
    private generateHashByOptions;
}
