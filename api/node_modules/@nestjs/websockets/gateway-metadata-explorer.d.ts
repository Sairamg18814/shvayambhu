import { MetadataScanner } from '@nestjs/core/metadata-scanner';
import { Observable } from 'rxjs';
import { NestGateway } from './interfaces/nest-gateway.interface';
export interface MessageMappingProperties {
    message: any;
    methodName: string;
    callback: (...args: any[]) => Observable<any> | Promise<any>;
}
export declare class GatewayMetadataExplorer {
    private readonly metadataScanner;
    constructor(metadataScanner: MetadataScanner);
    explore(instance: NestGateway): MessageMappingProperties[];
    exploreMethodMetadata(instancePrototype: object, methodName: string): MessageMappingProperties | null;
    scanForServerHooks(instance: NestGateway): IterableIterator<string>;
}
