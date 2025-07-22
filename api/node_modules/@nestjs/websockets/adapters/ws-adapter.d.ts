import { INestApplicationContext, WebSocketAdapter } from '@nestjs/common';
import { WsMessageHandler } from '@nestjs/common/interfaces';
import { Observable } from 'rxjs';
export interface BaseWsInstance {
    on: (event: string, callback: Function) => void;
    close: Function;
}
export declare abstract class AbstractWsAdapter<TServer extends BaseWsInstance = any, TClient extends BaseWsInstance = any, TOptions = any> implements WebSocketAdapter<TServer, TClient, TOptions> {
    protected readonly httpServer: any;
    private _forceCloseConnections;
    set forceCloseConnections(value: boolean);
    get forceCloseConnections(): boolean;
    constructor(appOrHttpServer?: INestApplicationContext | object);
    bindClientConnect(server: TServer, callback: Function): void;
    bindClientDisconnect(client: TClient, callback: Function): void;
    close(server: TServer): Promise<void>;
    dispose(): Promise<void>;
    abstract create(port: number, options?: TOptions): TServer;
    abstract bindMessageHandlers(client: TClient, handlers: WsMessageHandler[], transform: (data: any) => Observable<any>): any;
}
