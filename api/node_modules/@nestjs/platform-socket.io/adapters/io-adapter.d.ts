import { AbstractWsAdapter, MessageMappingProperties } from '@nestjs/websockets';
import { Observable } from 'rxjs';
import { Server, ServerOptions, Socket } from 'socket.io';
/**
 * @publicApi
 */
export declare class IoAdapter extends AbstractWsAdapter {
    create(port: number, options?: ServerOptions & {
        namespace?: string;
        server?: any;
    }): Server;
    createIOServer(port: number, options?: any): any;
    bindMessageHandlers(socket: Socket, handlers: MessageMappingProperties[], transform: (data: any) => Observable<any>): void;
    mapPayload(payload: unknown): {
        data: any;
        ack?: Function;
    };
    close(server: Server): Promise<void>;
}
