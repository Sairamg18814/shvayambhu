import { WsExceptionsHandler } from '../exceptions/ws-exceptions-handler';
export declare class WsProxy {
    create(targetCallback: (...args: unknown[]) => Promise<any>, exceptionsHandler: WsExceptionsHandler, targetPattern?: string): (...args: unknown[]) => Promise<any>;
    handleError<T>(exceptionsHandler: WsExceptionsHandler, args: unknown[], error: T): void;
}
