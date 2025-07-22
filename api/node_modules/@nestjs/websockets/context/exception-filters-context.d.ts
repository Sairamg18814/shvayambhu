import { BaseExceptionFilterContext } from '@nestjs/core/exceptions/base-exception-filter-context';
import { NestContainer } from '@nestjs/core/injector/container';
import { WsExceptionsHandler } from '../exceptions/ws-exceptions-handler';
/**
 * @publicApi
 */
export declare class ExceptionFiltersContext extends BaseExceptionFilterContext {
    constructor(container: NestContainer);
    create(instance: object, callback: <TClient>(client: TClient, data: any) => any, moduleKey: string): WsExceptionsHandler;
    getGlobalMetadata<T extends any[]>(): T;
}
