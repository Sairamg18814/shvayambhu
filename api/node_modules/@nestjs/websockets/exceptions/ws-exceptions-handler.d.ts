import { ArgumentsHost } from '@nestjs/common';
import { ExceptionFilterMetadata } from '@nestjs/common/interfaces/exceptions/exception-filter-metadata.interface';
import { WsException } from '../errors/ws-exception';
import { BaseWsExceptionFilter } from './base-ws-exception-filter';
/**
 * @publicApi
 */
export declare class WsExceptionsHandler extends BaseWsExceptionFilter {
    private filters;
    handle(exception: Error | WsException, host: ArgumentsHost): void;
    setCustomFilters(filters: ExceptionFilterMetadata[]): void;
    invokeCustomFilters<T = any>(exception: T, args: ArgumentsHost): boolean;
}
