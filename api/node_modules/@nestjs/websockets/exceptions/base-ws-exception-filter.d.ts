import { ArgumentsHost, Logger, WsExceptionFilter } from '@nestjs/common';
export interface ErrorPayload<Cause = {
    pattern: string;
    data: unknown;
}> {
    /**
     * Error message identifier.
     */
    status: 'error';
    /**
     * Error message.
     */
    message: string;
    /**
     * Message that caused the exception.
     */
    cause?: Cause;
}
interface BaseWsExceptionFilterOptions {
    /**
     * When true, the data that caused the exception will be included in the response.
     * This is useful when you want to provide additional context to the client, or
     * when you need to associate the error with a specific request.
     * @default true
     */
    includeCause?: boolean;
    /**
     * A factory function that can be used to control the shape of the "cause" object.
     * This is useful when you need a custom structure for the cause object.
     * @default (pattern, data) => ({ pattern, data })
     */
    causeFactory?: (pattern: string, data: unknown) => Record<string, any>;
}
/**
 * @publicApi
 */
export declare class BaseWsExceptionFilter<TError = any> implements WsExceptionFilter<TError> {
    protected readonly options: BaseWsExceptionFilterOptions;
    protected static readonly logger: Logger;
    constructor(options?: BaseWsExceptionFilterOptions);
    catch(exception: TError, host: ArgumentsHost): void;
    handleError<TClient extends {
        emit: Function;
    }>(client: TClient, exception: TError, cause: ErrorPayload['cause']): any;
    handleUnknownError<TClient extends {
        emit: Function;
    }>(exception: TError, client: TClient, data: ErrorPayload['cause']): void;
    isExceptionObject(err: any): err is Error;
}
export {};
