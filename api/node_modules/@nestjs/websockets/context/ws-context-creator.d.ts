import { ContextType, Controller, PipeTransform } from '@nestjs/common/interfaces';
import { GuardsConsumer } from '@nestjs/core/guards/guards-consumer';
import { GuardsContextCreator } from '@nestjs/core/guards/guards-context-creator';
import { ParamProperties } from '@nestjs/core/helpers/context-utils';
import { ExecutionContextHost } from '@nestjs/core/helpers/execution-context-host';
import { ParamsMetadata } from '@nestjs/core/helpers/interfaces';
import { InterceptorsConsumer, InterceptorsContextCreator } from '@nestjs/core/interceptors';
import { PipesConsumer, PipesContextCreator } from '@nestjs/core/pipes';
import { WsParamsFactory } from '../factories/ws-params-factory';
import { ExceptionFiltersContext } from './exception-filters-context';
import { WsProxy } from './ws-proxy';
type WsParamProperties = ParamProperties & {
    metatype?: any;
};
export interface WsHandlerMetadata {
    argsLength: number;
    paramtypes: any[];
    getParamsMetadata: (moduleKey: string) => WsParamProperties[];
}
export declare class WsContextCreator {
    private readonly wsProxy;
    private readonly exceptionFiltersContext;
    private readonly pipesContextCreator;
    private readonly pipesConsumer;
    private readonly guardsContextCreator;
    private readonly guardsConsumer;
    private readonly interceptorsContextCreator;
    private readonly interceptorsConsumer;
    private readonly contextUtils;
    private readonly wsParamsFactory;
    private readonly handlerMetadataStorage;
    constructor(wsProxy: WsProxy, exceptionFiltersContext: ExceptionFiltersContext, pipesContextCreator: PipesContextCreator, pipesConsumer: PipesConsumer, guardsContextCreator: GuardsContextCreator, guardsConsumer: GuardsConsumer, interceptorsContextCreator: InterceptorsContextCreator, interceptorsConsumer: InterceptorsConsumer);
    create<T extends ParamsMetadata = ParamsMetadata>(instance: Controller, callback: (...args: unknown[]) => void, moduleKey: string, methodName: string): (...args: any[]) => Promise<void>;
    reflectCallbackParamtypes(instance: Controller, callback: (...args: any[]) => any): any[];
    reflectCallbackPattern(callback: (...args: any[]) => any): string;
    createGuardsFn<TContext extends string = ContextType>(guards: any[], instance: Controller, callback: (...args: unknown[]) => any, contextType?: TContext): Function | null;
    getMetadata<TMetadata, TContext extends ContextType = ContextType>(instance: Controller, methodName: string, contextType: TContext): WsHandlerMetadata;
    exchangeKeysForValues<TMetadata = any>(keys: string[], metadata: TMetadata, moduleContext: string, paramsFactory: WsParamsFactory, contextFactory: (args: unknown[]) => ExecutionContextHost): ParamProperties[];
    createPipesFn(pipes: PipeTransform[], paramsOptions: (ParamProperties & {
        metatype?: unknown;
    })[]): ((args: unknown[], ...params: unknown[]) => Promise<void>) | null;
    getParamValue<T>(value: T, { metatype, type, data }: {
        metatype: any;
        type: any;
        data: any;
    }, pipes: PipeTransform[]): Promise<any>;
}
export {};
