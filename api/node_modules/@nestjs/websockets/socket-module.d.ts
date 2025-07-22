import { InjectionToken } from '@nestjs/common/interfaces';
import { Injectable } from '@nestjs/common/interfaces/injectable.interface';
import { NestApplicationContextOptions } from '@nestjs/common/interfaces/nest-application-context-options.interface';
import { ApplicationConfig } from '@nestjs/core/application-config';
import { NestContainer } from '@nestjs/core/injector/container';
import { InstanceWrapper } from '@nestjs/core/injector/instance-wrapper';
import { GraphInspector } from '@nestjs/core/inspector/graph-inspector';
export declare class SocketModule<THttpServer = any, TAppOptions extends NestApplicationContextOptions = NestApplicationContextOptions> {
    private readonly socketsContainer;
    private applicationConfig;
    private webSocketsController;
    private isAdapterInitialized;
    private httpServer;
    private appOptions;
    register(container: NestContainer, applicationConfig: ApplicationConfig, graphInspector: GraphInspector, appOptions: TAppOptions, httpServer?: THttpServer): void;
    connectAllGateways(providers: Map<InjectionToken, InstanceWrapper<Injectable>>, moduleName: string): void;
    connectGatewayToServer(wrapper: InstanceWrapper<Injectable>, moduleName: string): void;
    close(): Promise<any>;
    private initializeAdapter;
    private getContextCreator;
}
