import { PipeTransform, Type } from '@nestjs/common';
import 'reflect-metadata';
import { WsParamtype } from '../enums/ws-paramtype.enum';
export declare function createWsParamDecorator(paramtype: WsParamtype): (...pipes: (Type<PipeTransform> | PipeTransform)[]) => ParameterDecorator;
export declare const createPipesWsParamDecorator: (paramtype: WsParamtype) => (data?: any, ...pipes: (Type<PipeTransform> | PipeTransform)[]) => ParameterDecorator;
