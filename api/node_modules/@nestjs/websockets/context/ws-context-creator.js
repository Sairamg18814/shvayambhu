"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.WsContextCreator = void 0;
const constants_1 = require("@nestjs/common/constants");
const shared_utils_1 = require("@nestjs/common/utils/shared.utils");
const constants_2 = require("@nestjs/core/guards/constants");
const context_utils_1 = require("@nestjs/core/helpers/context-utils");
const handler_metadata_storage_1 = require("@nestjs/core/helpers/handler-metadata-storage");
const constants_3 = require("../constants");
const ws_exception_1 = require("../errors/ws-exception");
const ws_params_factory_1 = require("../factories/ws-params-factory");
const ws_metadata_constants_1 = require("./ws-metadata-constants");
class WsContextCreator {
    constructor(wsProxy, exceptionFiltersContext, pipesContextCreator, pipesConsumer, guardsContextCreator, guardsConsumer, interceptorsContextCreator, interceptorsConsumer) {
        this.wsProxy = wsProxy;
        this.exceptionFiltersContext = exceptionFiltersContext;
        this.pipesContextCreator = pipesContextCreator;
        this.pipesConsumer = pipesConsumer;
        this.guardsContextCreator = guardsContextCreator;
        this.guardsConsumer = guardsConsumer;
        this.interceptorsContextCreator = interceptorsContextCreator;
        this.interceptorsConsumer = interceptorsConsumer;
        this.contextUtils = new context_utils_1.ContextUtils();
        this.wsParamsFactory = new ws_params_factory_1.WsParamsFactory();
        this.handlerMetadataStorage = new handler_metadata_storage_1.HandlerMetadataStorage();
    }
    create(instance, callback, moduleKey, methodName) {
        const contextType = 'ws';
        const { argsLength, paramtypes, getParamsMetadata } = this.getMetadata(instance, methodName, contextType);
        const exceptionHandler = this.exceptionFiltersContext.create(instance, callback, moduleKey);
        const pipes = this.pipesContextCreator.create(instance, callback, moduleKey);
        const guards = this.guardsContextCreator.create(instance, callback, moduleKey);
        const interceptors = this.interceptorsContextCreator.create(instance, callback, moduleKey);
        const paramsMetadata = getParamsMetadata(moduleKey);
        const paramsOptions = paramsMetadata
            ? this.contextUtils.mergeParamsMetatypes(paramsMetadata, paramtypes)
            : [];
        const fnApplyPipes = this.createPipesFn(pipes, paramsOptions);
        const fnCanActivate = this.createGuardsFn(guards, instance, callback, contextType);
        const handler = (initialArgs, args) => async () => {
            if (fnApplyPipes) {
                await fnApplyPipes(initialArgs, ...args);
                return callback.apply(instance, initialArgs);
            }
            return callback.apply(instance, args);
        };
        const targetPattern = this.reflectCallbackPattern(callback);
        return this.wsProxy.create(async (...args) => {
            args.push(targetPattern);
            const initialArgs = this.contextUtils.createNullArray(argsLength);
            fnCanActivate && (await fnCanActivate(args));
            return this.interceptorsConsumer.intercept(interceptors, args, instance, callback, handler(initialArgs, args), contextType);
        }, exceptionHandler, targetPattern);
    }
    reflectCallbackParamtypes(instance, callback) {
        return Reflect.getMetadata(constants_1.PARAMTYPES_METADATA, instance, callback.name);
    }
    reflectCallbackPattern(callback) {
        return Reflect.getMetadata(constants_3.MESSAGE_METADATA, callback);
    }
    createGuardsFn(guards, instance, callback, contextType) {
        const canActivateFn = async (args) => {
            const canActivate = await this.guardsConsumer.tryActivate(guards, args, instance, callback, contextType);
            if (!canActivate) {
                throw new ws_exception_1.WsException(constants_2.FORBIDDEN_MESSAGE);
            }
        };
        return guards.length ? canActivateFn : null;
    }
    getMetadata(instance, methodName, contextType) {
        const cacheMetadata = this.handlerMetadataStorage.get(instance, methodName);
        if (cacheMetadata) {
            return cacheMetadata;
        }
        const metadata = this.contextUtils.reflectCallbackMetadata(instance, methodName, constants_3.PARAM_ARGS_METADATA) || ws_metadata_constants_1.DEFAULT_CALLBACK_METADATA;
        const keys = Object.keys(metadata);
        const argsLength = this.contextUtils.getArgumentsLength(keys, metadata);
        const paramtypes = this.contextUtils.reflectCallbackParamtypes(instance, methodName);
        const contextFactory = this.contextUtils.getContextFactory(contextType, instance, instance[methodName]);
        const getParamsMetadata = (moduleKey) => this.exchangeKeysForValues(keys, metadata, moduleKey, this.wsParamsFactory, contextFactory);
        const handlerMetadata = {
            argsLength,
            paramtypes,
            getParamsMetadata,
        };
        this.handlerMetadataStorage.set(instance, methodName, handlerMetadata);
        return handlerMetadata;
    }
    exchangeKeysForValues(keys, metadata, moduleContext, paramsFactory, contextFactory) {
        this.pipesContextCreator.setModuleContext(moduleContext);
        return keys.map(key => {
            const { index, data, pipes: pipesCollection } = metadata[key];
            const pipes = this.pipesContextCreator.createConcreteContext(pipesCollection);
            const type = this.contextUtils.mapParamType(key);
            if (key.includes(constants_1.CUSTOM_ROUTE_ARGS_METADATA)) {
                const { factory } = metadata[key];
                const customExtractValue = this.contextUtils.getCustomFactory(factory, data, contextFactory);
                return { index, extractValue: customExtractValue, type, data, pipes };
            }
            const numericType = Number(type);
            const extractValue = (...args) => paramsFactory.exchangeKeyForValue(numericType, data, args);
            return { index, extractValue, type: numericType, data, pipes };
        });
    }
    createPipesFn(pipes, paramsOptions) {
        const pipesFn = async (args, ...params) => {
            const resolveParamValue = async (param) => {
                const { index, extractValue, type, data, metatype, pipes: paramPipes, } = param;
                const value = extractValue(...params);
                args[index] = await this.getParamValue(value, { metatype, type, data }, pipes.concat(paramPipes));
            };
            await Promise.all(paramsOptions.map(resolveParamValue));
        };
        return paramsOptions.length ? pipesFn : null;
    }
    async getParamValue(value, { metatype, type, data }, pipes) {
        return (0, shared_utils_1.isEmpty)(pipes)
            ? value
            : this.pipesConsumer.apply(value, { metatype, type, data }, pipes);
    }
}
exports.WsContextCreator = WsContextCreator;
