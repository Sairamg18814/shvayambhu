"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.WebSocketsController = void 0;
const logger_service_1 = require("@nestjs/common/services/logger.service");
const metadata_scanner_1 = require("@nestjs/core/metadata-scanner");
const rxjs_1 = require("rxjs");
const operators_1 = require("rxjs/operators");
const constants_1 = require("./constants");
const invalid_socket_port_exception_1 = require("./errors/invalid-socket-port.exception");
const gateway_metadata_explorer_1 = require("./gateway-metadata-explorer");
const compare_element_util_1 = require("./utils/compare-element.util");
class WebSocketsController {
    constructor(socketServerProvider, config, contextCreator, graphInspector, appOptions = {}) {
        this.socketServerProvider = socketServerProvider;
        this.config = config;
        this.contextCreator = contextCreator;
        this.graphInspector = graphInspector;
        this.appOptions = appOptions;
        this.logger = new logger_service_1.Logger(WebSocketsController.name, {
            timestamp: true,
        });
        this.metadataExplorer = new gateway_metadata_explorer_1.GatewayMetadataExplorer(new metadata_scanner_1.MetadataScanner());
    }
    connectGatewayToServer(instance, metatype, moduleKey, instanceWrapperId) {
        const options = Reflect.getMetadata(constants_1.GATEWAY_OPTIONS, metatype) || {};
        const port = Reflect.getMetadata(constants_1.PORT_METADATA, metatype) || 0;
        if (!Number.isInteger(port)) {
            throw new invalid_socket_port_exception_1.InvalidSocketPortException(port, metatype);
        }
        this.subscribeToServerEvents(instance, options, port, moduleKey, instanceWrapperId);
    }
    subscribeToServerEvents(instance, options, port, moduleKey, instanceWrapperId) {
        const nativeMessageHandlers = this.metadataExplorer.explore(instance);
        const messageHandlers = nativeMessageHandlers.map(({ callback, message, methodName }) => ({
            message,
            methodName,
            callback: this.contextCreator.create(instance, callback, moduleKey, methodName),
        }));
        this.inspectEntrypointDefinitions(instance, port, messageHandlers, instanceWrapperId);
        if (this.appOptions.preview) {
            return;
        }
        const observableServer = this.socketServerProvider.scanForSocketServer(options, port);
        this.assignServerToProperties(instance, observableServer.server);
        this.subscribeEvents(instance, messageHandlers, observableServer);
    }
    subscribeEvents(instance, subscribersMap, observableServer) {
        const { init, disconnect, connection, server } = observableServer;
        const adapter = this.config.getIoAdapter();
        this.subscribeInitEvent(instance, init);
        this.subscribeConnectionEvent(instance, connection);
        this.subscribeDisconnectEvent(instance, disconnect);
        const handler = this.getConnectionHandler(this, instance, subscribersMap, disconnect, connection);
        adapter.bindClientConnect(server, handler);
        this.printSubscriptionLogs(instance, subscribersMap);
    }
    getConnectionHandler(context, instance, subscribersMap, disconnect, connection) {
        const adapter = this.config.getIoAdapter();
        return (...args) => {
            const [client] = args;
            connection.next(args);
            context.subscribeMessages(subscribersMap, client, instance);
            const disconnectHook = adapter.bindClientDisconnect;
            disconnectHook &&
                disconnectHook.call(adapter, client, () => disconnect.next(client));
        };
    }
    subscribeInitEvent(instance, event) {
        if (instance.afterInit) {
            event.subscribe(instance.afterInit.bind(instance));
        }
    }
    subscribeConnectionEvent(instance, event) {
        if (instance.handleConnection) {
            event
                .pipe((0, operators_1.distinctUntilChanged)((prev, curr) => (0, compare_element_util_1.compareElementAt)(prev, curr, 0)))
                .subscribe((args) => instance.handleConnection(...args));
        }
    }
    subscribeDisconnectEvent(instance, event) {
        if (instance.handleDisconnect) {
            event
                .pipe((0, operators_1.distinctUntilChanged)())
                .subscribe(instance.handleDisconnect.bind(instance));
        }
    }
    subscribeMessages(subscribersMap, client, instance) {
        const adapter = this.config.getIoAdapter();
        const handlers = subscribersMap.map(({ callback, message }) => ({
            message,
            callback: callback.bind(instance, client),
        }));
        adapter.bindMessageHandlers(client, handlers, data => (0, rxjs_1.from)(this.pickResult(data)).pipe((0, operators_1.mergeAll)()));
    }
    async pickResult(deferredResult) {
        const result = await deferredResult;
        if ((0, rxjs_1.isObservable)(result)) {
            return result;
        }
        if (result instanceof Promise) {
            return (0, rxjs_1.from)(result);
        }
        return (0, rxjs_1.of)(result);
    }
    inspectEntrypointDefinitions(instance, port, messageHandlers, instanceWrapperId) {
        messageHandlers.forEach(handler => {
            this.graphInspector.insertEntrypointDefinition({
                type: 'websocket',
                methodName: handler.methodName,
                className: instance.constructor?.name,
                classNodeId: instanceWrapperId,
                metadata: {
                    port,
                    key: handler.message,
                    message: handler.message,
                },
            }, instanceWrapperId);
        });
    }
    assignServerToProperties(instance, server) {
        for (const propertyKey of this.metadataExplorer.scanForServerHooks(instance)) {
            Reflect.set(instance, propertyKey, server);
        }
    }
    printSubscriptionLogs(instance, subscribersMap) {
        const gatewayClassName = instance?.constructor?.name;
        if (!gatewayClassName) {
            return;
        }
        subscribersMap.forEach(({ message }) => this.logger.log(`${gatewayClassName} subscribed to the "${message}" message`));
    }
}
exports.WebSocketsController = WebSocketsController;
