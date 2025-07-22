"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.SocketModule = void 0;
const guards_consumer_1 = require("@nestjs/core/guards/guards-consumer");
const guards_context_creator_1 = require("@nestjs/core/guards/guards-context-creator");
const load_adapter_1 = require("@nestjs/core/helpers/load-adapter");
const interceptors_consumer_1 = require("@nestjs/core/interceptors/interceptors-consumer");
const interceptors_context_creator_1 = require("@nestjs/core/interceptors/interceptors-context-creator");
const pipes_consumer_1 = require("@nestjs/core/pipes/pipes-consumer");
const pipes_context_creator_1 = require("@nestjs/core/pipes/pipes-context-creator");
const iterare_1 = require("iterare");
const constants_1 = require("./constants");
const exception_filters_context_1 = require("./context/exception-filters-context");
const ws_context_creator_1 = require("./context/ws-context-creator");
const ws_proxy_1 = require("./context/ws-proxy");
const socket_server_provider_1 = require("./socket-server-provider");
const sockets_container_1 = require("./sockets-container");
const web_sockets_controller_1 = require("./web-sockets-controller");
class SocketModule {
    constructor() {
        this.socketsContainer = new sockets_container_1.SocketsContainer();
    }
    register(container, applicationConfig, graphInspector, appOptions, httpServer) {
        this.applicationConfig = applicationConfig;
        this.appOptions = appOptions;
        this.httpServer = httpServer;
        const contextCreator = this.getContextCreator(container);
        const serverProvider = new socket_server_provider_1.SocketServerProvider(this.socketsContainer, applicationConfig);
        this.webSocketsController = new web_sockets_controller_1.WebSocketsController(serverProvider, applicationConfig, contextCreator, graphInspector, this.appOptions);
        const modules = container.getModules();
        modules.forEach(({ providers }, moduleName) => this.connectAllGateways(providers, moduleName));
    }
    connectAllGateways(providers, moduleName) {
        (0, iterare_1.iterate)(providers.values())
            .filter(wrapper => wrapper && !wrapper.isNotMetatype)
            .forEach(wrapper => this.connectGatewayToServer(wrapper, moduleName));
    }
    connectGatewayToServer(wrapper, moduleName) {
        const { instance, metatype } = wrapper;
        const metadataKeys = Reflect.getMetadataKeys(metatype);
        if (!metadataKeys.includes(constants_1.GATEWAY_METADATA)) {
            return;
        }
        if (!this.isAdapterInitialized) {
            this.initializeAdapter();
        }
        this.webSocketsController.connectGatewayToServer(instance, metatype, moduleName, wrapper.id);
    }
    async close() {
        if (!this.applicationConfig) {
            return;
        }
        const adapter = this.applicationConfig.getIoAdapter();
        if (!adapter) {
            return;
        }
        const servers = this.socketsContainer.getAll();
        await Promise.all((0, iterare_1.iterate)(servers.values())
            .filter(({ server }) => server)
            .map(async ({ server }) => adapter.close(server)));
        await adapter?.dispose();
        this.socketsContainer.clear();
    }
    initializeAdapter() {
        const forceCloseConnections = this.appOptions
            .forceCloseConnections;
        const adapter = this.applicationConfig.getIoAdapter();
        if (adapter) {
            adapter.forceCloseConnections =
                forceCloseConnections;
            this.isAdapterInitialized = true;
            return;
        }
        const { IoAdapter } = (0, load_adapter_1.loadAdapter)('@nestjs/platform-socket.io', 'WebSockets', () => require('@nestjs/platform-socket.io'));
        const ioAdapter = new IoAdapter(this.httpServer);
        ioAdapter.forceCloseConnections = forceCloseConnections;
        this.applicationConfig.setIoAdapter(ioAdapter);
        this.isAdapterInitialized = true;
    }
    getContextCreator(container) {
        return new ws_context_creator_1.WsContextCreator(new ws_proxy_1.WsProxy(), new exception_filters_context_1.ExceptionFiltersContext(container), new pipes_context_creator_1.PipesContextCreator(container), new pipes_consumer_1.PipesConsumer(), new guards_context_creator_1.GuardsContextCreator(container), new guards_consumer_1.GuardsConsumer(), new interceptors_context_creator_1.InterceptorsContextCreator(container), new interceptors_consumer_1.InterceptorsConsumer());
    }
}
exports.SocketModule = SocketModule;
