"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.SocketServerProvider = void 0;
const shared_utils_1 = require("@nestjs/common/utils/shared.utils");
const server_and_event_streams_factory_1 = require("./factories/server-and-event-streams-factory");
class SocketServerProvider {
    constructor(socketsContainer, applicationConfig) {
        this.socketsContainer = socketsContainer;
        this.applicationConfig = applicationConfig;
    }
    scanForSocketServer(options, port) {
        const serverAndStreamsHost = this.socketsContainer.getOneByConfig({
            port,
            path: options.path,
        });
        if (serverAndStreamsHost && options.namespace) {
            return this.decorateWithNamespace(options, port, serverAndStreamsHost.server);
        }
        return serverAndStreamsHost
            ? serverAndStreamsHost
            : this.createSocketServer(options, port);
    }
    createSocketServer(options, port) {
        const adapter = this.applicationConfig.getIoAdapter();
        const { namespace, server, ...partialOptions } = options;
        const ioServer = adapter.create(port, partialOptions);
        const serverAndEventStreamsHost = server_and_event_streams_factory_1.ServerAndEventStreamsFactory.create(ioServer);
        this.socketsContainer.addOne({ port, path: options.path }, serverAndEventStreamsHost);
        if (!namespace) {
            return serverAndEventStreamsHost;
        }
        return this.decorateWithNamespace(options, port, ioServer);
    }
    decorateWithNamespace(options, port, targetServer) {
        const namespaceServer = this.getServerOfNamespace(options, port, targetServer);
        const serverAndEventStreamsHost = server_and_event_streams_factory_1.ServerAndEventStreamsFactory.create(namespaceServer);
        this.socketsContainer.addOne({ port, path: options.path, namespace: options.namespace }, serverAndEventStreamsHost);
        return serverAndEventStreamsHost;
    }
    getServerOfNamespace(options, port, server) {
        const adapter = this.applicationConfig.getIoAdapter();
        return adapter.create(port, {
            ...options,
            namespace: this.validateNamespace(options.namespace || ''),
            server,
        });
    }
    validateNamespace(namespace) {
        if (!(0, shared_utils_1.isString)(namespace)) {
            return namespace;
        }
        return (0, shared_utils_1.addLeadingSlash)(namespace);
    }
}
exports.SocketServerProvider = SocketServerProvider;
