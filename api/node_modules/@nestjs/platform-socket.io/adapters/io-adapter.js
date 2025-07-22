"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.IoAdapter = void 0;
const shared_utils_1 = require("@nestjs/common/utils/shared.utils");
const websockets_1 = require("@nestjs/websockets");
const constants_1 = require("@nestjs/websockets/constants");
const rxjs_1 = require("rxjs");
const operators_1 = require("rxjs/operators");
const socket_io_1 = require("socket.io");
/**
 * @publicApi
 */
class IoAdapter extends websockets_1.AbstractWsAdapter {
    create(port, options) {
        if (!options) {
            return this.createIOServer(port);
        }
        const { namespace, server, ...opt } = options;
        return server && (0, shared_utils_1.isFunction)(server.of)
            ? server.of(namespace)
            : namespace
                ? this.createIOServer(port, opt).of(namespace)
                : this.createIOServer(port, opt);
    }
    createIOServer(port, options) {
        if (this.httpServer && port === 0) {
            return new socket_io_1.Server(this.httpServer, options);
        }
        return new socket_io_1.Server(port, options);
    }
    bindMessageHandlers(socket, handlers, transform) {
        const disconnect$ = (0, rxjs_1.fromEvent)(socket, constants_1.DISCONNECT_EVENT).pipe((0, operators_1.share)(), (0, operators_1.first)());
        handlers.forEach(({ message, callback }) => {
            const source$ = (0, rxjs_1.fromEvent)(socket, message).pipe((0, operators_1.mergeMap)((payload) => {
                const { data, ack } = this.mapPayload(payload);
                return transform(callback(data, ack)).pipe((0, operators_1.filter)((response) => !(0, shared_utils_1.isNil)(response)), (0, operators_1.map)((response) => [response, ack]));
            }), (0, operators_1.takeUntil)(disconnect$));
            source$.subscribe(([response, ack]) => {
                if (response.event) {
                    return socket.emit(response.event, response.data);
                }
                (0, shared_utils_1.isFunction)(ack) && ack(response);
            });
        });
    }
    mapPayload(payload) {
        if (!Array.isArray(payload)) {
            if ((0, shared_utils_1.isFunction)(payload)) {
                return { data: undefined, ack: payload };
            }
            return { data: payload };
        }
        const lastElement = payload[payload.length - 1];
        const isAck = (0, shared_utils_1.isFunction)(lastElement);
        if (isAck) {
            const size = payload.length - 1;
            return {
                data: size === 1 ? payload[0] : payload.slice(0, size),
                ack: lastElement,
            };
        }
        return { data: payload };
    }
    async close(server) {
        if (this.forceCloseConnections && server.httpServer === this.httpServer) {
            return;
        }
        return super.close(server);
    }
}
exports.IoAdapter = IoAdapter;
