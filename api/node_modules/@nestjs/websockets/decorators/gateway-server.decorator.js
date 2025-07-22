"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.WebSocketServer = void 0;
const constants_1 = require("../constants");
/**
 * Attaches native Web Socket Server to a given property.
 *
 * @publicApi
 */
const WebSocketServer = () => {
    return (target, propertyKey) => {
        Reflect.set(target, propertyKey, null);
        Reflect.defineMetadata(constants_1.GATEWAY_SERVER_METADATA, true, target, propertyKey);
    };
};
exports.WebSocketServer = WebSocketServer;
