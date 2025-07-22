"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.WebSocketGateway = WebSocketGateway;
const constants_1 = require("../constants");
function WebSocketGateway(portOrOptions, options) {
    const isPortInt = Number.isInteger(portOrOptions);
    // eslint-disable-next-line prefer-const
    let [port, opt] = isPortInt ? [portOrOptions, options] : [0, portOrOptions];
    opt = opt || {};
    return (target) => {
        Reflect.defineMetadata(constants_1.GATEWAY_METADATA, true, target);
        Reflect.defineMetadata(constants_1.PORT_METADATA, port, target);
        Reflect.defineMetadata(constants_1.GATEWAY_OPTIONS, opt, target);
    };
}
