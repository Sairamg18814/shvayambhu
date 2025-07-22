"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.SubscribeMessage = void 0;
const constants_1 = require("../constants");
/**
 * Subscribes to messages that fulfils chosen pattern.
 *
 * @publicApi
 */
const SubscribeMessage = (message) => {
    return (target, key, descriptor) => {
        Reflect.defineMetadata(constants_1.MESSAGE_MAPPING_METADATA, true, descriptor.value);
        Reflect.defineMetadata(constants_1.MESSAGE_METADATA, message, descriptor.value);
        return descriptor;
    };
};
exports.SubscribeMessage = SubscribeMessage;
