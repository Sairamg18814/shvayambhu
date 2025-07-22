"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.GatewayMetadataExplorer = void 0;
const shared_utils_1 = require("@nestjs/common/utils/shared.utils");
const constants_1 = require("./constants");
class GatewayMetadataExplorer {
    constructor(metadataScanner) {
        this.metadataScanner = metadataScanner;
    }
    explore(instance) {
        const instancePrototype = Object.getPrototypeOf(instance);
        return this.metadataScanner
            .getAllMethodNames(instancePrototype)
            .map(method => this.exploreMethodMetadata(instancePrototype, method))
            .filter(metadata => metadata);
    }
    exploreMethodMetadata(instancePrototype, methodName) {
        const callback = instancePrototype[methodName];
        const isMessageMapping = Reflect.getMetadata(constants_1.MESSAGE_MAPPING_METADATA, callback);
        if ((0, shared_utils_1.isUndefined)(isMessageMapping)) {
            return null;
        }
        const message = Reflect.getMetadata(constants_1.MESSAGE_METADATA, callback);
        return {
            callback,
            message,
            methodName,
        };
    }
    *scanForServerHooks(instance) {
        for (const propertyKey in instance) {
            if ((0, shared_utils_1.isFunction)(propertyKey)) {
                continue;
            }
            const property = String(propertyKey);
            const isServer = Reflect.getMetadata(constants_1.GATEWAY_SERVER_METADATA, instance, property);
            if (!(0, shared_utils_1.isUndefined)(isServer)) {
                yield property;
            }
        }
    }
}
exports.GatewayMetadataExplorer = GatewayMetadataExplorer;
