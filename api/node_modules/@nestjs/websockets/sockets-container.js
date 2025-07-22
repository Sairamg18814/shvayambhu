"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.SocketsContainer = void 0;
const hash = require("object-hash");
class SocketsContainer {
    constructor() {
        this.serverAndEventStreamsHosts = new Map();
    }
    getAll() {
        return this.serverAndEventStreamsHosts;
    }
    getOneByConfig(options) {
        const uniqueToken = this.generateHashByOptions(options);
        return this.serverAndEventStreamsHosts.get(uniqueToken);
    }
    addOne(options, host) {
        const uniqueToken = this.generateHashByOptions(options);
        this.serverAndEventStreamsHosts.set(uniqueToken, host);
    }
    clear() {
        this.serverAndEventStreamsHosts.clear();
    }
    generateHashByOptions(options) {
        return hash(options, { ignoreUnknown: true });
    }
}
exports.SocketsContainer = SocketsContainer;
