"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.ServerAndEventStreamsFactory = void 0;
const rxjs_1 = require("rxjs");
class ServerAndEventStreamsFactory {
    static create(server) {
        const init = new rxjs_1.ReplaySubject();
        init.next(server);
        const connection = new rxjs_1.Subject();
        const disconnect = new rxjs_1.Subject();
        return {
            init,
            connection,
            disconnect,
            server,
        };
    }
}
exports.ServerAndEventStreamsFactory = ServerAndEventStreamsFactory;
