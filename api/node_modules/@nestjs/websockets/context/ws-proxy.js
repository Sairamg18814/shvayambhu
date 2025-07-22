"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.WsProxy = void 0;
const execution_context_host_1 = require("@nestjs/core/helpers/execution-context-host");
const rxjs_1 = require("rxjs");
const operators_1 = require("rxjs/operators");
class WsProxy {
    create(targetCallback, exceptionsHandler, targetPattern) {
        return async (...args) => {
            args = [...args, targetPattern ?? 'unknown'];
            try {
                const result = await targetCallback(...args);
                return !(0, rxjs_1.isObservable)(result)
                    ? result
                    : result.pipe((0, operators_1.catchError)(error => {
                        this.handleError(exceptionsHandler, args, error);
                        return rxjs_1.EMPTY;
                    }));
            }
            catch (error) {
                this.handleError(exceptionsHandler, args, error);
            }
        };
    }
    handleError(exceptionsHandler, args, error) {
        const host = new execution_context_host_1.ExecutionContextHost(args);
        host.setType('ws');
        exceptionsHandler.handle(error, host);
    }
}
exports.WsProxy = WsProxy;
