"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.WsExceptionsHandler = void 0;
const select_exception_filter_metadata_util_1 = require("@nestjs/common/utils/select-exception-filter-metadata.util");
const shared_utils_1 = require("@nestjs/common/utils/shared.utils");
const invalid_exception_filter_exception_1 = require("@nestjs/core/errors/exceptions/invalid-exception-filter.exception");
const base_ws_exception_filter_1 = require("./base-ws-exception-filter");
/**
 * @publicApi
 */
class WsExceptionsHandler extends base_ws_exception_filter_1.BaseWsExceptionFilter {
    constructor() {
        super(...arguments);
        this.filters = [];
    }
    handle(exception, host) {
        const client = host.switchToWs().getClient();
        if (this.invokeCustomFilters(exception, host) || !client.emit) {
            return;
        }
        super.catch(exception, host);
    }
    setCustomFilters(filters) {
        if (!Array.isArray(filters)) {
            throw new invalid_exception_filter_exception_1.InvalidExceptionFilterException();
        }
        this.filters = filters;
    }
    invokeCustomFilters(exception, args) {
        if ((0, shared_utils_1.isEmpty)(this.filters))
            return false;
        const filter = (0, select_exception_filter_metadata_util_1.selectExceptionFilterMetadata)(this.filters, exception);
        filter && filter.func(exception, args);
        return !!filter;
    }
}
exports.WsExceptionsHandler = WsExceptionsHandler;
