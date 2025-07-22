"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.ExceptionFiltersContext = void 0;
const constants_1 = require("@nestjs/common/constants");
const shared_utils_1 = require("@nestjs/common/utils/shared.utils");
const base_exception_filter_context_1 = require("@nestjs/core/exceptions/base-exception-filter-context");
const ws_exceptions_handler_1 = require("../exceptions/ws-exceptions-handler");
/**
 * @publicApi
 */
class ExceptionFiltersContext extends base_exception_filter_context_1.BaseExceptionFilterContext {
    constructor(container) {
        super(container);
    }
    create(instance, callback, moduleKey) {
        this.moduleContext = moduleKey;
        const exceptionHandler = new ws_exceptions_handler_1.WsExceptionsHandler();
        const filters = this.createContext(instance, callback, constants_1.EXCEPTION_FILTERS_METADATA);
        if ((0, shared_utils_1.isEmpty)(filters)) {
            return exceptionHandler;
        }
        exceptionHandler.setCustomFilters(filters.reverse());
        return exceptionHandler;
    }
    getGlobalMetadata() {
        return [];
    }
}
exports.ExceptionFiltersContext = ExceptionFiltersContext;
