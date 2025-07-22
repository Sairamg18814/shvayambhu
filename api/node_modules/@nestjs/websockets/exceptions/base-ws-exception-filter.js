"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.BaseWsExceptionFilter = void 0;
const common_1 = require("@nestjs/common");
const shared_utils_1 = require("@nestjs/common/utils/shared.utils");
const constants_1 = require("@nestjs/core/constants");
const ws_exception_1 = require("../errors/ws-exception");
/**
 * @publicApi
 */
class BaseWsExceptionFilter {
    constructor(options = {}) {
        this.options = options;
        this.options.includeCause = this.options.includeCause ?? true;
        this.options.causeFactory =
            this.options.causeFactory ?? ((pattern, data) => ({ pattern, data }));
    }
    catch(exception, host) {
        const client = host.switchToWs().getClient();
        const pattern = host.switchToWs().getPattern();
        const data = host.switchToWs().getData();
        this.handleError(client, exception, {
            pattern,
            data,
        });
    }
    handleError(client, exception, cause) {
        if (!(exception instanceof ws_exception_1.WsException)) {
            return this.handleUnknownError(exception, client, cause);
        }
        const status = 'error';
        const result = exception.getError();
        if ((0, shared_utils_1.isObject)(result)) {
            return client.emit('exception', result);
        }
        const payload = {
            status,
            message: result,
        };
        if (this.options?.includeCause && cause) {
            payload.cause = this.options.causeFactory(cause.pattern, cause.data);
        }
        client.emit('exception', payload);
    }
    handleUnknownError(exception, client, data) {
        const status = 'error';
        const payload = {
            status,
            message: constants_1.MESSAGES.UNKNOWN_EXCEPTION_MESSAGE,
        };
        if (this.options?.includeCause && data) {
            payload.cause = this.options.causeFactory(data.pattern, data.data);
        }
        client.emit('exception', payload);
        if (!(exception instanceof common_1.IntrinsicException)) {
            const logger = BaseWsExceptionFilter.logger;
            logger.error(exception);
        }
    }
    isExceptionObject(err) {
        return (0, shared_utils_1.isObject)(err) && !!err.message;
    }
}
exports.BaseWsExceptionFilter = BaseWsExceptionFilter;
BaseWsExceptionFilter.logger = new common_1.Logger('WsExceptionsHandler');
