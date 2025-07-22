"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.ConnectedSocket = void 0;
const ws_paramtype_enum_1 = require("../enums/ws-paramtype.enum");
const param_utils_1 = require("../utils/param.utils");
/**
 * @publicApi
 */
exports.ConnectedSocket = (0, param_utils_1.createWsParamDecorator)(ws_paramtype_enum_1.WsParamtype.SOCKET);
