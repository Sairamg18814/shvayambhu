"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.MessageBody = MessageBody;
const ws_paramtype_enum_1 = require("../enums/ws-paramtype.enum");
const param_utils_1 = require("../utils/param.utils");
function MessageBody(propertyOrPipe, ...pipes) {
    return (0, param_utils_1.createPipesWsParamDecorator)(ws_paramtype_enum_1.WsParamtype.PAYLOAD)(propertyOrPipe, ...pipes);
}
