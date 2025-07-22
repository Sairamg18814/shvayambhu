"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.DEFAULT_CALLBACK_METADATA = void 0;
const ws_paramtype_enum_1 = require("../enums/ws-paramtype.enum");
exports.DEFAULT_CALLBACK_METADATA = {
    [`${ws_paramtype_enum_1.WsParamtype.PAYLOAD}:1`]: { index: 1, data: undefined, pipes: [] },
    [`${ws_paramtype_enum_1.WsParamtype.SOCKET}:0`]: { index: 0, data: undefined, pipes: [] },
};
