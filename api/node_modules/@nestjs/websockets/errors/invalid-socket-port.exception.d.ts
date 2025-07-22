import { RuntimeException } from '@nestjs/core/errors/exceptions/runtime.exception';
export declare class InvalidSocketPortException extends RuntimeException {
    constructor(port: number | string, type: any);
}
