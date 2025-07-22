import { GatewayMetadata } from '../interfaces';
/**
 * Decorator that marks a class as a Nest gateway that enables real-time, bidirectional
 * and event-based communication between the browser and the server.
 *
 * @publicApi
 */
export declare function WebSocketGateway(port?: number): ClassDecorator;
export declare function WebSocketGateway<T extends Record<string, any> = GatewayMetadata>(options?: T): ClassDecorator;
export declare function WebSocketGateway<T extends Record<string, any> = GatewayMetadata>(port?: number, options?: T): ClassDecorator;
