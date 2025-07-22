import { ServerAndEventStreamsHost } from '../interfaces/server-and-event-streams-host.interface';
export declare class ServerAndEventStreamsFactory {
    static create<T = any>(server: T): ServerAndEventStreamsHost<T>;
}
