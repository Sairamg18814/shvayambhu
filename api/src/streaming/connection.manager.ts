import { Injectable, Logger } from '@nestjs/common';
import { Socket } from 'socket.io';

export interface ConnectionInfo {
  id: string;
  socket: Socket;
  connectedAt: Date;
  lastActivity: Date;
  reconnectCount: number;
  metadata: Map<string, any>;
}

export interface ConnectionConfig {
  maxReconnectAttempts: number;
  reconnectTimeoutMs: number;
  heartbeatIntervalMs: number;
  connectionTimeoutMs: number;
}

@Injectable()
export class ConnectionManager {
  private logger = new Logger('ConnectionManager');
  private connections = new Map<string, ConnectionInfo>();
  private reconnectTimers = new Map<string, NodeJS.Timeout>();
  private heartbeatTimers = new Map<string, NodeJS.Timeout>();

  private config: ConnectionConfig = {
    maxReconnectAttempts: 5,
    reconnectTimeoutMs: 5000,
    heartbeatIntervalMs: 30000,
    connectionTimeoutMs: 60000,
  };

  /**
   * Register a new connection
   */
  registerConnection(socket: Socket): void {
    const connectionInfo: ConnectionInfo = {
      id: socket.id,
      socket,
      connectedAt: new Date(),
      lastActivity: new Date(),
      reconnectCount: 0,
      metadata: new Map(),
    };

    this.connections.set(socket.id, connectionInfo);
    this.startHeartbeat(socket.id);
    
    this.logger.log(`Connection registered: ${socket.id}`);
  }

  /**
   * Handle connection disconnect
   */
  handleDisconnect(socketId: string): void {
    const connection = this.connections.get(socketId);
    
    if (connection) {
      // Start reconnection timer
      this.startReconnectTimer(socketId);
      this.stopHeartbeat(socketId);
      
      this.logger.log(`Connection disconnected: ${socketId}, waiting for reconnect...`);
    }
  }

  /**
   * Handle successful reconnection
   */
  handleReconnect(oldSocketId: string, newSocket: Socket): void {
    const connection = this.connections.get(oldSocketId);
    
    if (connection) {
      // Update connection info
      connection.socket = newSocket;
      connection.id = newSocket.id;
      connection.lastActivity = new Date();
      connection.reconnectCount++;
      
      // Move to new socket ID
      this.connections.delete(oldSocketId);
      this.connections.set(newSocket.id, connection);
      
      // Clear reconnect timer
      this.clearReconnectTimer(oldSocketId);
      
      // Restart heartbeat with new socket ID
      this.stopHeartbeat(oldSocketId);
      this.startHeartbeat(newSocket.id);
      
      this.logger.log(`Reconnection successful: ${oldSocketId} -> ${newSocket.id}`);
    } else {
      // New connection
      this.registerConnection(newSocket);
    }
  }

  /**
   * Update last activity timestamp
   */
  updateActivity(socketId: string): void {
    const connection = this.connections.get(socketId);
    if (connection) {
      connection.lastActivity = new Date();
    }
  }

  /**
   * Set connection metadata
   */
  setMetadata(socketId: string, key: string, value: any): void {
    const connection = this.connections.get(socketId);
    if (connection) {
      connection.metadata.set(key, value);
    }
  }

  /**
   * Get connection metadata
   */
  getMetadata(socketId: string, key: string): any {
    const connection = this.connections.get(socketId);
    return connection?.metadata.get(key);
  }

  /**
   * Check if connection is active
   */
  isConnectionActive(socketId: string): boolean {
    const connection = this.connections.get(socketId);
    if (!connection) return false;
    
    const now = Date.now();
    const lastActivity = connection.lastActivity.getTime();
    
    return (now - lastActivity) < this.config.connectionTimeoutMs;
  }

  /**
   * Get all active connections
   */
  getActiveConnections(): ConnectionInfo[] {
    return Array.from(this.connections.values())
      .filter(conn => this.isConnectionActive(conn.id));
  }

  /**
   * Clean up stale connections
   */
  cleanupStaleConnections(): number {
    const staleConnections: string[] = [];
    
    this.connections.forEach((conn, id) => {
      if (!this.isConnectionActive(id)) {
        staleConnections.push(id);
      }
    });
    
    staleConnections.forEach(id => {
      this.removeConnection(id);
    });
    
    if (staleConnections.length > 0) {
      this.logger.log(`Cleaned up ${staleConnections.length} stale connections`);
    }
    
    return staleConnections.length;
  }

  /**
   * Remove a connection completely
   */
  private removeConnection(socketId: string): void {
    this.connections.delete(socketId);
    this.clearReconnectTimer(socketId);
    this.stopHeartbeat(socketId);
  }

  /**
   * Start reconnection timer
   */
  private startReconnectTimer(socketId: string): void {
    const timer = setTimeout(() => {
      const connection = this.connections.get(socketId);
      
      if (connection && connection.reconnectCount >= this.config.maxReconnectAttempts) {
        this.logger.warn(`Max reconnect attempts reached for ${socketId}`);
        this.removeConnection(socketId);
      }
    }, this.config.reconnectTimeoutMs);
    
    this.reconnectTimers.set(socketId, timer);
  }

  /**
   * Clear reconnection timer
   */
  private clearReconnectTimer(socketId: string): void {
    const timer = this.reconnectTimers.get(socketId);
    if (timer) {
      clearTimeout(timer);
      this.reconnectTimers.delete(socketId);
    }
  }

  /**
   * Start heartbeat for connection
   */
  private startHeartbeat(socketId: string): void {
    const timer = setInterval(() => {
      const connection = this.connections.get(socketId);
      
      if (connection && connection.socket.connected) {
        connection.socket.emit('heartbeat', { timestamp: Date.now() });
      } else {
        this.stopHeartbeat(socketId);
      }
    }, this.config.heartbeatIntervalMs);
    
    this.heartbeatTimers.set(socketId, timer);
  }

  /**
   * Stop heartbeat for connection
   */
  private stopHeartbeat(socketId: string): void {
    const timer = this.heartbeatTimers.get(socketId);
    if (timer) {
      clearInterval(timer);
      this.heartbeatTimers.delete(socketId);
    }
  }

  /**
   * Get connection statistics
   */
  getConnectionStats(): any {
    const connections = Array.from(this.connections.values());
    const activeConnections = connections.filter(conn => this.isConnectionActive(conn.id));
    
    return {
      totalConnections: connections.length,
      activeConnections: activeConnections.length,
      averageReconnects: connections.length > 0
        ? connections.reduce((sum, conn) => sum + conn.reconnectCount, 0) / connections.length
        : 0,
      oldestConnection: connections.length > 0
        ? Math.min(...connections.map(conn => conn.connectedAt.getTime()))
        : null,
      config: this.config,
    };
  }

  /**
   * Update configuration
   */
  updateConfig(config: Partial<ConnectionConfig>): void {
    this.config = { ...this.config, ...config };
    this.logger.log(`Connection config updated: ${JSON.stringify(this.config)}`);
  }
}
