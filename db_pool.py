"""
Database Connection Pool for Asmeranda
Provides connection reuse to improve performance
"""

import sqlite3
from threading import Lock
from contextlib import contextmanager

class ConnectionPool:
    """
    Simple connection pool for SQLite
    Note: For production with high concurrency, consider using SQLAlchemy with proper pooling
    """
    
    def __init__(self, db_path, pool_size=5):
        self.db_path = db_path
        self.pool_size = pool_size
        self._pool = []
        self._lock = Lock()
        self._initialized = False
        
    def initialize(self):
        """Initialize the pool with connections"""
        if not self._initialized:
            with self._lock:
                if not self._initialized:
                    for _ in range(self.pool_size):
                        conn = sqlite3.connect(self.db_path, check_same_thread=False)
                        conn.row_factory = sqlite3.Row
                        self._pool.append(conn)
                    self._initialized = True
    
    @contextmanager
    def get_connection(self):
        """Get a connection from the pool (context manager)"""
        self.initialize()
        conn = None
        try:
            with self._lock:
                if self._pool:
                    conn = self._pool.pop()
                else:
                    # Pool exhausted, create new connection
                    conn = sqlite3.connect(self.db_path, check_same_thread=False)
                    conn.row_factory = sqlite3.Row
            
            yield conn
            
        finally:
            if conn:
                with self._lock:
                    if len(self._pool) < self.pool_size:
                        self._pool.append(conn)
                    else:
                        conn.close()
    
    def close_all(self):
        """Close all connections in the pool"""
        with self._lock:
            for conn in self._pool:
                conn.close()
            self._pool = []
            self._initialized = False


# Global pool instance (lazy initialization)
_pool_instance = None
_pool_lock = Lock()

def get_pool(db_path='users.db', pool_size=5):
    """Get or create the global connection pool"""
    global _pool_instance
    if _pool_instance is None:
        with _pool_lock:
            if _pool_instance is None:
                _pool_instance = ConnectionPool(db_path, pool_size)
    return _pool_instance

def reset_pool():
    """Reset the global pool (useful for testing)"""
    global _pool_instance
    with _pool_lock:
        if _pool_instance:
            _pool_instance.close_all()
        _pool_instance = None
