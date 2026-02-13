"""MongoDB database connection and utilities."""
from typing import Optional
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from pymongo.database import Database
import logging
import ssl

from src.config.settings import settings

logger = logging.getLogger(__name__)


class MongoDatabase:
    """MongoDB connection manager."""
    
    _client: Optional[MongoClient] = None
    _db: Optional[Database] = None
    
    @classmethod
    def connect(cls) -> Database:
        """Establish MongoDB connection using ServerApi."""
        if cls._client is None:
            try:
                # Create client with ServerApi (MongoDB recommended approach)
                cls._client = MongoClient(
                    settings.MONGO_URI,
                    server_api=ServerApi('1'),
                    serverSelectionTimeoutMS=10000,
                    connectTimeoutMS=10000,
                )
                
                # Test connection with ping
                cls._client.admin.command('ping')
                cls._db = cls._client[settings.MONGO_DB_NAME]
                logger.info(f"✓ Connected to MongoDB database: {settings.MONGO_DB_NAME}")
            except Exception as e:
                logger.error(f"✗ Failed to connect to MongoDB: {e}")
                cls._client = None
                cls._db = None
                raise
        return cls._db
    
    @classmethod
    def disconnect(cls) -> None:
        """Close MongoDB connection."""
        if cls._client is not None:
            cls._client.close()
            cls._client = None
            cls._db = None
            logger.info("Disconnected from MongoDB")
    
    @classmethod
    def get_db(cls) -> Database:
        """Get database instance."""
        if cls._db is None:
            cls.connect()
        return cls._db


def get_database() -> Database:
    """Dependency for getting database instance."""
    return MongoDatabase.get_db()
