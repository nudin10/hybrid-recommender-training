import redis
import os
import redis.exceptions
from tools.logger import global_logger
from tools.errors import RedisCredentialsNotFound

class Redis_Client:
    def __init__(self) -> None:
        REDIS_HOST = os.getenv("REDIS_HOST")
        REDIS_PORT = os.getenv("REDIS_PORT")
        REDIS_DB = os.getenv("REDIS_DB")
        
        if not REDIS_HOST or not REDIS_PORT or not REDIS_DB:
            global_logger.error("Redis credentials not found.")
            raise RedisCredentialsNotFound
        
        self.host = REDIS_HOST
        self.port = int(REDIS_PORT)
        self.db = int(REDIS_DB)

        try:
            self.client = redis.StrictRedis(host=self.host, port=self.port, db=self.db, decode_responses=True)
            self.client.ping()
            global_logger.info("Successfully connected to redis")
        except redis.exceptions.ConnectionError as e:
            global_logger.error(f"Couldn't connect to Redis: {e}")
            raise
        except Exception:
            raise

    def get_pipeline(self):
        return self.client.pipeline()

    def hget(self, name: str, key: str):
        self.client.hget(name=name, key=key)

    def hset(self, name: str, key: str, value):
        self.client.hset(name=name, key=key, value=value)

redis_client: Redis_Client | None =None

def init_redis():
    global redis_client
    if redis_client is None:
        redis_client = Redis_Client()
    
    return redis_client

def get_redis() -> Redis_Client:
    if redis_client is None:
        raise RuntimeError("Redis client not initialised. Call init_redis() to initialise.")
    return redis_client