import polars as pl
from src.cache.redis_client import Redis_Client
from pathlib import Path
from typing import Generator, List, Dict, Any
import json
from src.data_loader.embeddings_consolidator import mean_pooling_consolidator
import numpy as np
import torch
from src.errors.redis import RedisValueInvalid


def load_into_redis_by_batch(r: Redis_Client, model_key:str, file_path: str, batch_num=1000):
    '''
    Loads data into redis from an ndjson file efficiently.

    Args:
        file_path (str): The path to the ndjson file.
        batch_num (str): Number of data per batch to be loaded into Redis

    Returns:
        None
    '''

    try:
        for batch in batch_read_ndjson(file_path=Path(file_path), batch_num=batch_num):
            pipeline = r.get_pipeline()

            for row in batch:
                item_id = str(row["item_id"])
                existing_embeddings_bytes: list[float] | None = r.hget(
                    name=model_key,
                    key=item_id
                ) 

                embeddings_to_set: list[float] = []
                new_embeddings: list[float] = row["embedding"]

                if existing_embeddings_bytes:
                    try:
                        existing_embeddings = json.loads(existing_embeddings_bytes)
                    except json.JSONDecodeError as e:
                        raise
                    except Exception as e:
                        raise

                    if not isinstance(existing_embeddings, list):
                        raise RedisValueInvalid
                    
                    try:
                        existing_embeddings_tensor = torch.from_numpy(np.array(existing_embeddings))
                        new_embeddings_tensor = torch.from_numpy(np.array(new_embeddings))
                        # consolidate embeddings if exist
                        pooled_embeddings = mean_pooling_consolidator([
                            new_embeddings_tensor,
                            existing_embeddings_tensor
                        ])

                        embeddings_to_set = pooled_embeddings.tolist()
                    except:
                        raise
                else:
                    embeddings_to_set = new_embeddings

                try:
                    value_to_set = json.dumps(embeddings_to_set)

                    pipeline.hset(
                        name=model_key,
                        key=item_id,
                        value=value_to_set
                    )

                except Exception as e:
                    raise
            
            pipeline.execute(raise_on_error=True)       
    except:
        raise

def batch_read_ndjson(file_path: Path, batch_num: int) -> Generator[List[Dict[str, Any]], None, None]:
    current_batch: List[Dict[str, Any]] = []
    line_count = 0

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line_count += 1
                stripped = line.strip()
                if not stripped:
                    continue

                try:
                    datum = json.loads(stripped)
                    current_batch.append(datum)

                    if len(current_batch) >= batch_num:
                        yield current_batch
                        current_batch = []

                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping invalid JSON on line {line_count}: {e}")
                except Exception as e:
                    print(f"Warning: An unexpected error occurred processing line {line_count}: {e}")
                
            if current_batch:
                yield current_batch
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        raise
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        raise
