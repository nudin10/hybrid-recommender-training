from typing import Any
from src.tools.common import get_project_root
from pathlib import Path
import csv

class Result:
    def __init__(self, name:str, **kwargs) -> None:
        self.name = name
        self.mapped_res = {}

        default_store_path = get_project_root() / "results"
        self.store_path: Path = kwargs.get("store_path", default_store_path)
        try:
            assert isinstance(self.store_path, Path)
        except AssertionError:
            raise AssertionError("Result store path must be a Path object")

    def collect(self, result: dict|Any):
        if isinstance(result, dict):
            self.mapped_res = result
        else:
            try:
                self.mapped_res = {"value": result}
            except:
                raise ValueError("Collected result must be a dictionary or convertible to a simple key-value pair.")
    
    def store(self) -> Path:
        try:
            assert str.strip(self.name) != ""
        except AssertionError:
            raise AssertionError("Result name cannot be empty")
        
        file_name = f"{self.name}.csv"
        file_path = self.store_path / file_name

        with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['Item', 'Value']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for key, value in self.mapped_res.items():
                writer.writerow({'Item': key, 'Value': value})
        
        return file_path

def analyse():
    pass

def visualise():
    pass
