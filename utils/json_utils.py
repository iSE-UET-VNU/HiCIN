import json
from pathlib import Path
from typing import Union, Dict, List, Any

class JSONFileReader:
    @staticmethod
    def read(file_path: Union[str, Path]) -> Union[Dict[str, Any], List[Any]]:
        path = Path(file_path)

        if not path.is_file():
            raise FileNotFoundError(f"File not found at path: {path.absolute()}")

        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to decode JSON from {path.name}: {e}")
        except Exception as e:
            raise RuntimeError(f"An unexpected error occurred while reading {path.name}: {e}")

class JSONFileWriter:
    @staticmethod
    def write(data: Union[Dict[str, Any], List[Any]], 
              file_path: Union[str, Path], 
              indent: int = 4,
              ensure_ascii: bool = False) -> None:
        
        path = Path(file_path)

        # Create parent directories if they don't exist
        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii)
        except TypeError as e:
            raise TypeError(f"Data provided is not JSON serializable: {e}")
        except Exception as e:
            raise IOError(f"An error occurred while writing to {path.name}: {e}")

if __name__ == "__main__":
    try:
        data = JSONFileReader.read('configs/dataset_config.json')
        print(data)
        pass
    except Exception as e:
        print(f"Error: {e}")
