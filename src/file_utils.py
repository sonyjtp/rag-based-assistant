import os
from langchain_community.document_loaders import TextLoader
from pathlib import Path

import yaml


def load_documents(folder: str, file_extns: str | tuple[str, ...] = ".txt") -> list[dict[str, str]]:
    """
    Load documents from text files using LangChain's TextLoader.

    Uses LangChain's TextLoader to parse documents from the specified folder,
    then extracts metadata including title from the first line of content.

    Args:
        folder: Path to the folder containing documents
        file_extns: File extension(s) to load (default: ".txt")

    Returns:
        List of document dictionaries with the following keys:
        - 'content': The document text content
        - 'filename': The source filename
        - 'title': The document title (extracted from first non-empty line)
    """
    documents: list[dict[str, str]] = []
    for filename in os.listdir(folder):
        if not filename.endswith(file_extns):
            print(f"Skipping {filename} as it does not match the specified extensions.")
            continue
        try:
            for doc in TextLoader(os.path.join(folder, filename), encoding="utf-8").load():
                # Extract title from first non-empty line
                content_lines = doc.page_content.strip().split('\n')
                title = next((line.strip() for line in content_lines if line.strip()), filename)

                documents.append({
                    'filename': filename,
                    'title': title,
                    'content': doc.page_content
                })
        except IOError as e:
            print(f"Error loading {filename}: {e}")
    return documents


def load_yaml(file_path: str | Path) -> dict:
    """Loads a YAML configuration file.

    Args:
        file_path: Path to the YAML file.

    Returns:
        Parsed YAML content as a dictionary.

    Raises:
        FileNotFoundError: If the file does not exist.
        yaml.YAMLError: If there's an error parsing YAML.
        IOError: If there's an error reading the file.
    """
    file_path = Path(file_path)

    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # Read and parse the YAML file
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return yaml.safe_load(file)
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing YAML file: {e}") from e
    except IOError as e:
        raise IOError(f"Error reading YAML file: {e}") from e
