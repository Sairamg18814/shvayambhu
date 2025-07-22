#!/usr/bin/env python3
"""Download and prepare seed data for Shvayambhu bootstrap training.

This script handles acquisition of minimal, high-quality seed data
from various public sources while respecting licenses and privacy.
All data is processed locally without external API calls.
"""

import os
import json
import hashlib
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import argparse
from dataclasses import dataclass
from datetime import datetime
import requests
from tqdm import tqdm
import tarfile
import zipfile


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class DataSource:
    """Represents a data source for seed data."""
    name: str
    url: str
    size_mb: float
    languages: List[str]
    license: str
    description: str
    format: str  # 'text', 'json', 'jsonl', 'csv'
    processing_fn: Optional[str] = None  # Function name for custom processing


# Define seed data sources (all public domain or permissively licensed)
SEED_DATA_SOURCES = [
    DataSource(
        name="wikipedia_sample",
        url="https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-abstract.xml.gz",
        size_mb=800,
        languages=["en"],
        license="CC-BY-SA",
        description="Wikipedia abstracts for general knowledge",
        format="xml"
    ),
    DataSource(
        name="project_gutenberg",
        url="https://www.gutenberg.org/cache/epub/feeds/rdf-files.tar.zip",
        size_mb=2000,
        languages=["en", "fr", "de", "es", "it"],
        license="Public Domain",
        description="Classic literature from Project Gutenberg",
        format="text"
    ),
    DataSource(
        name="openwebtext_sample",
        url="https://github.com/jcpeterson/openwebtext/raw/master/sample.tar",
        size_mb=500,
        languages=["en"],
        license="MIT",
        description="High-quality web text sample",
        format="text"
    ),
    DataSource(
        name="pile_sample",
        url="https://the-eye.eu/public/AI/pile/train/00.jsonl.zst",
        size_mb=5000,
        languages=["en"],
        license="Mixed Open Source",
        description="Diverse text from The Pile dataset",
        format="jsonl"
    ),
    DataSource(
        name="c4_sample",
        url="https://huggingface.co/datasets/allenai/c4/resolve/main/en/c4-train.00000-of-01024.json.gz",
        size_mb=300,
        languages=["en"],
        license="Apache 2.0",
        description="Common Crawl cleaned data",
        format="json"
    ),
    DataSource(
        name="code_samples",
        url="https://github.com/github/gitignore/archive/main.zip",
        size_mb=10,
        languages=["code"],
        license="CC0",
        description="Programming patterns from gitignore templates",
        format="text",
        processing_fn="process_code_samples"
    )
]


class SeedDataDownloader:
    """Handles downloading and processing of seed data."""
    
    def __init__(self, data_dir: str = "data/seed", cache_dir: str = "data/cache"):
        """Initialize downloader.
        
        Args:
            data_dir: Directory to store processed seed data
            cache_dir: Directory to cache downloads
        """
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Track download progress
        self.manifest = {
            "sources": [],
            "total_size_mb": 0,
            "languages": set(),
            "download_date": datetime.now().isoformat(),
            "version": "1.0"
        }
    
    def download_file(self, url: str, filename: str) -> Path:
        """Download file with progress bar.
        
        Args:
            url: URL to download from
            filename: Local filename
            
        Returns:
            Path to downloaded file
        """
        filepath = self.cache_dir / filename
        
        # Check if already downloaded
        if filepath.exists():
            logger.info(f"Using cached file: {filename}")
            return filepath
        
        logger.info(f"Downloading {filename} from {url}")
        
        # Download with progress bar
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(filepath, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=filename) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
        
        return filepath
    
    def extract_archive(self, filepath: Path, extract_to: Path) -> List[Path]:
        """Extract compressed archive.
        
        Args:
            filepath: Path to archive
            extract_to: Directory to extract to
            
        Returns:
            List of extracted file paths
        """
        extract_to.mkdir(parents=True, exist_ok=True)
        extracted_files = []
        
        if filepath.suffix == '.gz':
            import gzip
            output_path = extract_to / filepath.stem
            with gzip.open(filepath, 'rb') as f_in:
                with open(output_path, 'wb') as f_out:
                    f_out.write(f_in.read())
            extracted_files.append(output_path)
            
        elif filepath.suffix == '.tar':
            with tarfile.open(filepath, 'r') as tar:
                tar.extractall(extract_to)
                extracted_files.extend([extract_to / m.name for m in tar.getmembers()])
                
        elif filepath.suffix == '.zip':
            with zipfile.ZipFile(filepath, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
                extracted_files.extend([extract_to / f for f in zip_ref.namelist()])
                
        elif filepath.suffix == '.zst':
            import zstandard as zstd
            output_path = extract_to / filepath.stem
            with open(filepath, 'rb') as f_in:
                dctx = zstd.ZstdDecompressor()
                with open(output_path, 'wb') as f_out:
                    f_out.write(dctx.decompress(f_in.read()))
            extracted_files.append(output_path)
        
        return extracted_files
    
    def process_text_files(self, files: List[Path], output_dir: Path) -> Dict[str, Any]:
        """Process text files into training format.
        
        Args:
            files: List of text files
            output_dir: Output directory
            
        Returns:
            Processing statistics
        """
        stats = {
            "num_files": len(files),
            "total_bytes": 0,
            "num_documents": 0
        }
        
        output_file = output_dir / "text_data.jsonl"
        
        with open(output_file, 'w', encoding='utf-8') as out:
            for file_path in files:
                if file_path.is_file() and file_path.suffix in ['.txt', '.text', '']:
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            
                        # Clean and validate
                        content = content.strip()
                        if len(content) > 100:  # Minimum document length
                            doc = {
                                "text": content,
                                "source": file_path.name,
                                "length": len(content)
                            }
                            out.write(json.dumps(doc) + '\n')
                            stats["num_documents"] += 1
                            stats["total_bytes"] += len(content.encode('utf-8'))
                            
                    except Exception as e:
                        logger.warning(f"Error processing {file_path}: {e}")
        
        return stats
    
    def process_json_files(self, files: List[Path], output_dir: Path) -> Dict[str, Any]:
        """Process JSON/JSONL files into training format.
        
        Args:
            files: List of JSON files
            output_dir: Output directory
            
        Returns:
            Processing statistics
        """
        stats = {
            "num_files": len(files),
            "total_bytes": 0,
            "num_documents": 0
        }
        
        output_file = output_dir / "json_data.jsonl"
        
        with open(output_file, 'w', encoding='utf-8') as out:
            for file_path in files:
                if file_path.suffix in ['.json', '.jsonl']:
                    try:
                        if file_path.suffix == '.jsonl':
                            # Process line by line
                            with open(file_path, 'r', encoding='utf-8') as f:
                                for line in f:
                                    doc = json.loads(line.strip())
                                    if 'text' in doc and len(doc['text']) > 100:
                                        out.write(json.dumps(doc) + '\n')
                                        stats["num_documents"] += 1
                                        stats["total_bytes"] += len(doc['text'].encode('utf-8'))
                        else:
                            # Process as single JSON
                            with open(file_path, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                                
                            # Extract text fields
                            if isinstance(data, list):
                                for item in data:
                                    if isinstance(item, dict) and 'text' in item:
                                        if len(item['text']) > 100:
                                            out.write(json.dumps(item) + '\n')
                                            stats["num_documents"] += 1
                                            stats["total_bytes"] += len(item['text'].encode('utf-8'))
                                            
                    except Exception as e:
                        logger.warning(f"Error processing {file_path}: {e}")
        
        return stats
    
    def process_code_samples(self, files: List[Path], output_dir: Path) -> Dict[str, Any]:
        """Process code samples into training format.
        
        Args:
            files: List of code files
            output_dir: Output directory
            
        Returns:
            Processing statistics
        """
        stats = {
            "num_files": len(files),
            "total_bytes": 0,
            "num_documents": 0
        }
        
        output_file = output_dir / "code_data.jsonl"
        code_extensions = {'.py', '.js', '.java', '.cpp', '.c', '.rs', '.go', '.rb', '.php'}
        
        with open(output_file, 'w', encoding='utf-8') as out:
            for file_path in files:
                if file_path.is_file() and file_path.suffix in code_extensions:
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            
                        if len(content) > 50:  # Minimum code length
                            doc = {
                                "text": content,
                                "source": "code",
                                "language": file_path.suffix[1:],
                                "length": len(content)
                            }
                            out.write(json.dumps(doc) + '\n')
                            stats["num_documents"] += 1
                            stats["total_bytes"] += len(content.encode('utf-8'))
                            
                    except Exception as e:
                        logger.warning(f"Error processing {file_path}: {e}")
        
        return stats
    
    def download_source(self, source: DataSource) -> bool:
        """Download and process a single data source.
        
        Args:
            source: DataSource to download
            
        Returns:
            Success status
        """
        try:
            # Download file
            filename = source.url.split('/')[-1]
            downloaded_file = self.download_file(source.url, filename)
            
            # Create source directory
            source_dir = self.data_dir / source.name
            source_dir.mkdir(exist_ok=True)
            
            # Extract if needed
            if downloaded_file.suffix in ['.gz', '.tar', '.zip', '.zst']:
                extract_dir = self.cache_dir / f"{source.name}_extracted"
                extracted_files = self.extract_archive(downloaded_file, extract_dir)
            else:
                extracted_files = [downloaded_file]
            
            # Process based on format
            if source.processing_fn:
                # Custom processing function
                process_fn = getattr(self, source.processing_fn)
                stats = process_fn(extracted_files, source_dir)
            elif source.format == "text":
                stats = self.process_text_files(extracted_files, source_dir)
            elif source.format in ["json", "jsonl"]:
                stats = self.process_json_files(extracted_files, source_dir)
            else:
                logger.warning(f"Unsupported format: {source.format}")
                return False
            
            # Update manifest
            self.manifest["sources"].append({
                "name": source.name,
                "stats": stats,
                "license": source.license,
                "languages": source.languages
            })
            self.manifest["total_size_mb"] += source.size_mb
            self.manifest["languages"].update(source.languages)
            
            logger.info(f"Successfully processed {source.name}: {stats}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to download {source.name}: {e}")
            return False
    
    def download_all(self, sources: Optional[List[str]] = None):
        """Download all or specified sources.
        
        Args:
            sources: List of source names to download (None for all)
        """
        # Filter sources if specified
        if sources:
            download_sources = [s for s in SEED_DATA_SOURCES if s.name in sources]
        else:
            download_sources = SEED_DATA_SOURCES
        
        logger.info(f"Downloading {len(download_sources)} sources...")
        
        # Download each source
        success_count = 0
        for source in download_sources:
            if self.download_source(source):
                success_count += 1
        
        # Convert set to list for JSON serialization
        self.manifest["languages"] = list(self.manifest["languages"])
        
        # Save manifest
        manifest_path = self.data_dir / "manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(self.manifest, f, indent=2)
        
        logger.info(f"Download complete: {success_count}/{len(download_sources)} sources")
        logger.info(f"Manifest saved to: {manifest_path}")
    
    def verify_data(self) -> bool:
        """Verify downloaded data integrity.
        
        Returns:
            Verification status
        """
        manifest_path = self.data_dir / "manifest.json"
        if not manifest_path.exists():
            logger.error("No manifest found")
            return False
        
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        
        logger.info("Verifying data integrity...")
        all_valid = True
        
        for source in manifest["sources"]:
            source_dir = self.data_dir / source["name"]
            if not source_dir.exists():
                logger.error(f"Missing source directory: {source['name']}")
                all_valid = False
                continue
            
            # Check for expected files
            jsonl_files = list(source_dir.glob("*.jsonl"))
            if not jsonl_files:
                logger.error(f"No data files found in: {source['name']}")
                all_valid = False
            else:
                logger.info(f" {source['name']}: {len(jsonl_files)} files")
        
        return all_valid


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Download seed data for Shvayambhu training"
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        help="Specific sources to download (default: all)"
    )
    parser.add_argument(
        "--data-dir",
        default="data/seed",
        help="Directory to store seed data"
    )
    parser.add_argument(
        "--cache-dir",
        default="data/cache",
        help="Directory to cache downloads"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify existing data instead of downloading"
    )
    
    args = parser.parse_args()
    
    # Create downloader
    downloader = SeedDataDownloader(
        data_dir=args.data_dir,
        cache_dir=args.cache_dir
    )
    
    if args.verify:
        # Verify mode
        if downloader.verify_data():
            logger.info(" All data verified successfully")
        else:
            logger.error(" Data verification failed")
            exit(1)
    else:
        # Download mode
        downloader.download_all(sources=args.sources)
        
        # Verify after download
        if not downloader.verify_data():
            logger.warning("Some data verification checks failed")


if __name__ == "__main__":
    main()