import os
import shutil
from pathlib import Path
import logging
import filecmp
import json


class AudioFileOrganizer:
    def __init__(self, source_dir, destination_dir):
        self.source_dir = Path(source_dir)
        self.destination_dir = Path(destination_dir)
        self.setup_logging()
        self.load_categories()
        
        
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def load_categories(self):
        """Load category definitions from a JSON file in the current working directory."""
        categories_file_path = "categories.json" 

        if not Path(categories_file_path).exists():
            raise FileNotFoundError(f"Categories file not found: {categories_file_path}")

        try:
            with open(categories_file_path, 'r') as f:
                self.categories = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format in categories file: {str(e)}")
    
    def is_duplicate(self, source_path, dest_path):
        """Check if a file is already present in the destination using filecmp."""
        if not dest_path.exists():
            return False
            
        # Compare file sizes first (quick check)
        if source_path.stat().st_size != dest_path.stat().st_size:
            return False
            
        # Use filecmp for thorough comparison
        return filecmp.cmp(source_path, dest_path, shallow=False)
    
    # Alternative method using binary comparison if you prefer not to use filecmp
    def is_duplicate_binary(self, source_path, dest_path, chunk_size=8192):
        """Alternative duplicate check using direct binary comparison."""
        if not dest_path.exists():
            return False
            
        if source_path.stat().st_size != dest_path.stat().st_size:
            return False
            
        with open(source_path, 'rb') as sf, open(dest_path, 'rb') as df:
            while True:
                s_chunk = sf.read(chunk_size)
                d_chunk = df.read(chunk_size)
                
                if s_chunk != d_chunk:
                    return False
                    
                if not s_chunk:  # EOF reached
                    return True
    
    def find_unique_filename(self, dest_path):
        """Generate a unique filename if a file already exists."""
        if not dest_path.exists():
            return dest_path
            
        base = dest_path.stem
        extension = dest_path.suffix
        counter = 1
        
        while True:
            new_name = f"{base}_{counter}{extension}"
            new_path = dest_path.parent / new_name
            if not new_path.exists():
                return new_path
            counter += 1
    
    def create_directory_structure(self):
        """Create necessary directories in the destination path."""
        for category in self.categories:
            (self.destination_dir / category).mkdir(parents=True, exist_ok=True)
    
    def identify_category(self, filename):
        """Identify the category of the file based on keywords."""
        for category, keywords in self.categories.items():
            if isinstance(keywords, dict):  # Handle drums with subcategories
                for subcategory, sub_keywords in keywords.items():
                    if any(keyword in filename.lower() for keyword in sub_keywords):
                        return category, subcategory
            else:
                if any(keyword in filename.lower() for keyword in keywords):
                    return category, None
        return "others", None  # Default category if no match

    def organize_files(self):
        """Organize audio files into their respective categories."""
        supported_extensions = {'.wav', '.mp3', '.aif', '.aiff', '.flac', '.ogg'}
    
        progress_log = {
            'total_files': 0,
            'copied': 0,
            'skipped_duplicates': 0,
            'errors': 0
        }

        try:
            self.create_directory_structure()
            
            # Count total files first
            total_files = sum(1 for file in self.source_dir.rglob('*') 
                            if file.suffix.lower() in supported_extensions)
            progress_log['total_files'] = total_files
            
            # Process files
            for file_path in self.source_dir.rglob('*'):
                if file_path.suffix.lower() in supported_extensions:
                    try:
                        category, subcategory = self.identify_category(file_path.name)
                        
                        # Determine destination path
                        if category == 'drums' and subcategory:
                            dest_path = self.destination_dir / category / subcategory / file_path.name
                        else:
                            dest_path = self.destination_dir / category / file_path.name
                        
                        # Create parent directories
                        dest_path.parent.mkdir(parents=True, exist_ok=True)
                        
                        # Check if file is a duplicate
                        is_duplicate = self.is_duplicate_binary(file_path, dest_path)
                        
                        if is_duplicate:
                            logging.info(f"Skipped duplicate: {file_path.name}")
                            progress_log['skipped_duplicates'] += 1
                            continue
                        
                        # Handle name conflicts
                        if dest_path.exists():
                            dest_path = self.find_unique_filename(dest_path)  
                        
                        # Copy the file
                        shutil.copy2(file_path, dest_path)
                        progress_log['copied'] += 1

                        # Progress update
                        if total_files > 0:
                            progress = ((progress_log['copied'] + progress_log['skipped_duplicates']) / total_files * 100)
                            logging.info(f"Progress: {progress:.1f}% - Copied: {file_path.name} -> {dest_path}")
                    
                    except Exception as e:
                        logging.error(f"Error processing {file_path.name}: {str(e)}")
                        progress_log['errors'] += 1
            
            # Log final statistics
            logging.info("\nOrganization Complete:")
            logging.info(f"Total files processed: {total_files}")
            logging.info(f"Files copied: {progress_log['copied']}")
            logging.info(f"Duplicates skipped: {progress_log['skipped_duplicates']}")
            logging.info(f"Errors encountered: {progress_log['errors']}")
        
        except Exception as e:
            logging.error(f"Error organizing files: {str(e)}")
            progress_log['errors'] += 1
        
        # Always return progress_log, whether there was an error or not
        return progress_log

def main():
    try:
        source_directory = "splice_directory"
        destination_directory = "New_Splicefolder"
        
        organizer = AudioFileOrganizer(source_directory, destination_directory)
        progress_log = organizer.organize_files()
        
        print("\nFinal Statistics:")
        print(f"Total files: {progress_log['total_files']}")
        print(f"Successfully copied: {progress_log['copied']}")
        print(f"Duplicates skipped: {progress_log['skipped_duplicates']}")
        print(f"Errors: {progress_log['errors']}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Fatal error: {str(e)}")
