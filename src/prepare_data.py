"""
Script to prepare data for the housing price prediction model.
"""
import logging
import pathlib
import shutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def prepare_data():
    try:
        # Create data directory
        data_dir = pathlib.Path('data')
        data_dir.mkdir(exist_ok=True)
        logger.info(f"Created or verified data directory: {data_dir}")

        # Source and destination paths
        src = pathlib.Path('tests/test_data/test_housing.csv')
        dst = data_dir / 'housing.csv'

        # Ensure source exists
        if not src.exists():
            raise FileNotFoundError(f"Source file not found: {src}")

        # Remove destination if it exists
        if dst.exists():
            dst.unlink()
            logger.info(f"Removed existing file: {dst}")

        # Copy file
        logger.info(f'Copying {src} to {dst}')
        shutil.copy2(str(src), str(dst))
        
        # Verify copy
        if not dst.exists():
            raise FileNotFoundError(f"Failed to create: {dst}")
        
        logger.info(f"Successfully copied data file. Size: {dst.stat().st_size} bytes")
    except Exception as e:
        logger.error(f"Error preparing data: {str(e)}")
        raise

if __name__ == '__main__':
    prepare_data()
