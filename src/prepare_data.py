"""
Script to prepare data for the housing price prediction model.
"""
import logging
import pathlib
import shutil
import pandas as pd
from schema import HousingDataBatch, HousingData

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validate_housing_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate housing data using Pydantic models.
    
    Args:
        df: Input DataFrame containing housing data
    
    Returns:
        Validated DataFrame
    
    Raises:
        ValueError: If data validation fails
    """
    try:
        # Convert DataFrame to list of dictionaries
        records = df.to_dict('records')
        
        # Validate each record using Pydantic
        validated_records = []
        for record in records:
            try:
                validated = HousingData(**record)
                validated_records.append(validated.dict())
            except Exception as e:
                logger.error(f"Validation error in record: {record}")
                logger.error(f"Error details: {str(e)}")
                raise
        
        # Validate entire batch
        HousingDataBatch(data=[HousingData(**r) for r in records])
        
        logger.info(f"Successfully validated {len(validated_records)} records")
        return pd.DataFrame(validated_records)
    
    except Exception as e:
        logger.error("Data validation failed")
        logger.error(str(e))
        raise

def prepare_data():
    """
    Prepare and validate housing data for model training.
    """
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

        # Read and validate data
        logger.info(f"Reading data from {src}")
        df = pd.read_csv(src)
        logger.info(f"Loaded {len(df)} records")

        # Validate data
        logger.info("Validating data...")
        validated_df = validate_housing_data(df)
        
        # Save validated data
        logger.info(f"Saving validated data to {dst}")
        validated_df.to_csv(dst, index=False)
        
        # Verify save
        if not dst.exists():
            raise FileNotFoundError(f"Failed to create: {dst}")
        
        logger.info(f"Successfully saved validated data. Size: {dst.stat().st_size} bytes")
    except Exception as e:
        logger.error(f"Error preparing data: {str(e)}")
        raise

if __name__ == '__main__':
    prepare_data()
