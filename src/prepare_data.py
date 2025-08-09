"""
Script to prepare data for the housing price prediction model.
"""
import pathlib
import shutil

def prepare_data():
    # Create data directory
    data_dir = pathlib.Path('data')
    data_dir.mkdir(exist_ok=True)

    # Source and destination paths
    src = pathlib.Path('tests/test_data/test_housing.csv')
    dst = data_dir / 'housing.csv'

    # Copy file
    print(f'Copying {src} to {dst}')
    shutil.copy(str(src), str(dst))

if __name__ == '__main__':
    prepare_data()
