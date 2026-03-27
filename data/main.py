from src.file import load_csv, save_data
from src.processing_data import processing

if __name__ == "__main__":
    df = load_csv()
    df = processing(df)
    save_data(df)
    