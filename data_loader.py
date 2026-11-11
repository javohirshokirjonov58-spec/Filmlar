"""
data_loader.py - KaggleHub orqali IMDb datasetini yuklash va tayyorlash moduli
"""
import os
import pandas as pd
import kagglehub

DATASET_NAME = "yasserh/imdb-movie-ratings-sentiment-analysis"

def get_dataset_path():
    """
    KaggleHub orqali datasetni yuklab oladi va yo'lini qaytaradi.
    """
    path = kagglehub.dataset_download(DATASET_NAME)
    return path

def load_data(limit=None):
    """
    Datasetni yuklaydi va kerakli ustunlarni formatlaydi.
    """
    dataset_path = get_dataset_path()
    csv_file = os.path.join(dataset_path, "movie.csv")
    
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"movie.csv fayli topilmadi: {dataset_path}")
        
    df = pd.read_csv(csv_file)
    
    # Qo'shimcha tahliliy ustunlar yaratamiz
    df['char_count'] = df['text'].str.len()
    df['word_count'] = df['text'].apply(lambda x: len(str(x).split()))
    df['sentiment_uz'] = df['label'].map({1: "Ijobiy (Positive)", 0: "Salbiy (Negative)"})
    
    if limit:
        return df.head(limit), dataset_path
        
    return df, dataset_path

if __name__ == "__main__":
    print("Dataset yuklanmoqda...")
    data, p = load_data(limit=5)
    print(f"Dataset manzili: {p}")
    print(f"Jami ustunlar: {data.columns.tolist()}")
    print(data.head())

