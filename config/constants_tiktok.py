import os
from config.root import ROOT_DIR

# -- RUTAS DE SCRAPING
# Directorio de datos recolectados de TikTok
TIKTOK_SCRAPED_DATA_DIR = os.path.join(ROOT_DIR, "data", "scraped", "raw")
# Ruta del archivo con los datos recolectados y fusionados de TikTok
TIKTOK_MERGED_SCRAPED_DATA_PATH = os.path.join(ROOT_DIR, "data", "scraped", "scraped_merged", "tiktok_scraped_merged_data.csv")

# .. RUTAS DE DESCARGA Y TRANSCRIPCION
# Directorio donde serán descargados los videos
TIKTOK_DOWNLOAD_VIDEO_DIR = os.path.join(ROOT_DIR, "data", "scraped", "download_video")
# Ruta del archivo con las transcripciones
TIKTOK_TRANSCRIBED_VIDEOS_PATH = os.path.join(ROOT_DIR, "data", "scraped", "transcribed", "transcribed.csv")

# yt_dlp config
YDL_OPTS = {
    'format': 'bestaudio/bestvideo/best',  # Orden de preferencia separado por '/'
    'outtmpl': f'{TIKTOK_DOWNLOAD_VIDEO_DIR}/%(id)s.%(ext)s' # Formato de archivos descargados 
}

# -- RUTAS DE LIMPIEZA
TIKTOK_PRE_TRANSLATED_SENTENCES = os.path.join(ROOT_DIR, 'data','processed','tiktok','tiktok_pre_translated_sentences.csv')
TIKTOK_TRANSLATED_SENTENCES = os.path.join(ROOT_DIR, "data","processed","tiktok","tiktok_translated_Sentences.csv")
TIKTOK_TRANSLATED_TEXT = os.path.join(ROOT_DIR, "data","processed","tiktok","tiktok_translated_text.csv")

TIKTOK_PRE_SENTIMENT_TEXT = os.path.join(ROOT_DIR, "data","processed","tiktok","tiktok_pre_sentiment_text.csv")
TIKTOK_SENTIMENT_TEXT = os.path.join(ROOT_DIR, "data","processed","tiktok","tiktok_sentiment_text.csv")

TIKTOK_PRE_SENTIMENT_SENTENCES = os.path.join(ROOT_DIR, "data","processed","tiktok","tiktok_pre_sentiment_sentences.csv")
TIKTOK_SENTIMENT_SENTENCES = os.path.join(ROOT_DIR, "data","processed","tiktok","tiktok_sentiment_sentences.csv")

TIKTOK_DATASET_TEXT = os.path.join(ROOT_DIR, "data","clean","tiktok","tiktok_dataset_text.csv")
TIKTOK_DATASET_SENTENCES = os.path.join(ROOT_DIR, "data","clean","tiktok","tiktok_dataset_sentences.csv")

