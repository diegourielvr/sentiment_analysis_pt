import re
import unicodedata
import pytz
import emoji
import pandas as pd 
from datetime import datetime, timedelta

def replace_nbsp(texto: str):
    """Reemplazar el símbolo &nbsp; por un espacio regular
    """
    
    return re.sub(r'[\xa0]', ' ', str(texto))

def replace_whitespaces(text: str):
    """Reemplazar espacios consecutivos, al inicio, final o intermedios por uno solo
    """
    
    return re.sub(r'\s+', ' ', text).strip()

def replace_quotes(text:str):
    """Reemplazar comillas dobles por comillas simples
    """

    texto = re.sub(r'"', "'", text)
    texto = re.sub(r'[“”]', "'", texto)  # Para comillas curvas
    texto = re.sub(r'&quot;', "'", texto)  # Para la entidad HTML
    return texto

def replace_url(text:str, value='url'):
    """Reemplazar una url por otro valor
    """
    
    return re.sub(r'https?://\S+|www\.\S+', value, text)

def reduce_repeated_punct(text):
    """Reducir signos de puntuación repetidos
    """
    
    # Eliminar signos de puntuación repetidos
    return re.sub(r'([^\w\s])\1+', r'\1', text)

def remove_zwj(text):
    """Eliminar simbolo ZWJ (Zero Width Joiner)
    """
    
    return re.sub(r'\u200D', '', text)

def emoji_to_text(text: str, lang='es'):
    """Convertir emojis visibles a texto
    """

    return emoji.demojize(text, language=lang)

def normalize_text(text: str):
    """Convertir texto en negritas, cursivas a su forma normal y
    Unír unir acentos
    """

    # Eliminar negritas, cursivas, separar acentos, etc.
    text = unicodedata.normalize("NFKD", str(text))

    # Unir acentos para evitar problemas
    text = unicodedata.normalize("NFC", text)
    return text

def drop_non_letters_only_rows(df, col):
    """Elimina filas donde la columna especificada solo contenga valores diferentes
    a letras, números y guines bajos.
    """

    return df[~df[col].str.fullmatch(r'\W+', na=False)]

def drop_blank_or_nan_or_duplicated(df, col):
    """Eliminar cadenas vacías, espacios, tabulaciones, NA o valores duplicados
    """

    new_df = df[~df[col].str.match(r'^\s*$') & df[col].notna()]
    return new_df.drop_duplicates(subset=col,keep='first').reset_index(drop=True)

def text_to_emoji(text: str, lang='es'):
    return emoji.emojize(text,language=lang)

def remove_punctuation(text: str):
    return re.sub(r'[!"#&\'*+,-./<=>?@[\\]^`{|}~]', '', text)

def clean_text(text: str):
    """Mantienen letras en español, acentos y algunos signos de puntuación.
    Elimina emojis
    """

    # Expresión regular para capturar lo que queremos mantener:
    # - Letras mayúsculas y minúsculas: a-zA-Z
    # - Números: 0-9
    # - Signos de puntuación: !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
    # - Emojis en formato texto: :_ejemplo_:
    pattern = re.compile(r'(:[_a-zA-Z0-9]+_:|[a-zA-ZáéíóúüÁÉÍÓÚÜñÑ0-9!"$%&\'(),/:;?[\]_{}]+)', re.UNICODE)


    # Extraer y unir los elementos permitidos
    filtered_text = ' '.join(pattern.findall(text))
    
    return filtered_text
    
def reduce_repeated_words(text):
    """
    Reduce palabras repetidas como 'jajajaja', 'abcabc', 'holahola' a su versión más corta: 'jaja', 'abc', 'hola'.
    """
    pattern = re.compile(r'\b(\w{2,})\1+\b', re.IGNORECASE)  # Detecta repeticiones de cualquier palabra de 2+ caracteres
    return pattern.sub(r'\1', text)

def reemplazar_nbsp(texto):
    """Reemplaza el símbolo &nbsp; o '\xa0' por un espacio en blanco
    """
    #return string.replace("\xa0", " ")
    return unicodedata.normalize("NFKD", str(texto))

def corregir_fecha_tiktok(fecha, fecha_referencia):
    """Tranformar fechas a un formato común yyyy-mm-dd
    """
    fecha = reemplazar_nbsp(fecha)
    zona_horaria = pytz.timezone('America/Mexico_City')
    #fecha_actual = datetime.now(tz=pytz.utc).astimezone(zona_horaria)
    fecha_actual = pd.to_datetime(fecha_referencia)
    unidad = ""
    if fecha.startswith('Hace'):
        partes = fecha.split(" ") # <Hace> <1> <s | min | h | dia(s) | semana(s)>
        cantidad = partes[1]
        unidad = partes[2] # <s | min | h | dias(s) | semana(s)>

    if unidad == "s":
        return (fecha_actual - timedelta(seconds=int(cantidad))).date()
    elif unidad == "min":
        return (fecha_actual - timedelta(minutes=int(cantidad))).date()
    elif unidad == "h":
        return (fecha_actual - timedelta(hours=int(cantidad))).date()
    elif unidad == reemplazar_nbsp("día(s)"): # La tílde es un caracter especial. Significa que esta tilde í y esta otra í son diferentes (internamente)
        return (fecha_actual - timedelta(days=int(cantidad))).date()
    elif unidad == "semana(s)":
        return (fecha_actual - timedelta(weeks=int(cantidad))).date()

    partes = fecha.split("-")
    if len(partes) == 2: # 10-19 -> mm-dd
        mes, dia = map(int, partes)
        return datetime(fecha_actual.year, mes, dia, tzinfo=zona_horaria).date()
    elif len(partes) == 3: # 2023-10-19 -> yyyy-mm-dd
        partes[2] = partes[2].split(" ")[0]
        anio, mes, dia = map(int, partes)
        return datetime(anio, mes, dia, tzinfo=zona_horaria).date()

    return pd.NA
