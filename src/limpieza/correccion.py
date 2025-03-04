import os
from symspellpy import SymSpell
from config.config_nlp import SYMSPELL_DICTIONARIES_PATHS
from tqdm import tqdm

def spell(texto, lang="es"):
    if lang not in SYMSPELL_DICTIONARIES_PATHS.keys():
        print(f"Idioma no soportado: {lang}")
        return None
    spell = SymSpell(
        max_dictionary_edit_distance=2, # Distancia de búsqueda
        prefix_length=7 # Prefijos de palabras
    )

    load = spell.load_dictionary(
        SYMSPELL_DICTIONARIES_PATHS[lang],
        term_index=0, # posicion donde se encuentran los terminos
        count_index=1, # posicion donde se encuentran las frecuencias
        encoding="utf-8"
    )
    if not load:
        print("[sympellpy]: No ha sido posible cargar el diccionario")
        return None

    sugerencias = spell.lookup_compound(
        texto,
        max_edit_distance=2,
        ignore_non_words=True, # ignorar caracteres como números
    )
    return sugerencias[0].term

def spell_pipe(documentos: list[str], lang="es"):
    """Corregir ortografia de una lista de textos

    :param documentos:  Lista de textos a corregir. Puede recibir df[column_name].tolist()
    :type documentos: list[str]
    :param lang: Idioma del texto a corregir
    :type lang: str
    :return: Devuelve la lista con los textos corregidos
    :rtype: list[str]
    """
    if lang not in SYMSPELL_DICTIONARIES_PATHS.keys():
        print(f"Idioma no soportado: {lang}")
        return None
    spell = SymSpell(
        max_dictionary_edit_distance=2, # Distancia de búsqueda
        prefix_length=7 # Prefijos de palabras
    )

    load = spell.load_dictionary(
        SYMSPELL_DICTIONARIES_PATHS[lang],
        term_index=0, # posicion donde se encuentran los terminos
        count_index=1, # posicion donde se encuentran las frecuencias
        encoding="utf-8"
    )
    if not load:
        print("[sympellpy]: No ha sido posiblecargar el diccionario")
        print(f"[symspellpy]: Ruta: {SYMSPELL_DICTIONARIES_PATHS[lang]}")
        return None
    print(f"Diccionario cargado: {SYMSPELL_DICTIONARIES_PATHS[lang]}")
    sugestions = []
    for doc in tqdm(documentos):
        sugestions.append(spell.lookup_compound(doc, max_edit_distance=2)[0].term)
        
    return sugestions

    # return list(map(
    #     lambda documento: spell.lookup_compound(documento, max_edit_distance=2)[0].term,
    #     documentos
    # ))

def valid_word(word, spell):
    suggestions = spell.lookup(word, max_edit_distance=2)
    # Si la sugerencia es la mismia palabra de entrada, entonces existe en el diccionario y devuelve Verdadero
    return suggestions and suggestions[0].term == word

def drop_words_pipe(documentos: list[str], lang, sep=" "):
    """Elimina las palabras del texto
    que no se encuentren en el diccionario del lenguaje especificado
    """
    if lang not in SYMSPELL_DICTIONARIES_PATHS.keys():
        print(f"Idioma no soportado: {lang}")
        return None
    spell = SymSpell(
        max_dictionary_edit_distance=2, # Distancia de búsqueda
        prefix_length=7 # Prefijos de palabras
    )

    load = spell.load_dictionary(
        SYMSPELL_DICTIONARIES_PATHS[lang],
        term_index=0, # posicion donde se encuentran los terminos
        count_index=1, # posicion donde se encuentran las frecuencias
        encoding="utf-8"
    )
    if not load:
        print("[sympellpy]: No ha sido posiblecargar el diccionario")
        print(f"[symspellpy]: Ruta: {SYMSPELL_DICTIONARIES_PATHS[lang]}")
        return None
    print(f"Diccionario cargado: {SYMSPELL_DICTIONARIES_PATHS[lang]}")

    # dividir el texto
    list_docs = list(map(lambda doc: doc.split(sep), documentos))
    return [
        list(filter(lambda word: valid_word(word, spell), doc))
        for doc in list_docs
    ]