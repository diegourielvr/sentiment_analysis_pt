import spacy
from tqdm import tqdm

from config.config_nlp import SPACY_DEFAULT_LANG_MODEL, SPACY_NAME_MODELS

nlp = spacy.load(SPACY_NAME_MODELS[SPACY_DEFAULT_LANG_MODEL])

def change_model(lang: str="en"):
    """Cambiar el idioma del modelo de spacy
    :param lang: 'es' o 'en'
    """
    nlp = spacy.load(SPACY_NAME_MODELS[lang])

def get_sentences(documentos: list[str]) -> list[list[str]]:
    """ Devuelve una lista de oraciones para cada documento
    TODO: ¿Que se considera una sentencia?
    Ej. [['oracion 1 doc 1', 'oracion 2 doc 1'], ['oracion 1 doc 2']]
    """
    
    docs = nlp.pipe(documentos)
    docs_sentences = []
    for doc in tqdm(docs):
        docs_sentences.append(list(map(
                    lambda sentence: sentence.text.strip(), doc.sents
                    )))
    return docs_sentences

def get_sentences_with_ids(documentos: list[str]):
    """ Devuelve una lista de oraciones para cada documento
    TODO: ¿Que se considera una sentencia?
    Ej. [['oracion 1 doc 1', 'oracion 2 doc 1'], ['oracion 1 doc 2']]
    """
    
    docs = nlp.pipe(documentos)
    sentence_data = []
    for doc_id, doc in tqdm(enumerate(docs)):
        for sentence_id, sentence in enumerate(doc.sents):
            sentence_data.append({
                'doc_id': doc_id,
                'sentence_id': sentence_id,
                'sentence': sentence.text.strip() 
            })
    return sentence_data

def reconstruct_text_from_df(sentences_df,col_doc_id='doc_id', col_sentence_id='sentence_id',col_sentence='sentence'):
    # Ordenar por doc_id y sentence_id
    sentences_df_sorted = sentences_df.sort_values(by=[col_doc_id, col_sentence_id])
    
    # Agrupar las oraciones por doc_id y reconstruir el texto
    reconstructed_texts = sentences_df_sorted.groupby(col_doc_id)[col_sentence].apply(lambda x: ' '.join(x)).reset_index()
    
    return reconstructed_texts

def get_no_stopwords(documentos: list[str], sep = ' '):
    """Devuelve las palabras de un documento que no son stopwords"""
    docs = nlp.pipe(documentos)
    docs_no_stopwords = []
    for doc in tqdm(docs):
        no_stopwords = []
        for token in doc:
            if not token.is_stop:
                no_stopwords.append(token.text)
        docs_no_stopwords.append(sep.join(no_stopwords))
    return docs_no_stopwords

def tokenize(documentos: list[str]):
    docs = nlp.pipe(tqdm(documentos))
    return list(
        map(
            lambda doc: [token.text for token in doc], 
            tqdm(docs)
        )
    )