from tqdm import tqdm
from gensim.models import Word2Vec

from config.constants_models import RANDOM_STATE

def create_w2v_model(docs: list[list[str]], dim_embeddings=100, epochs=50, sg=1, path="w2v_model.model"):
    print("Entrenando modelo de W2V...")
    model = Word2Vec(
        sentences=docs,
        vector_size=dim_embeddings,
        epochs=epochs,
        seed=RANDOM_STATE,
        sg=sg # skipgram or cbow
    )
    
    model.save(path)
    print(f"Embeddings guardados en {path}")
    
def get_model(path):
    return Word2Vec.load(path)