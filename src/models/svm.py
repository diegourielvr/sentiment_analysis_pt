from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from config.constants_models import RANDOM_STATE, TEST_SIZE
from src.models.utils import evaluate

def svm(X, y, vec="TFIDF"):
    print(f"SVM + {vec}")
    print(f"test_size: {TEST_SIZE}, random_state: {RANDOM_STATE}")
    x_train, x_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )
    print(f"shape test: {x_test.shape}")
    
    if vec == "TFIDF":
        model = make_pipeline(
            TfidfVectorizer(),
            SVC()
        )
    elif vec == "BOW":
        model = make_pipeline(
            CountVectorizer(),
            SVC()
        )

    print("Entrenando modelo...")
    model.fit(x_train, y_train)

    print("Evaluando modelo...")
    y_pred = model.predict(x_test)
    report = evaluate(y_pred, y_test)
    
    report['model'] = f"SVM_{vec}"
    return model, report
