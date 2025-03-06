from src.eval.plots.mc import mostrar_mc
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

def evaluate(y_pred, y_test):
    report = {
        "accuracy": accuracy_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred, average="macro"),
        "precision": precision_score(y_test, y_pred, average="macro"),
        "f1": f1_score(y_test, y_pred, average="macro")
    }
    
    print(f"Accuracy: {report['accuracy']}")
    print("Reporte de clasificacion")
    print(classification_report(y_test, y_pred))

    print("Matriz de confusión")
    mc = confusion_matrix(y_test, y_pred)
    print(mc)
    mostrar_mc(mc)
    return report