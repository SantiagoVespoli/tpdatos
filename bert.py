import os
import warnings
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_recall_fscore_support)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------
# Configuración de entorno (CPU forzado)
# ---------------------------------------------------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = ""
torch.cuda.is_available = lambda: False

print("BERT – Clasificador de atraso cambiario")
print("=" * 60)

# ---------------------------------------------------------------------
# Carga de datos
# ---------------------------------------------------------------------
FILE_PATH = "training.xlsx"  # Cambiar si corresponde

try:
    df = pd.read_excel(FILE_PATH)
    data = df[df["label_llm"].notna()].copy()
    data["text"] = data["text"].astype(str)
    data = data[data["text"].str.len() > 20]

    print(f"Filas válidas        : {len(data)}")
    print("Distribución de etiquetas:", data["label_llm"].value_counts().to_dict())

except Exception as e:
    raise RuntimeError(f"Error al cargar datos: {e}")

# ---------------------------------------------------------------------
# División en entrenamiento y test
# ---------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    data["text"].reset_index(drop=True),
    data["label_llm"].astype(int).reset_index(drop=True),
    test_size=0.3,
    random_state=42,
    stratify=data["label_llm"],
)

print(f"Entrenamiento: {len(X_train)} registros")
print(f"Test        : {len(X_test)} registros")

# ---------------------------------------------------------------------
# Modelo DistilBERT multilingüe
# ---------------------------------------------------------------------
MODEL_NAME = "distilbert-base-multilingual-cased"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
model.to(torch.device("cpu")).train()

print("Modelo DistilBERT cargado (CPU)")

# ---------------------------------------------------------------------
# Dataset ligero
# ---------------------------------------------------------------------
class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {"text": self.texts.iloc[idx], "label": int(self.labels.iloc[idx])}


train_ds = TextDataset(X_train, y_train)
test_ds = TextDataset(X_test, y_test)

# ---------------------------------------------------------------------
# Entrenamiento manual
# ---------------------------------------------------------------------
LEARNING_RATE = 2e-5
EPOCHS = 2
BATCH_SIZE = 4

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

class_weights = compute_class_weight(
    "balanced", classes=np.unique(y_train), y=y_train
)
class_weights = torch.tensor(class_weights, dtype=torch.float)

def make_batch(dataset, idx_slice):
    texts = [dataset[i]["text"] for i in idx_slice]
    labels = [dataset[i]["label"] for i in idx_slice]
    inputs = tokenizer(
        texts, truncation=True, padding=True, max_length=256, return_tensors="pt"
    )
    return inputs, torch.tensor(labels, dtype=torch.long)

print("\nInicio de entrenamiento")
total_steps = 0
for epoch in range(EPOCHS):
    indices = np.random.permutation(len(train_ds))
    epoch_loss = 0.0
    for i in range(0, len(indices), BATCH_SIZE):
        batch_idx = indices[i : i + BATCH_SIZE]
        inputs, labels = make_batch(train_ds, batch_idx)

        outputs = model(**inputs)
        loss = torch.nn.CrossEntropyLoss(weight=class_weights)(
            outputs.logits, labels
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        total_steps += 1

    avg_loss = epoch_loss / (len(indices) // BATCH_SIZE)
    print(f"Época {epoch + 1}/{EPOCHS} – Pérdida media: {avg_loss:.4f}")

print("Entrenamiento finalizado")

# ---------------------------------------------------------------------
# Evaluación
# ---------------------------------------------------------------------
model.eval()
all_preds, all_labels, all_conf = [], [], []

with torch.no_grad():
    for i in range(0, len(test_ds), BATCH_SIZE):
        idx_range = list(range(i, min(i + BATCH_SIZE, len(test_ds))))
        inputs, labels = make_batch(test_ds, idx_range)

        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)

        all_preds.extend(torch.argmax(probs, dim=-1).tolist())
        all_labels.extend(labels.tolist())
        all_conf.extend(torch.max(probs, dim=-1)[0].tolist())

acc = accuracy_score(all_labels, all_preds)
prec, rec, f1, _ = precision_recall_fscore_support(
    all_labels, all_preds, average="binary", zero_division=0
)

print("\nMétricas en test:")
print(f"  Accuracy : {acc:.4f}")
print(f"  Precision: {prec:.4f}")
print(f"  Recall   : {rec:.4f}")
print(f"  F1‑Score : {f1:.4f}")

cm = confusion_matrix(all_labels, all_preds)
print("\nMatriz de confusión")
print("          Pred. 0  Pred. 1")
print(f"Real 0   {cm[0,0]:7d}  {cm[0,1]:7d}")
print(f"Real 1   {cm[1,0]:7d}  {cm[1,1]:7d}")

# ---------------------------------------------------------------------
# Guardado del modelo
# ---------------------------------------------------------------------
SAVE_DIR = "./bert_simple_model"
model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)
print(f"\nModelo y tokenizer guardados en '{SAVE_DIR}'")

# ---------------------------------------------------------------------
# Función de inferencia
# ---------------------------------------------------------------------
def predict_atraso(text: str) -> Tuple[str, float]:
    """Clasifica un texto como atraso cambiario (1) o no (0)."""
    model.eval()
    inputs = tokenizer(text, truncation=True, padding=True, max_length=256, return_tensors="pt")
    with torch.no_grad():
        probs = F.softmax(model(**inputs).logits, dim=-1)
    pred_class = torch.argmax(probs, dim=-1).item()
    confidence = torch.max(probs).item()
    etiqueta = "CON atraso cambiario" if pred_class == 1 else "SIN atraso cambiario"
    return etiqueta, confidence

# Ejemplo de prueba
texto_demo = "La devaluación del peso sugiere atraso cambiario"
resultado, conf = predict_atraso(texto_demo)
print("\nPrueba rápida:")
print(f"  Texto      : {texto_demo}")
print(f"  Clasificación: {resultado} (confianza: {conf:.2f})")
