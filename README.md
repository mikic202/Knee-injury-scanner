# Knee Injury Scanner — AI Diagnosis

## Opis projektu
Aplikacja wykorzystująca sieci neuronowe do analizy badań MRI kolana. System wspomaga diagnozę uszkodzeń więzadła krzyżowego przedniego (ACL), klasyfikując przypadki jako:
- Zdrowe
- Częściowo uszkodzone
- Całkowicie zerwane

Dodatkowo aplikacja oferuje moduł **XAI (Explainable AI)**, który wizualizuje obszary decyzyjne modelu za pomocą metod:
- LIME
- Integrated Gradients
- Saliency Maps

## Uruchomienie aplikacji

### Wymagania wstępne
- Python 3.10+
- Zainstalowane zależności systemowe (`libgl1` dla obsługi obrazów)

### Instalacja
1. Sklonuj repozytorium (pamiętaj o submodułach):
   ```bash
   git clone --recurse-submodules <URL_REPOZYTORIUM>
   cd Knee-injury-scanner
   ```

2. Zainstaluj zależności:
   ```bash
   # Używając uv (rekomendowane)
   uv sync
   
   # LUB używając pip
   pip install .
   ```

### Uruchomienie (Lokalnie)
Aby uruchomić aplikację webową:

```bash
PYTHONPATH=. streamlit run src/web_app/main.py

#lub

PYTHONPATH=. uv run streamlit run src/web_app/main.py
```

Aplikacja będzie dostępna pod adresem: `http://localhost:8501`

## 🧪 Testy
Projekt posiada zestaw testów jednostkowych oraz prosty benchmark wydajnościowy.

Aby uruchomić testy:
```bash
# Wszystkie testy
pytest

# Tylko testy aplikacji
pytest tests/test_app.py
```

## 📂 Struktura projektu
```
.
├── checkpoints/       # Zapisane wagi modeli (.pt)
├── datasets/          # Dane wejściowe (pliki .pck i metadata.csv) - w projekcie korzystano ze zbioru KneeMRI z Kaggle
├── src/
│   ├── explainibility/      # Metody XAI (m.in. Saliency-pochodne, Guided Grad-CAM)
│   ├── model_architecture/  # Definicje modeli (ResNet3D, CNN, SAE, transformer)
│   ├── model_training/      # Skrypty treningowe
│   └── web_app/             # Aplikacja Streamlit
│       ├── config.py        # Konfiguracja
│       └── main.py          # Główny plik aplikacji
└── tests/             # Folder z testami zaimplementowanych metod
```

## ⚙️ Konfiguracja
Aplikacja korzysta ze zmiennych środowiskowych (zdefiniowanych w `src/web_app/config.py`):
- `MODEL_PATH`: Ścieżka do pliku z wagami modelu (domyślnie: `checkpoints/resnet3d_best...`)
- `LOG_LEVEL`: Poziom logowania (INFO, DEBUG, ERROR)
