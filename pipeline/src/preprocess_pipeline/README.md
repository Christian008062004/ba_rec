# Preprocess-Pipeline

Verarbeitet eTracker-Rohdaten zu fertigen Modelleingabedaten.
Jeder Schritt kann einzeln ausgeführt werden und produziert ein lesbares Zwischenergebnis.

## Schritte

### 1. `steps/build_staging_base.py`
**Eingabe:** `raw/*.zip`  
**Ausgabe:** `model_input/staging/dataset.inter`, `.item`, `.user`

Liest alle eTracker-onsite-CSVs aus den ZIP-Dateien, filtert relevante Events,
wendet k-core-Filterung an und schreibt die Basis-Atomic-Files.
Item-IDs sind zu diesem Zeitpunkt noch Variant-IDs aus dem eTracker.

```bash
python src/preprocess_pipeline/steps/build_staging_base.py --test-mode --max-zip-files 1
```

---

### 2. `steps/enrich_parent_ids.py` **Eingabe:** `model_input/staging/dataset.inter`, `.item` + `cache/product_cache.json`  
**Ausgabe:** `model_input/staging/dataset.inter`, `.item` (überschrieben)

Ersetzt Variant-IDs durch `parentProductNumber` in `.inter` und `.item`.

---

### 3. `steps/enrich_item_api.py` **Eingabe:** `model_input/staging/dataset.item` + `cache/product_cache.json`  
**Ausgabe:** `model_input/staging/dataset.item` (überschrieben, neue Spalten)

Ergänzt `.item` um API-Daten: Preis, Marke, Kategorien.

---

### 4. `steps/enrich_user_geo.py` **Eingabe:** `model_input/staging/dataset.user` + `cache/geocoding_cache.json`  
**Ausgabe:** `model_input/staging/dataset.user` (überschrieben, neue Spalten)

Ergänzt `.user` um Geokoordinaten per GeoNames-DE-Index und Geocoding-API.

---

### 5. `steps/build_embedding_input.py`
**Eingabe:** `cache/product_cache.json`  
**Ausgabe:** `model_input/temp/embedding_input.json`

Baut den Embedder-Input-Text pro `parentProductNumber` aus den Produktfeldern.

```bash
python src/preprocess_pipeline/steps/build_embedding_input.py
```

---

### 6. `steps/build_embeddings.py`
**Eingabe:** `model_input/temp/embedding_input.json`  
**Ausgabe:** `model_input/embeddings.npz`

Berechnet Sentence-Transformer-Embeddings. Bereits berechnete IDs werden übersprungen.

```bash
python src/preprocess_pipeline/steps/build_embeddings.py
```

---

### 7. `steps/build_train_parquet.py`
**Eingabe:** `model_input/staging/dataset.inter`, `.item`, `.user`  
**Ausgabe:** `model_input/train.parquet`

Joined alle Staging-Files zu einer fertigen Trainingsdatei.

```bash
python src/preprocess_pipeline/steps/build_train_parquet.py
```

---

## Hilfsfunktionen (`utils/`)

| Datei | Inhalt |
|---|---|
| `utils/atomic_files.py` | `write_inter_file`, `write_item_file`, `write_user_file` |
| `utils/geocoding.py` | GeoNames-Index, Geocoding-API-Abfragen, Cache |

## Komplette Pipeline auf einmal

```bash
python src/preprocess_pipeline/pipeline.py
```
