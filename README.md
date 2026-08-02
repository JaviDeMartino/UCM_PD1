<div align="center">

# DATA PROJECT | Novella

### Predicting *New York Times* bestsellers with machine learning

![Course](https://img.shields.io/badge/course-Data%20Project%20I-blue?style=flat-square)
![Academic Year](https://img.shields.io/badge/academic%20year-2023--2024-blue?style=flat-square)
![Python](https://img.shields.io/badge/python-3.10%2B-blue?style=flat-square)

Carmen Fernández González · Javier Martín Fuentes · María Romero Huertas

</div>

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Repository Structure](#repository-structure)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Dependencies](#dependencies)
7. [Data Sources](#data-sources)
8. [Results and Evaluation](#results-and-evaluation)
9. [Future Work](#future-work)
10. [Acknowledgments](#acknowledgments)

## Project Overview

The goal of this project is to build a machine learning system capable of predicting which books have the potential to become **bestsellers in the United States**. The system is trained on historical data about books that have appeared on the *New York Times* Best Sellers list, which is updated weekly.

For more information about how this list is compiled, see [About the Best Sellers](https://www.nytimes.com/books/best-sellers/methodology/).

The project follows a full data-science pipeline: web scraping data from multiple sources, cleaning and merging it into a single dataset, exploring it, engineering features, training and tuning several candidate models, and finally evaluating the selected model against newly captured data and a simple heuristic baseline.

## Features

- Web scraping of weekly *NYT* bestseller lists and monthly GoodReads "popular books" lists, with configurable start date and lookback window.
- Collection of per-book metadata from GoodReads (rating, page count, genres, historical reviews, etc.).
- Price scraping from Barnes & Noble.
- Author metadata extraction from Wikipedia and GoodReads (nationality, years active, gender, social media presence, etc.).
- Popularity signals over time via Google Trends.
- Data cleaning pipeline: deduplication, null handling, feature engineering, date formatting, price imputation.
- Exploratory data analysis notebooks for books, authors, and drift between training and newly captured data.
- Feature-relevance analysis (Spearman correlation, mutual information, chi-squared test).
- Hyperparameter tuning (grid search and randomized search, ~200 combinations each) for three candidate models: regularized Logistic Regression, Random Forest, and a Multi-Layer Perceptron (MLP), tracked with MLflow.
- Candidate model comparison and selection based on held-out test performance.
- Final model evaluation against a heuristic baseline, including segment-level analysis, on newly captured data.
- Utility script to download intermediate datasets and model registries from Google Drive.

## Repository Structure

```
.
├── requirements.txt
└── src/
    ├── adquisicion/        # Data acquisition (web scraping)
    │   ├── librosNYT.py             # NYT bestseller list scraper
    │   ├── librosPopulares.py       # GoodReads "popular books" scraper
    │   ├── goodreads.py             # Per-book GoodReads metadata
    │   ├── goodreadsReviews.py      # Historical GoodReads reviews (Playwright)
    │   ├── barnesAndNoble.py        # Book price scraper
    │   ├── autoresWikipedia.py      # Author info from Wikipedia
    │   ├── autoresGoodreads.py      # Author info from GoodReads
    │   ├── googleTrends.py          # Popularity over time (Google Trends)
    │   ├── main_adquisicion_1.py    # Stage 1: NYT + popular book lists
    │   ├── main_adquisicion_2.py    # Stage 2: enrichment from clean titles
    │   └── main_nuevaCaptura_1.py   # Stage 1 equivalent for newly captured data
    ├── limpieza/            # Data cleaning and feature engineering
    │   ├── limpieza.py               # Book cleaning functions
    │   ├── limpieza_autores.py       # Author cleaning functions
    │   ├── main_limpieza_1.py        # Merge + clean raw book lists
    │   ├── main_limpieza2.py         # Final book/author cleaning + integration
    │   └── main_limpieza_autores_nuevaCaptura.py  # Author cleaning for new data
    ├── exploracion/         # Exploratory data analysis notebooks
    │   ├── exploracion.ipynb
    │   ├── exploracion_autores.ipynb
    │   └── exploracion_variables_relevantes.ipynb  # Drift analysis
    ├── modelos/             # Feature analysis, training and tuning
    │   ├── AnalisisVariablesRelevantes.ipynb
    │   ├── evaluacionModelosCandidatos.ipynb
    │   ├── utilidadesModelos.py      # Shared constants, metrics, pipeline, MLflow helpers
    │   ├── regresionLogistica/
    │   ├── RandomForest/
    │   └── MLP/
    ├── evaluacion/          # Final model evaluation
    │   └── evaluacionModeloFinal.ipynb
    └── drive/               # Google Drive dataset/model download helper
        ├── drive.py
        ├── descarga_archivos_GD.py
        └── archivos_info.txt      # Google Drive file IDs and destinations
```

No datasets are stored in this repository. Notebooks and scripts read/write data under `src/data/raw`, `src/data/clean` and `src/databases` (MLflow's SQLite backends), which are downloaded on demand via the `drive` module — see [Data Sources](#data-sources).

## Installation

Clone the repository:

```bash
git clone <repository-url>
cd UCM_PD1
```

Make sure you have Python installed (3.10 or higher is recommended). Then install the required libraries:

```bash
pip install -r requirements.txt
```

You can work with the `.py` files in any editor/IDE (the original authors used PyCharm). The `.ipynb` notebooks require Jupyter:

```bash
jupyter notebook path/to/notebook.ipynb
```

Anaconda Navigator can also be used to launch Jupyter Notebook and open the notebooks from its interface.

## Usage

All commands below assume the working directory noted for each step — this mirrors how the scripts were originally written and executed.

**1. Acquisition and cleaning**

Both the acquisition and cleaning stages have two "main" scripts each, since correctly obtaining the data requires alternating between them. The correct execution order is:

- From `src/`: `python -m adquisicion.main_adquisicion_1` — downloads the NYT bestseller list and the list of popular GoodReads books.
- From `src/adquisicion/`: `python goodreadsReviews.py` — downloads historical GoodReads reviews up to a chosen date. This is a separate module because it takes a long time to run; the rest of the pipeline can run in parallel since it's only integrated into the final dataset at the end.
- From `src/`: `python -m limpieza.main_limpieza_1` — cleans titles, merges all books and removes duplicates.
- From `src/`: `python -m adquisicion.main_adquisicion_2` — downloads further GoodReads info, prices, Google Trends data and author data (Wikipedia/GoodReads) based on the cleaned titles.
- From `src/limpieza/`: `python main_limpieza2.py` — cleans the combined book and author dataset (nulls, new variables, date formatting, unnecessary columns removed) and integrates historical reviews, leaving the data ready for model training.

The same applies to the new-data-capture stage: in some cases the scripts above must be replaced with their `nuevaCaptura` counterparts, which target different time windows / more efficient capture methods.

**2. Exploration**

Explore the data distributions with the notebooks in `src/exploracion/`: `exploracion.ipynb` and `exploracion_autores.ipynb`.

**3. Model training**

Before selecting candidate models, variable relevance with respect to the response variable was analyzed using metrics appropriate to each variable's type (Spearman correlation, mutual information, chi-squared test), discarding variables with no influence on the outcome — run `src/modelos/AnalisisVariablesRelevantes.ipynb`.

Hyperparameters were tuned for three models: a regularized Logistic Regression, a Random Forest, and a Multi-Layer Perceptron. Each model has a notebook for variable selection (where applicable) and another for hyperparameter search, under `src/modelos/regresionLogistica/`, `src/modelos/RandomForest/` and `src/modelos/MLP/`. Two search strategies were used — grid and random search — with around 200 hyperparameter combinations tried per strategy.

Finally, the three models were evaluated on the same test set in `src/modelos/evaluacionModelosCandidatos.ipynb`, and the MLP was selected for its superior metrics. This notebook also performs a segment-level analysis and inspects the most relevant variables per model.

**4. New data capture**

The same acquisition/cleaning mains can be reused for capturing new data, substituting in the `nuevaCaptura`-suffixed mains where the time window or capture method differs, as in step 1.

**5. Drift analysis**

Drift in the new data can be analyzed with `src/exploracion/exploracion_variables_relevantes.ipynb`, which produces visualizations and runs statistical tests to check whether the distributions of key variables come from the same population in both datasets.

**6. Model evaluation**

Finally, the selected candidate model can be evaluated ahead of production by comparing it against the heuristic baseline and running a segment-level analysis in `src/evaluacion/evaluacionModeloFinal.ipynb`.

## Dependencies

Main libraries and tools used across the pipeline (see `requirements.txt` for exact pinned versions):

- **Data handling:** pandas, numpy, pyarrow, scipy
- **Web scraping:** requests, BeautifulSoup4, Selenium, Playwright
- **Text/entity matching:** fuzzywuzzy, python-Levenshtein, country_converter
- **Trends data:** pytrends
- **Machine learning:** scikit-learn, imbalanced-learn (SMOTE-NC)
- **Experiment tracking:** MLflow
- **Visualization:** seaborn
- **Notebooks:** Jupyter

## Data Sources

- **[The New York Times Best Sellers](https://www.nytimes.com/books/best-sellers/)** — the list of books that became bestsellers in the United States.
- **[GoodReads](https://www.goodreads.com)** — monthly lists of popular books by publication date, plus per-book details (publication date, rating, genres, etc.) and author information.
- **[Wikipedia](https://es.wikipedia.org/wiki/Wikipedia:Portada)** — biographical author information (years active, gender, nationality, etc.).
- **[Barnes & Noble](https://www.barnesandnoble.com)** — the largest US bookstore, used as the source for book prices.
- **[Google Trends](https://trends.google.es/trends/)** — popularity measures for each book.

None of the collected/processed datasets are committed to this repository. `src/drive/archivos_info.txt` lists the Google Drive file IDs, local filenames and destination folders (`data/raw`, `data/clean`, `databases`) for every intermediate artifact; `src/drive/drive.py` and `src/drive/descarga_archivos_GD.py` download them on demand and are called directly from several notebooks.

## Results and Evaluation

After tuning hyperparameters for three different models with different search strategies, the **Multi-Layer Perceptron (MLP)** was selected to move on to production, with the following hyperparameters:

- *activation* — `logistic`
- *alpha* — 0.8773407884629941
- *early_stopping* — `True`
- *hidden_layer_sizes* — (150, 150)
- *learning_rate* — `adaptive`
- *learning_rate_init* — 0.0023019050769459534
- *random_state* — 22

The model was evaluated against a heuristic baseline defined as: *"a book will be a bestseller if its author has had at least one previous bestseller."* Results on the full training dataset and on the newly captured dataset:

| Model | Data | Balanced Accuracy | Sensitivity | Specificity |
|---|---|---|---|---|
| MLP | Test (train) | 0.78 | 0.73 | 0.83 |
| MLP | New data | 0.72 | 0.52 | 0.92 |
| Heuristic | Test (train) | 0.63 | 0.29 | 0.97 |
| Heuristic | New data | 0.73 | 0.50 | 0.97 |

The MLP's metrics degrade somewhat on the new data while the heuristic's predictions improve slightly. This can be attributed to several factors, such as noise introduced during Google Trends data capture, or shifts in the distribution of authors' previous bestsellers — confirmed by a Mann–Whitney U test performed during drift analysis.

However, for books whose authors had no previous bestsellers, the model showed real predictive power that the heuristic completely lacked (the heuristic can never predict a bestseller in this segment by construction):

| Model | Data | Balanced Accuracy | Sensitivity | Specificity |
|---|---|---|---|---|
| MLP | Test (train), no previous bestsellers | 0.75 | 0.65 | 0.84 |
| MLP | New data, no previous bestsellers | 0.58 | 0.22 | 0.93 |
| Heuristic | Test (train), no previous bestsellers | 0.50 | 0 | 1 |
| Heuristic | New data, no previous bestsellers | 0.50 | 0 | 1 |

## Future Work

Based on these results, a new direction opens up for this project: building a model specialized in books whose authors have had no previous bestsellers. This is precisely the segment where conventional heuristics fail, yet where a good bestseller-potential prediction could offer a significant competitive advantage.

## Acknowledgments

We would like to thank Javier Arroyo for supervising this project. His detailed feedback and constant guidance were fundamental to its development.
