# OSIRIS

> **Ingérer, profiler et explorer des données structurées — avec une boîte à outils Python modulaire.**

OSIRIS transforme des sources brutes en données prêtes à analyser. Il réunit le profilage, les contrôles qualité, la similarité et plusieurs modes de recherche dans un même projet.

## Sommaire

- [Fonctionnalités](#fonctionnalités)
- [Installation](#installation)
- [Services complémentaires](#services-complémentaires)
- [Commandes](#commandes)
- [Organisation du projet](#organisation-du-projet)

## Fonctionnalités

| | Fonction |
|---|---|
| **Ingestion** | Normalisation des sources et création de shards Parquet |
| **Profilage** | Analyse des colonnes, des types et de la qualité des données |
| **Similarité** | Comparaison de textes et détection de rapprochements |
| **Recherche** | Recherche lexicale, hybride ou vectorielle avec Qdrant |
| **Embeddings** | Génération via Text Embeddings Inference (TEI) |
| **Transfert** | Échange de caches avec Kafka |

## Installation

Prérequis : Python 3.10 ou plus récent. Docker Compose est nécessaire uniquement pour les services complémentaires.

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Le fichier `requirements.txt` installe toutes les dépendances Python du projet en une seule étape.

## Services complémentaires

Les services sont regroupés dans [`compose.yaml`](compose.yaml) et se lancent à la demande.

**Qdrant et Kafka**

```powershell
docker compose --profile kafka up -d qdrant kafka
```

**Embeddings avec TEI** — nécessite une configuration Docker compatible avec l’image CUDA et un GPU NVIDIA.

```powershell
docker compose --profile embeddings up -d tei
```

## Commandes

Les fonctionnalités sont disponibles sous forme de modules Python. Pour afficher les options d’une commande :

```powershell
python -m src.command.ingest_raw --help
python -m src.command.profiler --help
python -m src.command.semantic_profile --help
python -m src.command.query --help
python -m src.command.hybrid_query --help
python -m src.command.qdrant --help
python -m src.command.kafka_transfer --help
```

## Organisation du projet

```text
src/
├── command/       Interfaces en ligne de commande
├── embeddings/    Modèles et traitement des embeddings
├── ingestion/     Ingestion et normalisation
├── profiling/     Profilage statistique
├── quality/       Contrôles qualité
├── semantic/      Analyse et rapprochement sémantiques
├── similarity/    Mesures de similarité
└── vector_search/ Index et recherche vectorielle

cache/              Fichiers générés et caches
compose.yaml        Services Docker
requirements.txt    Dépendances Python
```
