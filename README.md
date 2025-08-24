# Topic-Modelling-and-Clustering-for-Low-Resource-Parallel-Corpus

This repository contains code and resources for my bachelor thesis on **Topic-Modelling-and-Clustering-for-Low-Resource-Parallel-Corpus** in the **German–Sorbian** language pairs.
All experiments and examples in this repository use Upper Sorbian (hsb) as the endangered language.
For Lower Sorbian (dsb), the exact same pipeline applies: simply replace the input data files with the dsb–de parallel corpus.
The project explores multilingual embeddings, clustering, cross-lingual mapping, and label transfer to evaluate representation quality in endangered languages.

---

## Project Overview
- **Goal:** Evaluate multilingual sentence embeddings for parallel sentence mining in low-resource settings.  
- **Data:** WMT22 Very Low Resource Shared Task (German–Upper/Lower Sorbian) URL:https://statmt.org/wmt22/unsup_and_very_low_res.html.  
- **Methods:** LaBSE, Glot500, LASER, XLM-R; K-means clustering; TF-IDF & LDA topic extraction; cross-lingual mapping; CBIE isotropy enhancement; classifier training and label projection..  

---
## Code Notes
- The implementation of **CBIE** is adapted from the function `cluster_based` in  [kathyhaem/outliers](https://github.com/kathyhaem/outliers), file `src/post_processing.py`. The original code was reused and modified for this project.

---
## Project Structure

```
├── README.md
├── requirements.txt
├── data/
│   ├── train.hsb-de.hsb
│   ├── train.hsb-de.de
│   ├── 40194_train_dsb_de.dsb
│   └── 40194_train_dsb_de.de
├── notebooks/
│   └── LaBSE.ipynb
├── src/
│   ├── data/
│   │   ├── preprocess.py
│   │   └── keyword_seed.py
│   ├── embeddings/
│   │   ├── encode_Glot500.py
│   │   ├── encode_XLM-R.py
│   │   ├── encode_Laser.py
│   │   └── encode_labse.py
│   ├── clustering/
│   │   ├── clustering_analysis.py
│   │   ├── kmeans.py
│   │   ├── tfidf_keywords.py
│   │   └── lda_keywords.py
│   ├── mapping/
│   │   ├── mapping_report.py
│   │   └── model_comparison.py
│   ├── classification/
│   │   ├── classifier_de_side.py
│   │   ├── cv_eval.py
│   │   ├── label_projection_eval.py
│   │   ├── label_vis.py
│   │   └── topic_labeling.py
│   ├── tsne_umap_cbie/
│   │   ├── CBIE.py
│   │   ├── crosslingual_consistency.py
│   │   ├── tsne_vis.py
│   │   └── umap_vis.py
```
## Installation
```bash
git clone https://github.com/yourusername/Topic-Modelling-and-Clustering-for-Low-Resource-Parallel-Corpus.git
cd Topic-Modelling-and-Clustering-for-Low-Resource-Parallel-Corpus
pip install -r requirements.txt

## Usage
### Generate embeddings
```bash
python src/embeddings/encode_labse.py --input data/train.hsb-de.hsb --output embeddings/hsb_labse.npy
