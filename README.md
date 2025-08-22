# Topic-Modelling-and-Clustering-for-Low-Resource-Parallel-Corpus

This repository contains code and resources for my bachelor thesis on **Topic-Modelling-and-Clustering-for-Low-Resource-Parallel-Corpus** in the **German–Sorbian** language pairs.  
The project explores multilingual embeddings, clustering, cross-lingual mapping, and label transfer to evaluate representation quality in endangered languages.

---

## Project Overview
- **Goal:** Evaluate multilingual sentence embeddings for parallel sentence mining in low-resource settings.  
- **Data:** WMT22 Very Low Resource Shared Task (German–Upper/Lower Sorbian) URL:https://statmt.org/wmt22/unsup_and_very_low_res.html.  
- **Methods:** LaBSE, Glot500, LASER, XLM-R; K-means clustering; TF-IDF & LDA topic extraction; cross-lingual mapping; CBIE isotropy enhancement; classifier training and label projection..  

---
## Code Notes
- The implementation of **CBIE** is adapted from the function `cluster_based` in  [kathyhaem/outliers](https://github.com/kathyhaem/outliers), file `src/post_processing.py`. The original code was reused and modified for this project.
