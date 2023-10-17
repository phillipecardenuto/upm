# Dataset Overview

This repository provides access to various datasets for testing the provenance analysis method. Below is a brief description of the datasets and their contents.

## Toy Data

The toy data files (found in `dev-*.json`) offer a small-scale version of the dataset for testing the methods provided in this repository.

- `dev_dataset_gt.json`: Ground truth for two graphs reported by Dr. Bik.
- `dev_dataset.json`: Panel image annotations, including class, path, and document ID.
- `dev_doc_annotation.json`: Document identifiers with shared content links annotations. If a document is absent from this file, it does not share elements with any other.
- `dev_doc`: Contains all documents from the dataset.
- `dev_evidence.json`: Deep learning features describing each panel image within the dataset.

## Annotations

The JSON files in the `annotations` directory provide ground truth for images, panels, or documents sharing elements with others. If an image or document is not present in these files, it is not reported as sharing content with others.
The file `our_annotation.json` provides a panel-level ground truth, while the `document-level-annotation.json` provides the document-level one.
The original dataset annotation is located at [Dr. Bik's website](https://scienceintegritydigest.com/2020/07/05/the-stock-photo-paper-mill/).

## Metadata

This directory contains metadata essential for downloading articles, figures, and extracting panels from the SPP, SPP-v1, and SPP-v2 datasets. It includes URLs used for dataset content retrieval.

- `spm/spm-annotation.json`: Comprehensive information on the stock-photo-papermill dataset, including annotations for '_id,' 'title,' 'doi,' 'pmid,' 'abstract,' 'url,' 'docPath,' 'publisher,' 'publishedDate,' 'figures,' and 'panels.'
- `annotated_panels/*-metadata.csv`: Metadata for the dataset used in training the panel extractor solution. These data were incorporated into SPP to create SPP-v1.
- `extracted_panels/extracted_panels.csv`: Metadata for all articles added to SPP-v1, contributing to the creation of SPP-V2.



## Dataset Issues
If you face any issues while downloading and organizing the data, please contact the corresponding author (Phillipe Cardenuto).