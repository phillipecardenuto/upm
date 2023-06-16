# Unveiling Scientific Articles from Paper Mills with Provenance Analysis

This repository offers a promising solution to identify fraudulent manuscripts, and it could be a valuable tool for supporting scientific integrity. Its code was developed aiming to identify suspect paper mill cases.

The repo is under construction and its code and annotated database will be released during the following weeks.

## Panel Extractor
Panel extraction is essential to focus on the image regions of interest to the scientific integrity problem, and filter out those that might raise false alarms due to their intrinsic similarity (e.g., diagrams, drawings, and legend indicative letters).
We collected and annotated 3,836 biomedical scientific figures under creative commons license from different journals, creating a dataset of 3,236 figures (32,507 panels) for training the detector of panels, and 600 figures (4,888 panels) for testing it.

| Class            | Images | Labels |     P   |     R   | mAP@.5  | mAP@.5:.95: |
|------------------|--------|--------|---------|---------|---------|-------------|
| all              |   600  |  4888  |  0.941  |  0.935  |  0.95   |    0.901    |
| Blots            |   600  |   804  |  0.998  |  0.989  |  0.995  |    0.871    |
| Graphs           |   600  |  1618  |  0.968  |  0.947  |  0.98   |    0.944    |
| Microscopy       |   600  |  1838  |  0.955  |  0.934  |  0.941  |    0.922    |
| Body Imaging     |   600  |   379  |  0.833  |  0.828  |  0.859  |    0.814    |
| Flow Cytometry   |   600  |   249  |  0.948  |  0.976  |  0.973  |    0.953    |

The [README](panel-extractor/README.md) file from panel-extractor provides instructions to reproduce our 
results.

##### AUTHORS

Phillipe Cardenuto, Daniel Moreira, and Anderson Rocha

```
		                     UNICAMP (University of Campinas) RECOD.AI
```
