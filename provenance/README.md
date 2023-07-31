# Unveiling Paper mills with image provenance 

This repository implements a provenance analysis method dedicated to track reused and 
manipulated images on scientific article.


Disclaimer
> Thought the inhere tool can pinpoint to suspect systematic produced images and documents, its output cannot be interpreted without the expertise of scientific integrity analysts.
> If by any chance do you find any suspicious case, please, report to integrity organizations
> responsible for such case.

We tested this tool on a suspect collection of articles reported by Dr. Bik and 
other investigators named [The Stock Photo Paper Mill](https://scienceintegritydigest.com/2020/07/05/the-stock-photo-paper-mill/).

We also created two other versions of the suspect collection, adding articles 
with no reported problematic images.

The first collection named SPP extended v1 adds 969 scientific figures that contains
7,562 figure panels (e.g., a microscopy image analyzed by the scientific figure).

The second collection named SPP extended v2 adds 3,836 scientific figures to
the SPP extended v1, which includes more 37,397 panels.


<mark>TODO</mark>: Include table with results

<mark>TODO</mark>: Include visual results


## Dataset

<mark>TODO</mark>: Include link to the dataset

## Quick Run
### Installation
After installing an environment with python 3.8+, install the modules required
to run the tool using:
```bash
$ pip install -r requirements.txt
```

### Panel Extraction
For running the provenance solution, we need to extract to isolate the panels from
all scientific figures from the analyzed collection.

<mark>TODO</mark>: Instructions for running the Panel Extraction

### Create panel embedding database
The provenance solution uses the a deep-neural network representation of each 
image for finding traces of manipulation or reuse.

To create such representation and organize it on a database, follow the instruction:

<mark>TODO</mark>: Instructions for running the embeddings database

### Provenance Analysis

<mark>TODO</mark>: Instructions for running the provenance analysis


```
                                   UNICAMP (University of Campinas) Recod.ai
```