# Data Descriptor
This module implements keypoint-based descriptors for a given image dataset.
The implemented method runs the descriptors in parallel for each image within the dataset.

There are a variety of descriptors implemented, but we recommend using VLFeat SIFT with 
Histogram Equalization, which provides the best descriptor for scientific images,
according our [ablation study](ablation).
Set the flag `descriptor=vlfeat_sift_heq` for selecting such descriptor.


| Image Descriptor | Description                                   |
|------------------|-----------------------------------------------|
| **vlfeat_sift_heq**  | **VLFeat SIFT with histogram equalization** |
| sila                 | SILA descriptor                             |
| cv_rsift_heq         | OpenCV RootSIFT with histogram equalization |
| cv_rsift             | OpenCV RootSIFT                             |
| cv_sift_heq          | OpenCV SIFT with histogram equalization     |
| cv_sift              | OpenCV SIFT                                 |
| vlfeat_phow          | VLFeat PHOW descriptor                      |
| vlfeat_dsift         | VLFeat Dense SIFT descriptor                |
| vlfeat_rsift_heq     | VLFeat RootSIFT with histogram equalization |
| vlfeat_rsift         | VLFeat RootSIFT                             |
| vlfeat_sift          | VLFeat SIFT                                 |
    
---


## Quick run
For running the data description, change the parameters of `run_dataset_descriptor.sh`
for specific dataset.

The dataset must be a JSON file with the following structure, similar as found in
[ssp-jsons](spp-jsons):

```json
{
    "<Image_ID_1>": 
    {
        "panel_path": "path/to/image", # Absolute path to panel image
        "panel_class": "<Panel Type>", # Panel type (e.g., Microscopy)
        "doc_id": "<panel-document-id" # Document in which panels belongs to
     },

     ...

    "<Image_ID_2>": 
    {
        "panel_path": "path/to/image",
        "panel_class": "<Panel Type>",
        "doc_id": "<panel-document-id"
     },
}
```

After fixing the `DATASET_BASE_PATH` in `run_dataset_descriptor.sh`, just run:

```
$ ./run_dataset_descriptor.sh
```

After the script ends, you should see a new directory named `description-<panel_class>`,
in which `<panel_class>` is the class type of all files within the input dataset.




```
                                   UNICAMP (University of Campinas) Recod.ai
```