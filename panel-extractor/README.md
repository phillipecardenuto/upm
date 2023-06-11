# Panel Detection
## Quick Start

### 1. Get docker image

1.1 Pull from DockerHub:

   ```
   docker pull phillipecardenuto/panel-detection:latest
   ```

1.2 (optional) Build the from Dockerfile
   ```
   ./build.sh
   ```

### 2. Download models:
You may need to install gdown (`pip install gdown`)

	```
	gdown --id 1GGNsJ90VRey_FfuVpOkvikKd7Sbov0Cf
	```
### 3. Unzip models:
	```
	unzip panel_extraction_models.zip
	```


### 4. Run Solution using bash script

```
./extract.sh -i <path/to/figure> -o <dirname>
```

Example

```
./extract.sh -o test -i fig1.png fig3.png
```

Arguments

```
[--weights WEIGHTS] (optional)
			Path to new model weight.
			By default the container already uses the weights locate inside the docker.
			There are two weights models inside the docker one addresing a 4 classes output (best of them);
			other addresing the 5 classe output
[--device DEVICE](optional)
		GPU/CPU Device used during (e.g., 0,1; or cpu)
		(Default) cpu
--output-path -o SAVE_PATH (Required):
		Directory name to save the extracted panels.

--input-path -i INPUT_FIGURE(Required):
		Path to Figure. You can input multiple figure at once 
```


---
# Model training
For training the mode, we annotated a dataset of 3,836 figures of which 3,236 figures (32,507 panels) were used for training and  600 figures (4,888 panels) for testing.
The dataset is freely available at this [link](https://drive.google.com/file/d/1ahGR_-Kcdux_CpWZi9f-6CTMUkJKsfph/view?usp=sharing).

### Training
For training the model, we fine-tuned yolov5 `yolov5x6.pt` weights on our dataset.
You can run train.py from yolov5 directory to reproduce our experiments.

# Model Testing
To reproduce our results:
1. Download the dataset:
`gdown --id 1ahGR_-Kcdux_CpWZi9f-6CTMUkJKsfph`
2. Unzip dataset:
`unzip panels-extraction-dataset.zip`

3. Initiate a docker container from panel-extractor directory:
```
docker run --rm -it \
    --userns=host \
    --gpus all \
    --shm-size=2gb \
    -v `pwd`:/work \
    phillipecardenuto/panel-detection
```

4. Run the test
```
python test.py --data test.yaml --weights=/work/model_5_class.pt
```
After that, the following metrics should be displayed on your screen.

| Class            | Images | Labels |     P   |     R   | mAP@.5  | mAP@.5:.95: |
|------------------|--------|--------|---------|---------|---------|-------------|
| all              |   600  |  4888  |  0.941  |  0.935  |  0.95   |    0.901    |
| Blots            |   600  |   804  |  0.998  |  0.989  |  0.995  |    0.871    |
| Graphs           |   600  |  1618  |  0.968  |  0.947  |  0.98   |    0.944    |
| Microscopy       |   600  |  1838  |  0.955  |  0.934  |  0.941  |    0.922    |
| Body Imaging     |   600  |   379  |  0.833  |  0.828  |  0.859  |    0.814    |
| Flow Cytometry   |   600  |   249  |  0.948  |  0.976  |  0.973  |    0.953    |

### Acknowledgement
The present solution partially uses the code from [YoloV5](https://github.com/ultralytics/yolov5)


**Author**

```
João Phillipe Cardenuto
RECOD.AI - Institute of Computing - UNICAMP
```