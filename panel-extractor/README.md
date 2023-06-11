# Panel Detection
## Quick Start

### 1. Get docker image

1.1 Pull from DockerHub:

   ```
   docker pull phillipecardenuto/panel-detection:latest
   ```

1.2 Or Build the from Dockerfile
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


### Acknowledgement
The present solution partially uses the code from [YoloV5](https://github.com/ultralytics/yolov5)


**Author**

```
João Phillipe Cardenuto
RECOD.AI - Institute of Computing - UNICAMP
```