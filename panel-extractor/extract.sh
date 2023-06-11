# Input Args
args=$@

# Docker CMD
docker run --rm -it \
    --userns=host \
    --privileged \
    -v `pwd`:/work \
    phillipecardenuto/panel-detection python /app/extract.py $args
