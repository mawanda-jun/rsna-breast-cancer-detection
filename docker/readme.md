# How to build the image
1. Change information inside docker image: UNAME, GNAME, UID, GID, git information.
2. Run
```
docker build \
    -t <your image name> \
    .
```

# How to run container (so to use attach VSCode)
1. Create two folders: one containing the repository source, one for the data (the provided dataset and the network outputs).
2. Then run
```
docker run \
    -it \
    --gpus all \
    --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    --volume /PATH/TO/REPOSITORY/SOURCE/rsna-breast-cancer-detection:/projects/rsna-breast-cancer-detection \
    --volume /PATH/TO/STORED/DATASET/rsna-breast-cancer-detection:/data/rsna-breast-cancer-detection \
    --name RSNA-BCD \
    <your image name>
```