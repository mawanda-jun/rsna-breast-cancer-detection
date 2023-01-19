
names="effv2s_12_l2_heavyshift.yaml
effv2s_12_heaviershift.yaml

"

for config in $names
do
    docker run \
        -it \
        --rm \
        --gpus all \
        --shm-size=48gb \
        --volume /home/mawanda/projects/rsna-breast-cancer-detection:/projects/rsna-breast-cancer-detection \
        --volume /home/mawanda/Documents/rsna-breast-cancer-detection:/data/rsna-breast-cancer-detection \
        --name RSNA-BCD-trainer \
        mawanda/misc:Pillar \
        python /projects/rsna-breast-cancer-detection/src/trainer.py \
           --path /projects/rsna-breast-cancer-detection/src/configs_12bits/$config
done
