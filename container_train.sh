# "effv2s_16_1024_smooth_neg5_baseline.yaml 
# effv2s_16_1024_smooth_neg5_brightness01.yaml 
# effv2s_16_1024_smooth_neg5_clahe.yaml 
# effv2s_16_1024_smooth_neg5_tonecurve.yaml 
# effv2s_16_1024_smooth_neg5_huesaturation.yaml 
# effv2s_16_1024_smooth_neg5_elastic.yaml 
# effv2s_16_1024_smooth_neg5_randomgamma.yaml 
# effv2s_16_1024_smooth_neg5_gaussianblur.yaml 
# effv2s_16_1024_smooth_neg5_imagecompression.yaml 
# effv2s_16_1024_smooth_neg5_perspective.yaml 
# effv2s_16_1024_smooth_neg5_pixeldropout.yaml"

# names="effv2s_smooth_baseline.yaml 
# effv2s_light_shift.yaml 
# effv2s_brightness.yaml 
# effv2s_brightness.yaml 
# effv2s_clahe.yaml 
# effv2s_coarsedropout.yaml "

names="effv2s_heavyshift_brightness_clahe.yaml 
effv2s_smooth_heavyshift_brightness_clahe.yaml "

for config in $names
do

    docker run \
        -it \
        --rm \
        --gpus all \
        --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
        --volume /home/mawanda/projects/rsna-breast-cancer-detection:/projects/rsna-breast-cancer-detection \
        --volume /home/mawanda/Documents/rsna-breast-cancer-detection:/data/rsna-breast-cancer-detection \
        --name RSNA-BCD-trainer \
        mawanda/misc:Pillar \
        python /projects/rsna-breast-cancer-detection/src/trainer.py \
            --path /projects/rsna-breast-cancer-detection/src/configs_big_aug/$config

done