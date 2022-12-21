cd src/
python trainer.py --path /projects/rsna-breast-cancer-detection/src/configs_big_aug/effv2s_16_1024_smooth_neg5_baseline_long.yaml
sleep 30

python trainer.py --path /projects/rsna-breast-cancer-detection/src/configs_big_aug/effv2s_16_1024_smooth_neg5_brightness01_long.yaml
sleep 30

python trainer.py --path /projects/rsna-breast-cancer-detection/src/configs_big_aug/effv2s_16_1024_smooth_neg5_clahe_long.yaml
sleep 30

python trainer.py --path /projects/rsna-breast-cancer-detection/src/configs_big_aug/effv2s_16_1024_smooth_neg5_tonecurve_long.yaml
sleep 30

python trainer.py --path /projects/rsna-breast-cancer-detection/src/configs_big_aug/effv2s_16_1024_smooth_neg5_huesaturation_long.yaml
sleep 30

python trainer.py --path /projects/rsna-breast-cancer-detection/src/configs_big_aug/effv2s_16_1024_smooth_neg5_elastic_long.yaml
sleep 30

python trainer.py --path /projects/rsna-breast-cancer-detection/src/configs_big_aug/effv2s_16_1024_smooth_neg5_randomgamma_long.yaml
sleep 30

python trainer.py --path /projects/rsna-breast-cancer-detection/src/configs_big_aug/effv2s_16_1024_smooth_neg5_gaussianblur_long.yaml
sleep 30

python trainer.py --path /projects/rsna-breast-cancer-detection/src/configs_big_aug/effv2s_16_1024_smooth_neg5_imagecompression_long.yaml
sleep 30

python trainer.py --path /projects/rsna-breast-cancer-detection/src/configs_big_aug/effv2s_16_1024_smooth_neg5_perspective_long.yaml
sleep 30

python trainer.py --path /projects/rsna-breast-cancer-detection/src/configs_big_aug/effv2s_16_1024_smooth_neg5_pixeldropout_long.yaml
sleep 30
