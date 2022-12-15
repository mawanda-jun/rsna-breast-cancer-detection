cd src/

python trainer.py --path /projects/rsna-breast-cancer-detection/src/configs_uint16/eff4_16_1024_smooth_neg5.yaml
sleep 30
python trainer.py --path /projects/rsna-breast-cancer-detection/src/configs_uint16/eff4_16_1024_smooth_neg5_higherLR.yaml
sleep 30
# python trainer.py --path /projects/rsna-breast-cancer-detection/src/configs_bigger/eff4_1024_smooth_neg5_brightness01.yaml
# sleep 30
# python trainer.py --path /projects/rsna-breast-cancer-detection/src/configs_bigger/eff4_1024_smooth_neg5_tonecurve.yaml
# sleep 30
# python trainer.py --path /projects/rsna-breast-cancer-detection/src/configs_bigger/eff4_1024_smooth_neg5_brightness03.yaml
