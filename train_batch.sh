cd src/

python trainer.py --path /projects/rsna-breast-cancer-detection/src/configs/eff4_512_sim_smooth_neg1_brightness.yaml
sleep 30
python trainer.py --path /projects/rsna-breast-cancer-detection/src/configs/eff4_512_sim_smooth_neg1_randomresizedcrop.yaml
sleep 30
python trainer.py --path /projects/rsna-breast-cancer-detection/src/configs/eff4_512_sim_smooth_neg1_tone.yaml
sleep 30
python trainer.py --path /projects/rsna-breast-cancer-detection/src/configs/eff4_512_sim_smooth_neg1_clahe.yaml
sleep 30
python trainer.py --path /projects/rsna-breast-cancer-detection/src/configs/eff4_512_sim_smooth_neg1_coarsedropout.yaml
sleep 30
python trainer.py --path /projects/rsna-breast-cancer-detection/src/configs/eff4_512_sim_smooth_neg1_pixdrop.yaml
sleep 30
python trainer.py --path /projects/rsna-breast-cancer-detection/src/configs/eff4_512_sim_smooth_neg1_huesat.yaml
sleep 30
python trainer.py --path /projects/rsna-breast-cancer-detection/src/configs/eff4_512_sim_smooth_neg1_gaussblur.yaml
sleep 30
python trainer.py --path /projects/rsna-breast-cancer-detection/src/configs/eff4_512_sim_smooth_neg1_elastic.yaml
sleep 30
python trainer.py --path /projects/rsna-breast-cancer-detection/src/configs/eff4_512_sim_smooth_neg1_gamma.yaml
sleep 30
python trainer.py --path /projects/rsna-breast-cancer-detection/src/configs/eff4_512_sim_smooth_neg1_iso.yaml
sleep 30
python trainer.py --path /projects/rsna-breast-cancer-detection/src/configs/eff4_512_sim_smooth_neg1_perspective.yaml
sleep 30