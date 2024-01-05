CUDA_VISIBLE_DEVICES=1 python inference.py -ir /project/jayinnn/ML_final_project/datasets/train_val/val -o eval_wo_comb.csv -pr records/FGVC-HERBS/ml_final_train_val_wo_combiner/
CUDA_VISIBLE_DEVICES=1 python inference.py -ir /project/jayinnn/ML_final_project/datasets/train_val/val -o eval_high_temp_wo_comb.csv -pr records/FGVC-HERBS/ml_final_train_val_high_temp_wo_comb/
CUDA_VISIBLE_DEVICES=1 python inference.py -ir /project/jayinnn/ML_final_project/datasets/train_val/val -o eval_high_temp.csv -pr records/FGVC-HERBS/ml_final_train_val_high_temp/
CUDA_VISIBLE_DEVICES=1 python inference.py -ir /project/jayinnn/ML_final_project/datasets/train_val/val -o eval_ori.csv -pr records/FGVC-HERBS/ml_final_train_val/
