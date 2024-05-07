python train_SR_Inference_Module.py --gpu_id '0' --gpu_memory_fraction 1 --root_path "../data_train/rDL-SIM/SR/" --data_folder "Microtubules" --save_weights_path "../trained_models/SR_Inference_Module/" --save_weights_suffix "" --load_weights_flag 0 --model_name "DFCAN" --total_iterations 150 --sample_interval 5 --validate_interval 5 --validate_num 2 --batch_size 3 --start_lr 1e-4 --input_height 128 --input_width 128 --input_channels 9 --scale_factor 2 --norm_flag 0