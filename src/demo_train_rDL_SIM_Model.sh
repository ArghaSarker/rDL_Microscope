python train_SR_Inference_Module.py --gpu_id '0' --gpu_memory_fraction 0.95 --root_path "../data_train/rDL-SIM/SR/" --data_folder "Microtubules" --save_weights_path "../trained_models/SR_Inference_Module/" --save_weights_suffix "" --load_weights_flag 1 --model_name "DFCAN" --total_iterations 150 --sample_interval 5 --validate_interval 5 --validate_num 2 --batch_size 3 --start_lr 1e-4 --input_height 128 --input_width 128 --input_channels 9 --scale_factor 2 --norm_flag 0
python train_rDL_Denoising_Module.py --gpu_id '1' --gpu_memory_fraction 0.9 --root_path "../data_train/rDL-SIM/DN/" --data_folder "Microtubules" --save_weights_path "../trained_models/rDL_Denoising_Module/" --save_weights_suffix "" --denoise_model "rDL_Denoiser" --load_sr_module_dir "../trained_models/SR_Inference_Module/" --load_sr_module_filter "*Best.weights.h5" --sr_model "DFCAN" --total_iterations 150 --sample_interval 5 --validate_interval 5 --validate_num 2 --input_height 128 --input_width 128 --wave_length 488 --excNA 1.35

