# Example: Running inference with 2-GPU parallelism
# CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node 2 infer_helios.py \
#     --enable_parallelism \
#     --cp_backend "ulysses" \   #  ["ring", "ulysses", "unified", "ulysses_anything"]

CUDA_VISIBLE_DEVICES=1 python infer_helios.py \
    --base_model_path "/data/models/Helios-Distilled" \
    --transformer_path "/data/models/Helios-Distilled" \
    --sample_type "i2v" \
    --image_path "/data/input/唐老师原图未加工.jpg" \
    --width 384 \
    --height 640 \
    --prompt "A man is talking." \
    --num_frames 240 \
    --guidance_scale 1.0 \
    --is_enable_stage2 \
    --pyramid_num_inference_steps_list 2 2 2 \
    --is_amplify_first_chunk \
    --enable_compile \
    --output_folder "./output_helios/helios-distilled"


    # --enable_low_vram_mode \
    # --group_offloading_type "leaf_level" \  # ["leaf_level", "block_level"]
    # --num_blocks_per_group
    # --pyramid_num_inference_steps_list 1 1 1 \