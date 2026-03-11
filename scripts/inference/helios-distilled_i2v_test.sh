# Example: Running inference with 2-GPU parallelism
# CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node 2 infer_helios.py \
#     --enable_parallelism \
#     --cp_backend "ulysses" \   #  ["ring", "ulysses", "unified", "ulysses_anything"]

CUDA_VISIBLE_DEVICES=1 python infer_helios.py \
    --base_model_path "/data/models/Helios-Distilled" \
    --transformer_path "/data/models/Helios-Distilled" \
    --sample_type "i2v" \
    --image_path "/data/input/tang640.jpg" \
    --prompt "A man speaks to the camera, which remains still. He has a slight smile, with both hands resting naturally and motionless. The lighting is highly cinematic with soft golden light, the background features Parisian streets and city scenery, with a depth of field effect and a cinematic 35mm film texture." \
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