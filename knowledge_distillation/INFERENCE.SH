python3 DETECTOR/inference_tool_student.py \
    --input="$input_path/GroundTruths/RGB" \
    --weights="$weights" \
    --config_file="$config_file" \
    --data_file="$data_file" \
    --nms_thresh=0.5 \
    --thresh=0.45 \
    --save_labels

echo "INFERENCE COMPLETED!"

