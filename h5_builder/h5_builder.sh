#!/bin/bash


input_script="batch_ph5_builder.py"
output_prefix="dataset_channel7_part"
final_output="dataset_channel7_final.h5"

total_events=32501

#total_events=202
batch_size=500
current_start=0

while [ $current_start -lt $total_events ]; do
    current_end=$((current_start + batch_size))
    if [ $current_end -gt $total_events ]; then
        current_end=$total_events
    fi

    output_file="${output_prefix}_${current_start}_${current_end}.h5"
    echo "Processing events from $current_start to $current_end, output to $output_file"

    sed -i "7s|^h5_output = .*|h5_output = \"$output_file\"|" "$input_script"
    sed -i "11s|^start_num = .*|start_num = $current_start|" "$input_script"
    sed -i "12s|^end_num = .*|end_num = $current_end|" "$input_script"

    python3 $input_script

    current_start=$current_end
done

echo "Merging all .h5 files into $final_output"
first_file=true
for file in ${output_prefix}_*.h5; do
    if $first_file; then
        cp "$file" "$final_output"
        first_file=false
    else
        python3 - <<EOF
import h5py
import numpy as np

with h5py.File("$final_output", "a") as final_h5:
    with h5py.File("$file", "r") as part_h5:
        dataset_name = "Channel7"
        part_data = part_h5[dataset_name][:]
        final_h5[dataset_name].resize((final_h5[dataset_name].shape[0] + part_data.shape[0]), axis=0)
        final_h5[dataset_name][-part_data.shape[0]:] = part_data
EOF
    fi
    echo "Merged $file into $final_output"
done

echo "All files merged into $final_output successfully."
