#!/bin/bash

echo "Processing samples 0 to 735..."

for i in {0..735}
do
    echo "Running sample $i..."
    python explain_transformer_output.py --config configs/SegFormer3D.json --sample_idx $i
    echo "---"
done

echo "Done!"