#!/usr/bin/env bash
# Convert all four trained models to TensorFlow.js format.
# Uses SavedModel format as input to avoid Keras 3 serialization issues.
# Run this after all four training scripts have completed.
#
# Requirements:
#   pip install tensorflowjs
#
# Usage:
#   bash train/convert_to_tfjs.sh

set -e

MODELS=(oxford_pet utkface esc50 ballroom)

for MODEL in "${MODELS[@]}"; do
    SRC="models/${MODEL}_savedmodel"
    DST="demo/models/${MODEL}"

    if [ ! -d "$SRC" ]; then
        echo "  SKIP: $SRC not found (run training first)"
        continue
    fi

    echo "Converting $MODEL ..."
    tensorflowjs_converter \
        --input_format=tf_saved_model \
        --quantize_float16 \
        "$SRC" \
        "$DST"

    echo "  -> $DST"
done

echo ""
echo "All done. Commit demo/models/ and demo/test_data/ to the repo,"
echo "then push to trigger a new Vercel deployment."
