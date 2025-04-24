#!/bin/bash

# split_bicamera_video.sh
# Splits a 3840x1080 bicamera video into left/right 1920x1080 halves with optional perspective correction

INPUT_VIDEO="$1"
OUTPUT_DIR="$2"
HALF="${3:-left}"
PERSPECTIVE_CORRECT="${4:-no}"  # 'yes' to apply perspective correction, 'no' otherwise

# Validate inputs
if [ -z "$INPUT_VIDEO" ] || [ -z "$OUTPUT_DIR" ]; then
    echo "Usage: $0 <input_video> <output_dir> [left|right] [yes|no]"
    exit 1
fi

if [ ! -f "$INPUT_VIDEO" ]; then
    echo "Error: Input video $INPUT_VIDEO does not exist"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"
BASENAME=$(basename "$INPUT_VIDEO" | cut -d. -f1)

# Perspective correction parameters (adjust based on measured points)
# Example: Map distorted quadrilateral to rectangle (0,0,1920,0,1920,1080,0,1080)
# Replace with actual measured points from video
PERSPECTIVE_FILTER=""
if [ "$PERSPECTIVE_CORRECT" = "yes" ]; then
    PERSPECTIVE_FILTER="perspective=x0=100:y0=100:x1=1820:y1=100:x2=1920:y2=980:x3=0:y3=980:sense=source"
    echo "Applying perspective correction: $PERSPECTIVE_FILTER"
fi

# Crop and optionally correct perspective
if [ "$HALF" = "left" ]; then
    if [ "$PERSPECTIVE_CORRECT" = "yes" ]; then
        ffmpeg -i "$INPUT_VIDEO" -vf "crop=1920:1080:0:0,$PERSPECTIVE_FILTER" -c:v libx264 -c:a copy "$OUTPUT_DIR/${BASENAME}_left_corrected.mp4"
        echo "Saved left half (corrected): $OUTPUT_DIR/${BASENAME}_left_corrected.mp4"
    else
        ffmpeg -i "$INPUT_VIDEO" -vf "crop=1920:1080:0:0" -c:v libx264 -c:a copy "$OUTPUT_DIR/${BASENAME}_left.mp4"
        echo "Saved left half: $OUTPUT_DIR/${BASENAME}_left.mp4"
    fi
elif [ "$HALF" = "right" ]; then
    if [ "$PERSPECTIVE_CORRECT" = "yes" ]; then
        ffmpeg -i "$INPUT_VIDEO" -vf "crop=1920:1080:1920:0,$PERSPECTIVE_FILTER" -c:v libx264 -c:a copy "$OUTPUT_DIR/${BASENAME}_right_corrected.mp4"
        echo "Saved right half (corrected): $OUTPUT_DIR/${BASENAME}_right_corrected.mp4"
    else
        ffmpeg -i "$INPUT_VIDEO" -vf "crop=1920:1080:1920:0" -c:v libx264 -c:a copy "$OUTPUT_DIR/${BASENAME}_right.mp4"
        echo "Saved right half: $OUTPUT_DIR/${BASENAME}_right.mp4"
    fi
else
    echo "Invalid half: '$HALF'. Use 'left' or 'right'."
    exit 1
fi