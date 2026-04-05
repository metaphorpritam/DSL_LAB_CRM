#!/usr/bin/env python3
"""
Extract images from Jupyter notebook output cells without re-executing.

Reads the notebook JSON, finds all PNG/base64 images in cell outputs,
and saves them as individual PNG files.

Usage:
    python scripts/extract_images_from_notebook.py [--notebook PATH] [--output-dir PATH]
"""

import argparse
import json
import os
import base64
import re


def extract_images(nb_path, output_dir):
    """Extract all images from a notebook's output cells."""
    notebook_name = os.path.splitext(os.path.basename(nb_path))[0]
    cell_output_dir = os.path.join(output_dir, notebook_name)
    os.makedirs(cell_output_dir, exist_ok=True)

    with open(nb_path, 'r') as f:
        nb = json.load(f)

    image_count = 0
    text_output_count = 0

    for cell_idx, cell in enumerate(nb.get('cells', [])):
        if cell.get('cell_type') != 'code':
            continue

        outputs = cell.get('outputs', [])
        for out_idx, output in enumerate(outputs):
            data = output.get('data', {})

            # Check for image/png in the output data
            if 'image/png' in data:
                png_data = data['image/png']

                # Handle both string (base64) and list of strings
                if isinstance(png_data, list):
                    png_b64 = ''.join(png_data)
                else:
                    png_b64 = png_data

                # Clean up the base64 string (remove newlines)
                png_b64 = png_b64.strip()

                # Decode and save
                try:
                    png_bytes = base64.b64decode(png_b64)
                    fname = f"{notebook_name}_cell{cell_idx:03d}_out{out_idx:03d}.png"
                    fpath = os.path.join(cell_output_dir, fname)

                    with open(fpath, 'wb') as img_f:
                        img_f.write(png_bytes)

                    print(f"  Saved: {fname} ({len(png_bytes)} bytes)")
                    image_count += 1

                except Exception as e:
                    print(f"  Error decoding image in cell {cell_idx}, output {out_idx}: {e}")

            # Count text outputs for debugging
            if 'text/plain' in data:
                text_output_count += 1

    print(f"\n{'='*60}")
    print(f"Notebook: {notebook_name}")
    print(f"Images extracted: {image_count}")
    print(f"Output directory: {cell_output_dir}")
    print(f"{'='*60}")

    return image_count


def main():
    parser = argparse.ArgumentParser(description='Extract images from Jupyter notebooks')
    parser.add_argument('--notebook', type=str, default=None,
                       help='Process a single notebook (default: all notebooks)')
    parser.add_argument('--output-dir', type=str, default='Output/img',
                       help='Output directory for images')
    args = parser.parse_args()

    notebooks = [
        'notebooks/eda.ipynb',
        'notebooks/boosting_models.ipynb',
        'notebooks/moe_model_with_log_income.ipynb',
        'notebooks/summary.ipynb',
    ]

    if args.notebook:
        notebooks = [args.notebook]

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    total_images = 0
    for nb_path in notebooks:
        if not os.path.exists(nb_path):
            print(f"Notebook not found: {nb_path}")
            continue
        imgs = extract_images(nb_path, output_dir)
        total_images += imgs

    print(f"\n{'='*60}")
    print(f"TOTAL IMAGES EXTRACTED: {total_images}")
    print(f"OUTPUT DIRECTORY: {os.path.abspath(output_dir)}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
