#!/usr/bin/env python3
"""
Extract all matplotlib figures from Jupyter notebooks and save them to Output/img/

Usage:
    python scripts/extract_notebook_images.py [--notebook PATH] [--output-dir PATH]

This script:
1. Reads notebook cells
2. Adds matplotlib savefig calls before plt.show() calls
3. Executes the notebook
4. Saves all figures to the output directory
"""

import argparse
import nbformat
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import os
import sys


def get_notebooks():
    """List of notebooks to process."""
    return [
        'notebooks/eda.ipynb',
        'notebooks/boosting_models.ipynb',
        'notebooks/moe_model_with_log_income.ipynb',
        'notebooks/summary.ipynb',
    ]


def add_savefig_calls(nb, output_dir, notebook_name):
    """
    Modify notebook cells to save figures before showing them.
    Adds fig.savefig() calls before plt.show() calls.
    """
    figure_counter = 0

    for cell in nb.cells:
        if cell.cell_type != 'code':
            continue

        source_lines = cell.source.split('\n')
        new_source = []

        for line in source_lines:
            stripped = line.strip()

            # Check if line contains plt.show()
            if 'plt.show()' in stripped:
                indent = line[:len(line) - len(line.lstrip())]
                # Get the current figure
                add_lines = [
                    f'{indent}# Save figure',
                    f'{indent}fig_num = plt.gcf().number',
                    f'{indent}os.makedirs("{output_dir}", exist_ok=True)',
                    f'{indent}plt.savefig(f"{{output_dir}}/{notebook_name}_fig_{{figure_counter:03d}}.png", dpi=150, bbox_inches="tight")',
                    f'{indent}figure_counter += 1',
                ]
                new_source.append('\n'.join(add_lines))
                figure_counter += 1
                new_source.append(line)  # Keep original plt.show()
            else:
                new_source.append(line)

        cell.source = '\n'.join(new_source)

    return nb, figure_counter


def add_savefig_calls_v2(nb, output_dir, notebook_name):
    """
    Improved version: Track figures more robustly.
    """
    figure_counter = 0

    for cell_idx, cell in enumerate(nb.cells):
        if cell.cell_type != 'code':
            continue

        # Add import and counter at beginning of first code cell
        if cell_idx == 0:
            init_code = 'import os; figure_counter = 0; import matplotlib.pyplot as plt\n'
            cell.source = init_code + cell.source

        source_lines = cell.source.split('\n')
        new_source = []
        in_multiline_string = False
        string_char = None

        for line in source_lines:
            # Skip comment lines
            if line.strip().startswith('#'):
                new_source.append(line)
                continue

            # Check if plt.show() appears in this line
            if 'plt.show()' in line and not line.strip().startswith('#'):
                # Add savefig before show
                save_code = f"""\n# Auto-save figure
if 'figure_counter' in dir():
    os.makedirs(r'{output_dir}', exist_ok=True)
    _fig = plt.gcf()
    if _fig.get_axes():
        plt.savefig(
            rf'{output_dir}\{notebook_name}_fig_{{figure_counter:03d}}.png',
            dpi=150, bbox_inches='tight'
        )
        figure_counter += 1
        plt.close(_fig)
else:
    pass
"""
                new_source.append(save_code)
                figure_counter += 1

            new_source.append(line)

        cell.source = '\n'.join(new_source)

    return nb, figure_counter


def add_savefig_calls_v3(nb, output_dir, notebook_name):
    """
    Version 3: Use a more targeted approach - find plt.show() calls and
    add savefig before them.
    """
    figure_counter = 0

    for cell in nb.cells:
        if cell.cell_type != 'code':
            continue

        # Find all plt.show() or plt.show() variations in source
        import re
        # Pattern to match plt.show() with optional arguments
        pattern = r'(\s*)(plt\.show\(\s*(?:[^)]*)?\s*\))'

        def replace_show(match):
            nonlocal figure_counter
            indent = match.group(1)
            count = figure_counter
            figure_counter += 1

            save_code = f'''import os; os.makedirs(r\"{output_dir}\", exist_ok=True)
try:
    _fig{count} = plt.gcf()
    if _fig{count}.get_axes():
        _fig{count}.savefig(
            os.path.join(r"{output_dir}", "{notebook_name}_fig_{count:03d}.png"),
            dpi=150, bbox_inches="tight"
        )
        plt.close(_fig{count})
except Exception as e:
    print(f"Error saving figure {count}: {{e}}")
{indent}pass'''
            return save_code + f'\n{indent}# {match.group(2).strip()}'

        cell.source = re.sub(pattern, replace_show, cell.source)

    return nb, figure_counter


def process_notebook(nb_path, output_dir):
    """Process a single notebook and save figures."""
    notebook_name = os.path.splitext(os.path.basename(nb_path))[0]
    cell_output_dir = os.path.join(output_dir, notebook_name)

    print(f"\n{'='*60}")
    print(f"Processing: {nb_path}")
    print(f"Output directory: {cell_output_dir}")
    print(f"{'='*60}")

    try:
        with open(nb_path, 'r') as f:
            nb = nbformat.read(f, as_version=4)
    except Exception as e:
        print(f"Error reading {nb_path}: {e}")
        return 0

    # Modify notebook to save figures
    nb, expected_figs = add_savefig_calls_v3(nb, cell_output_dir, notebook_name)

    print(f"Expected figures: {expected_figs}")

    # Create a temporary notebook cell hook to save figures
    # We'll use a different approach: add a hook that saves on each figure

    # Add a setup cell at the beginning
    setup_cell = nbformat.v4.new_code_cell(f'''
import os
import matplotlib.pyplot as plt

# Global figure counter
_global_fig_counter = [0]
_fig_save_dir = r"{cell_output_dir}"
_notebook_name = "{notebook_name}"

os.makedirs(_fig_save_dir, exist_ok=True)

# Monkey-patch plt.show to also save
_original_show = plt.show
def _auto_save_show(*args, **kwargs):
    try:
        fig = plt.gcf()
        if fig.get_axes():  # Only save if figure has content
            counter = _global_fig_counter[0]
            fname = os.path.join(_fig_save_dir, f"{{_notebook_name}}_fig_{{counter:03d}}.png")
            fig.savefig(fname, dpi=150, bbox_inches=\"tight\")
            print(f"Saved: {{fname}}")
            _global_fig_counter[0] += 1
    except Exception as e:
        print(f"Error saving figure: {{e}}")
    return _original_show(*args, **kwargs)

plt.show = _auto_save_show
print(f"Auto-save enabled. Output: {{_fig_save_dir}}")
''')

    # Insert setup cell at the beginning
    nb.cells.insert(0, setup_cell)

    # Execute the notebook
    from nbconvert.preprocessors import ExecutePreprocessor

    ep = ExecutePreprocessor(timeout=1800, kernel_name='python3')

    try:
        ep.preprocess(nb, {'metadata': {'path': os.path.dirname(nb_path)}})
        print(f"\nSuccessfully executed: {nb_path}")

        # Save the executed notebook
        exec_path = nb_path.replace('.ipynb', '_executed.ipynb')
        with open(exec_path, 'w') as f:
            nbformat.write(nb, f)
        print(f"Saved executed notebook: {exec_path}")

    except Exception as e:
        print(f"Error executing notebook {nb_path}: {e}")
        # Try to save partial results
        try:
            exec_path = nb_path.replace('.ipynb', '_executed.ipynb')
            with open(exec_path, 'w') as f:
                nbformat.write(nb, f)
            print(f"Saved partial execution: {exec_path}")
        except:
            pass

    # Count saved figures
    if os.path.exists(cell_output_dir):
        saved_figs = [f for f in os.listdir(cell_output_dir) if f.endswith('.png')]
        print(f"Total figures saved: {len(saved_figs)}")
        return len(saved_figs)

    return 0


def main():
    parser = argparse.ArgumentParser(description='Extract figures from Jupyter notebooks')
    parser.add_argument('--notebook', type=str, default=None,
                       help='Process a single notebook (default: all notebooks)')
    parser.add_argument('--output-dir', type=str, default='Output/img',
                       help='Output directory for figures')
    args = parser.parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    if args.notebook:
        notebooks = [args.notebook]
    else:
        notebooks = get_notebooks()

    total_figs = 0
    for nb_path in notebooks:
        if not os.path.exists(nb_path):
            print(f"Notebook not found: {nb_path}")
            continue
        figs = process_notebook(nb_path, output_dir)
        total_figs += figs

    print(f"\n{'='*60}")
    print(f"Total figures saved: {total_figs}")
    print(f"Output directory: {os.path.abspath(output_dir)}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
