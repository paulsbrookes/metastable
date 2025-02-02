#!/usr/bin/env python3

import json
import os
from pathlib import Path

def md_to_notebook(md_path):
    """Convert a markdown file to a Jupyter notebook with a single cell."""
    with open(md_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": content.split('\n')
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.8.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    return notebook

def main():
    # Create notebooks directory if it doesn't exist
    notebooks_dir = Path('notebooks')
    notebooks_dir.mkdir(exist_ok=True)
    
    # Process all markdown files in explain directory
    explain_dir = Path('explain')
    for md_file in explain_dir.glob('*.md'):
        notebook = md_to_notebook(md_file)
        
        # Create output path
        output_path = notebooks_dir / f"{md_file.stem}.ipynb"
        
        # Save notebook
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=2)
        
        print(f"Converted {md_file} to {output_path}")

if __name__ == '__main__':
    main() 