import json
import os

def patch_notebook(nb_path, depth=2):
    print(f"Patching {nb_path}...")
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    # Path prefix based on depth
    prefix = "../" * depth
    
    # 1. Update sys.path in an early code cell
    # We'll look for the first code cell or insert one at the top.
    path_boilerplate = [
        "import sys, os\n",
        f"sys.path.append(os.path.abspath('{prefix}'))\n",
        "print(f\"Root project directory added to sys.path: {os.path.abspath('{prefix}')}\")"
    ]
    
    # Check if first cell is setup.
    found_setup = False
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            cell['source'] = path_boilerplate + ["\n\n"] + cell['source']
            found_setup = True
            break
            
    if not found_setup:
        new_cell = {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": path_boilerplate
        }
        nb['cells'].insert(0, new_cell)

    # 2. Update hardcoded paths in all cells (heuristic)
    # Common result dirs: master_figures, heavy_master_results, DoHBrw-2020
    targets = ["master_figures", "heavy_master_results", "DoHBrw-2020", "experiment_results", "Thesis_Results"]
    
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            new_source = []
            for line in cell['source']:
                for target in targets:
                    # Simple replacement: target/ -> ../../target/
                    # We avoid replacing if it already has ../
                    if target in line and f"{prefix}{target}" not in line:
                        line = line.replace(f'"{target}/', f'"{prefix}{target}/')
                        line = line.replace(f"'{target}/", f"'{prefix}{target}/")
                        line = line.replace(f'"{target}"', f'"{prefix}{target}"')
                        line = line.replace(f"'{target}'", f"'{prefix}{target}'")
                new_source.append(line)
            cell['source'] = new_source

    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

# List of notebooks to patch
notebooks = [
    ("notebooks/main_experiments/main_evaluation_pipeline.ipynb", 2),
    ("notebooks/main_experiments/legacy_dns_tunnelling.ipynb", 2),
    ("notebooks/main_experiments/full_security_analysis.ipynb", 2),
    ("notebooks/collaborations/mohsin_rl_features.ipynb", 2),
    ("notebooks/collaborations/mohsin_feature_engineering.ipynb", 2),
    ("notebooks/benchmarks/optimization_benchmarks.ipynb", 2)
]

for nb_path, depth in notebooks:
    if os.path.exists(nb_path):
        patch_notebook(nb_path, depth)
    else:
        print(f"Skipping {nb_path}, not found.")
