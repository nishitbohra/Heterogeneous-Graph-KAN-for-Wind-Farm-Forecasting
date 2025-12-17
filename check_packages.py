"""
Check and display package installation status.
"""

import sys
import subprocess

def check_package(package_name, import_name=None):
    """Check if a package is installed and can be imported."""
    if import_name is None:
        import_name = package_name
    
    try:
        __import__(import_name)
        return True, "✓ Installed"
    except ImportError:
        return False, "✗ Missing"

print("="*70)
print("PACKAGE INSTALLATION CHECK")
print("="*70)

packages = {
    'Core Packages': [
        ('torch', 'torch', 'PyTorch'),
        ('numpy', 'numpy', 'NumPy'),
        ('pandas', 'pandas', 'Pandas'),
        ('scipy', 'scipy', 'SciPy'),
    ],
    'Visualization': [
        ('matplotlib', 'matplotlib', 'Matplotlib'),
        ('seaborn', 'seaborn', 'Seaborn'),
    ],
    'Machine Learning': [
        ('scikit-learn', 'sklearn', 'Scikit-learn'),
    ],
    'Jupyter': [
        ('jupyter', 'jupyter', 'Jupyter'),
        ('ipykernel', 'ipykernel', 'IPyKernel'),
    ],
    'Optional (for advanced features)': [
        ('torch-geometric', 'torch_geometric', 'PyTorch Geometric'),
    ]
}

all_installed = True

for category, pkgs in packages.items():
    print(f"\n{category}:")
    for pip_name, import_name, display_name in pkgs:
        installed, status = check_package(pip_name, import_name)
        print(f"  {status} {display_name}")
        if not installed:
            all_installed = False
            if category != 'Optional (for advanced features)':
                print(f"      Install with: pip install {pip_name}")

print("\n" + "="*70)
if all_installed:
    print("✅ ALL REQUIRED PACKAGES INSTALLED")
else:
    print("⚠ SOME PACKAGES MISSING")
    print("\nTo install missing packages, run:")
    print("  pip install -r requirements.txt")

print("\n" + "="*70)
print("PYTHON VERSION")
print("="*70)
print(f"  Python: {sys.version}")

# Check CUDA availability
try:
    import torch
    print("\n" + "="*70)
    print("PYTORCH CONFIGURATION")
    print("="*70)
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  GPU device: {torch.cuda.get_device_name(0)}")
    else:
        print("  Running on CPU (this is fine for testing)")
except:
    pass

print("\n" + "="*70)
print("Note: PyTorch Geometric is optional but recommended for")
print("advanced graph neural network features.")
print("The current implementation works without it.")
print("="*70)
