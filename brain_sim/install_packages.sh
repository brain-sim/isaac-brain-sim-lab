#!/bin/bash
# Uninstall and reinstall all brain_sim packages in editable mode

echo "Uninstalling existing brain_sim packages..."

# Uninstall packages first
for pkg in source/brain_sim_*; do
    if [ -d "$pkg" ]; then
        pkg_name=$(basename "$pkg")
        echo "Uninstalling $pkg_name..."
        python -m pip uninstall -y "$pkg_name" 2>/dev/null || echo "  $pkg_name not installed, skipping..."
    fi
done

echo ""
echo "Installing brain_sim packages..."

# Install packages
for pkg in source/brain_sim_*; do
    if [ -d "$pkg" ]; then
        echo "Installing $pkg..."
        python -m pip install -e "$pkg"
    fi
done

echo "All packages reinstalled!"
