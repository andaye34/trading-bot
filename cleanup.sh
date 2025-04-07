#!/bin/bash

# ===== CLEANUP SCRIPT FOR UBUNTU VM (TRADING-BOT) =====
# WARNING: This will delete cached files, swap file, ISO, and more to free space.

# Confirm before continuing
read -p "This will permanently delete large unnecessary files. Continue? (y/n): " CONFIRM
if [[ $CONFIRM != "y" ]]; then
  echo "Aborted."
  exit 1
fi

# Delete Ubuntu ISO (if exists)
echo "Deleting Ubuntu ISO..."
sudo rm -f /mnt/shared/ubuntu-24.04.1-desktop-amd64.iso

# Remove pip, npm, and other cached files
echo "Deleting user cache files..."
rm -rf ~/.cache/*

# Delete swap file (optional)
echo "Disabling and deleting swap.img..."
sudo swapoff /swap.img 2>/dev/null
sudo rm -f /swap.img

# Clean node_modules (you can reinstall later)
echo "Deleting node_modules..."
rm -rf ~/trading-bot/frontend/node_modules

# Clean Python __pycache__ and compiled files
echo "Removing Python __pycache__ and .pyc files..."
find ~/trading-bot -type d -name "__pycache__" -exec rm -r {} +
find ~/trading-bot -type f -name "*.pyc" -delete

# Display disk usage after cleanup
echo "\nDisk usage after cleanup:"
df -h /


