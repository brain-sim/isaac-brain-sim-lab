#!/bin/bash

# Script to add IsaacLab source paths to VS Code Python analysis extra paths

ISAACLAB_PATH="/home/yihao/Downloads/software/IsaacLab"

mkdir -p .vscode

cat > .vscode/settings.json << EOF
{
    "python.analysis.extraPaths": [
        "${ISAACLAB_PATH}/source/isaaclab_rl",
        "${ISAACLAB_PATH}/source/isaaclab_mimic",
        "${ISAACLAB_PATH}/source/isaaclab",
        "${ISAACLAB_PATH}/source/isaaclab_assets",
        "${ISAACLAB_PATH}/source/isaaclab_tasks"
    ]
}
EOF

echo "✅ Created .vscode/settings.json with IsaacLab Python analysis paths"
echo "📁 Location: $(pwd)/.vscode/settings.json"
echo ""
echo "The following absolute paths have been added to python.analysis.extraPaths:"
echo "  - ${ISAACLAB_PATH}/source/isaaclab_rl"
echo "  - ${ISAACLAB_PATH}/source/isaaclab_mimic"
echo "  - ${ISAACLAB_PATH}/source/isaaclab"
echo "  - ${ISAACLAB_PATH}/source/isaaclab_assets"
echo "  - ${ISAACLAB_PATH}/source/isaaclab_tasks"
echo ""
echo "ℹ️  Restart VS Code or reload the window for the settings to take effect."