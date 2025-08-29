#!/bin/bash

# Script to create VS Code settings for IsaacLab source code analysis
# This script creates .vscode/settings.json with Python analysis extra paths

# Get the workspace folder (current directory)
WORKSPACE_FOLDER=$(pwd)

# Create .vscode directory if it doesn't exist
mkdir -p .vscode

# Create settings.json with Python analysis extra paths
cat > .vscode/settings.json << EOF
{
    "python.analysis.extraPaths": [
        "\${workspaceFolder}/source/isaaclab_rl",
        "\${workspaceFolder}/source/isaaclab_mimic",
        "\${workspaceFolder}/source/isaaclab",
        "\${workspaceFolder}/source/isaaclab_assets",
        "\${workspaceFolder}/source/isaaclab_tasks"
    ]
}
EOF

echo "✅ Created .vscode/settings.json with IsaacLab Python analysis paths"
echo "📁 Location: $WORKSPACE_FOLDER/.vscode/settings.json"
echo ""
echo "The following paths have been added to python.analysis.extraPaths:"
echo "  - \${workspaceFolder}/source/isaaclab_rl"
echo "  - \${workspaceFolder}/source/isaaclab_mimic"
echo "  - \${workspaceFolder}/source/isaaclab"
echo "  - \${workspaceFolder}/source/isaaclab_assets"
echo "  - \${workspaceFolder}/source/isaaclab_tasks"
echo ""
echo "ℹ️  Restart VS Code or reload the window for the settings to take effect."
