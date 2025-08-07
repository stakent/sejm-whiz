#!/bin/bash
# Shell completion installer for sejm-whiz CLI

set -e

CLI_NAME="sejm-whiz-cli"
COMPLETION_DIR=""
SHELL_TYPE=""

# Detect shell type
if [ -n "$ZSH_VERSION" ]; then
    SHELL_TYPE="zsh"
    COMPLETION_DIR="${HOME}/.zsh/completions"
elif [ -n "$BASH_VERSION" ]; then
    SHELL_TYPE="bash"
    COMPLETION_DIR="${HOME}/.bash_completion.d"
else
    echo "❌ Unsupported shell. Only bash and zsh are supported."
    exit 1
fi

echo "🐚 Detected shell: $SHELL_TYPE"
echo "📁 Completion directory: $COMPLETION_DIR"

# Create completion directory if it doesn't exist
mkdir -p "$COMPLETION_DIR"

# Generate completion script
echo "⚙️ Generating completion script..."
if [ "$SHELL_TYPE" = "zsh" ]; then
    # Generate zsh completion
    uv run python sejm-whiz-cli.py --install-completion zsh > "$COMPLETION_DIR/_$CLI_NAME"
    echo "✅ Zsh completion installed to: $COMPLETION_DIR/_$CLI_NAME"

    # Add to fpath if not already there
    if ! grep -q "$COMPLETION_DIR" ~/.zshrc 2>/dev/null; then
        echo "" >> ~/.zshrc
        echo "# sejm-whiz completion" >> ~/.zshrc
        echo "fpath=(~/.zsh/completions \$fpath)" >> ~/.zshrc
        echo "autoload -U compinit && compinit" >> ~/.zshrc
        echo "📝 Added completion setup to ~/.zshrc"
    fi

elif [ "$SHELL_TYPE" = "bash" ]; then
    # Generate bash completion
    uv run python sejm-whiz-cli.py --install-completion bash > "$COMPLETION_DIR/$CLI_NAME"
    echo "✅ Bash completion installed to: $COMPLETION_DIR/$CLI_NAME"

    # Source completion in .bashrc if not already there
    if ! grep -q "$COMPLETION_DIR/$CLI_NAME" ~/.bashrc 2>/dev/null; then
        echo "" >> ~/.bashrc
        echo "# sejm-whiz completion" >> ~/.bashrc
        echo "source $COMPLETION_DIR/$CLI_NAME" >> ~/.bashrc
        echo "📝 Added completion setup to ~/.bashrc"
    fi
fi

echo ""
echo "🎉 Shell completion installed successfully!"
echo ""
echo "To activate completion, either:"
echo "  1. Restart your shell"
echo "  2. Run: source ~/.${SHELL_TYPE}rc"
echo ""
echo "📋 Available commands to try:"
echo "  $CLI_NAME <TAB><TAB>        # Show all commands"
echo "  $CLI_NAME system <TAB>     # Show system subcommands"
echo "  $CLI_NAME db <TAB>         # Show database subcommands"
echo ""
