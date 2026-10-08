#!/bin/sh
# Install a symlink to this checkout; moving it means running this script again.
set -eu
repo_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
install_dir=${TY_BIN_DIR:-"$HOME/.local/bin"}
command -v python3 >/dev/null 2>&1 || { echo 'Python 3.11+ is required.' >&2; exit 1; }
python3 -c 'import sys; assert sys.version_info >= (3, 11), "Python 3.11+ is required"'
mkdir -p "$install_dir"
if [ -e "$install_dir/ty" ] && [ ! -L "$install_dir/ty" ]; then
    echo "$install_dir/ty already exists. Set TY_BIN_DIR to another directory." >&2
    exit 1
fi
chmod +x "$repo_dir/ty.py"
ln -sfn "$repo_dir/ty.py" "$install_dir/ty"
printf 'Installed ty → %s/ty.py\n' "$repo_dir"
case ":$PATH:" in
  *":$install_dir:"*) ;;
  *) printf 'Add %s to PATH to run ty from any directory.\n' "$install_dir" ;;
esac
