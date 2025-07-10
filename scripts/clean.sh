#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DRY_RUN=false
VERBOSE=false

usage() {
    echo "Usage: $0 [--dry-run] [--verbose] [--help]"
    echo "Clean up all files and folders listed in .gitignore"
}

is_protected() {
    local path="${1#./}"
    [[ "$path" =~ ^(ios|cpp|scripts)(/|$) ]] || [[ "$path" =~ ^(/tmp|/var|/usr|/bin|/sbin|/etc|/home|/root)(/|$) ]]
}

remove_item() {
    local item="$1"
    [[ -e "$item" ]] || return 0
    is_protected "$item" && { [[ "$VERBOSE" == true ]] && echo "SKIP $item (protected)"; return 0; }
    
    if [[ "$DRY_RUN" == true ]]; then
        echo "WOULD REMOVE: $item"
    else
        [[ "$VERBOSE" == true ]] && echo "REMOVING: $item"
        rm -rf "$item" && ((removed_count++))
    fi
}

while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--dry-run) DRY_RUN=true ;;
        -v|--verbose) VERBOSE=true ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1"; usage; exit 1 ;;
    esac
    shift
done

cd "$ROOT_DIR"
[[ -f .gitignore ]] || { echo "Error: .gitignore not found"; exit 1; }

[[ "$DRY_RUN" == true ]] && echo "DRY RUN - showing what would be removed:" || echo "Cleaning up gitignored files..."
echo "========================================"

removed_count=0

while IFS= read -r line; do
    line=$(echo "$line" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
    [[ -z "$line" || "$line" =~ ^# ]] && continue
    
    is_protected "$line" && { [[ "$VERBOSE" == true ]] && echo "SKIP $line (protected)"; continue; }
    
    if [[ "$line" == */ ]]; then
        find . -name "${line%/}" -type d 2>/dev/null | while read -r dir; do remove_item "$dir"; done
    elif [[ "$line" == *.* && "$line" != */* ]]; then
        find . -name "$line" -type f 2>/dev/null | while read -r file; do remove_item "$file"; done
    else
        remove_item "$line"
    fi
done < .gitignore

echo "========================================"
[[ "$DRY_RUN" == true ]] && echo "DRY RUN completed" || echo "Cleanup completed"
[[ "$DRY_RUN" == true ]] && echo "Run without --dry-run to actually remove these files" 