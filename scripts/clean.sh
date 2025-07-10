#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

DRY_RUN=false
VERBOSE=false

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Clean up all files and folders listed in .gitignore"
    echo ""
    echo "Options:"
    echo "  -d, --dry-run    Show what would be removed without actually removing"
    echo "  -v, --verbose    Verbose output"
    echo "  -h, --help       Show this help message"
}

# System directories to never remove
SYSTEM_DIRS=("/tmp" "/var" "/usr" "/bin" "/sbin" "/etc" "/home" "/root" "/")

is_system_dir() {
    local path="$1"
    # Convert to absolute path
    local abs_path=$(realpath -m "$path" 2>/dev/null || echo "$path")
    
    for sys_dir in "${SYSTEM_DIRS[@]}"; do
        if [[ "$abs_path" == "$sys_dir" ]] || [[ "$abs_path" == "$sys_dir"/* ]]; then
            return 0
        fi
    done
    return 1
}

while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--dry-run)
            DRY_RUN=true
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

cd "$ROOT_DIR"

if [[ ! -f .gitignore ]]; then
    echo "Error: .gitignore file not found in $ROOT_DIR"
    exit 1
fi

if [[ "$DRY_RUN" == true ]]; then
    echo "DRY RUN - showing what would be removed:"
    echo "========================================"
else
    echo "Cleaning up gitignored files and folders..."
    echo "==========================================="
fi

remove_gitignored_files() {
    local gitignore_file="$1"
    local dry_run="$2"
    local verbose="$3"
    local removed_count=0

    if [[ ! -f "$gitignore_file" ]]; then
        echo "Gitignore file not found: $gitignore_file"
        return 1
    fi

    echo "Processing gitignore patterns from: $gitignore_file"
    echo

    while IFS= read -r line; do
        line=$(echo "$line" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
        
        if [[ -z "$line" || "$line" =~ ^# ]]; then
            continue
        fi

        local pattern="$line"
        
        if is_system_dir "$pattern"; then
            if [[ "$verbose" == "true" ]]; then
                echo "SKIP $pattern (system directory)"
            fi
            continue
        fi

        # Handle different types of patterns
        if [[ "$pattern" == */ ]]; then
            # Directory patterns (e.g., node_modules/, .expo/)
            local dir_name="${pattern%/}"
            local found_dirs=()
            
            # Find all directories with this name
            while IFS= read -r -d '' dir; do
                found_dirs+=("$dir")
            done < <(find . -name "$dir_name" -type d -print0 2>/dev/null)
            
            if [[ ${#found_dirs[@]} -eq 0 ]]; then
                if [[ "$verbose" == "true" ]]; then
                    echo "SKIP $pattern (not found)"
                fi
            else
                for dir in "${found_dirs[@]}"; do
                    if [[ "$dry_run" == "true" ]]; then
                        echo "WOULD REMOVE: $dir"
                    else
                        if [[ "$verbose" == "true" ]]; then
                            echo "REMOVING: $dir"
                        fi
                        rm -rf "$dir"
                        if [[ $? -eq 0 ]]; then
                            ((removed_count++))
                        fi
                    fi
                done
            fi
        elif [[ "$pattern" == *.* && "$pattern" != */* ]]; then
            # File extension patterns (e.g., *.log, *.tmp)
            local found_files=()
            
            while IFS= read -r -d '' file; do
                found_files+=("$file")
            done < <(find . -name "$pattern" -type f -print0 2>/dev/null)
            
            if [[ ${#found_files[@]} -eq 0 ]]; then
                if [[ "$verbose" == "true" ]]; then
                    echo "SKIP $pattern (not found)"
                fi
            else
                for file in "${found_files[@]}"; do
                    if [[ "$dry_run" == "true" ]]; then
                        echo "WOULD REMOVE: $file"
                    else
                        if [[ "$verbose" == "true" ]]; then
                            echo "REMOVING: $file"
                        fi
                        rm -f "$file"
                        if [[ $? -eq 0 ]]; then
                            ((removed_count++))
                        fi
                    fi
                done
            fi
        else
            # Direct file/directory patterns
            if [[ -e "$pattern" ]]; then
                if [[ "$dry_run" == "true" ]]; then
                    echo "WOULD REMOVE: $pattern"
                else
                    if [[ "$verbose" == "true" ]]; then
                        echo "REMOVING: $pattern"
                    fi
                    rm -rf "$pattern"
                    if [[ $? -eq 0 ]]; then
                        ((removed_count++))
                    fi
                fi
            else
                if [[ "$verbose" == "true" ]]; then
                    echo "SKIP $pattern (not found)"
                fi
            fi
        fi
        
    done < "$gitignore_file"

    echo
    echo "Removed: $removed_count items"
}

remove_gitignored_files .gitignore "$DRY_RUN" "$VERBOSE"

echo "========================================"
if [[ "$DRY_RUN" == true ]]; then
    echo "DRY RUN completed"
else
    echo "Cleanup completed"
fi

if [[ "$DRY_RUN" == true ]]; then
    echo ""
    echo "Run without --dry-run to actually remove these files"
fi 