#!/bin/bash
# Block forbidden file operations BEFORE they happen

INPUT=$(cat)
TOOL=$(echo "$INPUT" | jq -r '.tool_name')
TOOL_INPUT=$(echo "$INPUT" | jq -r '.tool_input')

# Only check Write and Edit tools
if [[ "$TOOL" != "Write" && "$TOOL" != "Edit" ]]; then
    exit 0
fi

# Extract file path
FILE_PATH=$(echo "$TOOL_INPUT" | jq -r '.file_path // .path // empty')

if [ -z "$FILE_PATH" ]; then
    exit 0
fi

# Forbidden file patterns
FORBIDDEN_PATTERNS=(
    "TODO\.md"
    "NOTES\.md"
    "SESSION.*\.md"
    "BACKLOG\.md"
    "SCRATCH\.md"
    "TASKS\.md"
    "FEATURE_IDEAS\.md"
    "RESEARCH\.md"
)

# Check each pattern
for pattern in "${FORBIDDEN_PATTERNS[@]}"; do
    if echo "$FILE_PATH" | grep -qE "$pattern"; then
        # Extract base name for better error message
        BASENAME=$(basename "$FILE_PATH")

        # Suggest appropriate IdlerGear alternative
        case "$BASENAME" in
            TODO.md|TASKS.md|BACKLOG.md)
                ALTERNATIVE="idlergear task create \"...\""
                ;;
            NOTES.md|SCRATCH.md|SESSION*.md)
                ALTERNATIVE="idlergear note create \"...\""
                ;;
            FEATURE_IDEAS.md)
                ALTERNATIVE="idlergear note create \"...\" --tag idea"
                ;;
            RESEARCH.md)
                ALTERNATIVE="idlergear note create \"...\" --tag explore"
                ;;
            *)
                ALTERNATIVE="idlergear task create \"...\" or idlergear note create \"...\""
                ;;
        esac

        cat <<EOF >&2
❌ FORBIDDEN FILE: $FILE_PATH

IdlerGear projects use commands, not markdown files, for knowledge management.

Instead of creating $BASENAME, use:
  $ALTERNATIVE

Why? Knowledge in IdlerGear is:
  • Queryable (idlergear search)
  • Linkable (tasks ↔ commits ↔ notes)
  • Synced with GitHub (optional)
  • Available to all AI sessions via MCP

See CLAUDE.md for full guidelines.
EOF
        exit 2  # Exit code 2 = blocking error
    fi
done

exit 0
