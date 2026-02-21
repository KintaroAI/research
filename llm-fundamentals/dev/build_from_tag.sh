#!/usr/bin/env bash
set -e

usage() {
    echo "Usage: $0 <tag-or-commit> [output-dir]"
    echo ""
    echo "Build train and generate binaries from a specific git tag or commit."
    echo ""
    echo "  tag-or-commit   Git tag (e.g. exp/00012) or commit hash"
    echo "  output-dir      Where to copy binaries (default: current directory)"
    echo ""
    echo "Examples:"
    echo "  $0 exp/00012"
    echo "  $0 exp/00012 /tmp/build"
    exit 1
}

if [ $# -lt 1 ]; then
    usage
fi

TAG="$1"
OUTPUT_DIR="${2:-.}"
REPO_ROOT="$(git rev-parse --show-toplevel)"
WORKTREE_DIR="$(mktemp -d)"

cleanup() {
    echo "Cleaning up worktree..."
    git worktree remove --force "$WORKTREE_DIR" 2>/dev/null || rm -rf "$WORKTREE_DIR"
}
trap cleanup EXIT

# Verify the tag/commit exists
if ! git rev-parse "$TAG" >/dev/null 2>&1; then
    echo "Error: '$TAG' is not a valid tag or commit"
    exit 1
fi

echo "Creating worktree at $WORKTREE_DIR for $TAG..."
git worktree add --detach "$WORKTREE_DIR" "$TAG"

echo "Building in worktree..."
make -C "$WORKTREE_DIR/llm-fundamentals/dev" train generate

mkdir -p "$OUTPUT_DIR"
cp "$WORKTREE_DIR/llm-fundamentals/dev/train" "$OUTPUT_DIR/train"
cp "$WORKTREE_DIR/llm-fundamentals/dev/generate" "$OUTPUT_DIR/generate"

echo "Binaries copied to $OUTPUT_DIR/"
echo "  $OUTPUT_DIR/train"
echo "  $OUTPUT_DIR/generate"
