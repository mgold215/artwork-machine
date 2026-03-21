#!/bin/zsh
# artwork-machine — interactive generator
# Usage: just run ./generate.sh and answer the prompts

cd "$(dirname "$0")"

echo ""
echo "── artwork-machine ──────────────────────────────"
echo ""

# Audio file — accept a drag-and-drop path
echo "Drag your audio file here (or paste the path):"
read -r AUDIO_FILE

# Strip surrounding quotes that macOS adds on drag-and-drop
AUDIO_FILE="${AUDIO_FILE//\'/}"
AUDIO_FILE="${AUDIO_FILE//\"/}"
# Strip trailing whitespace/newline
AUDIO_FILE="${AUDIO_FILE%"${AUDIO_FILE##*[![:space:]]}"}"

if [[ ! -f "$AUDIO_FILE" ]]; then
  echo "File not found: $AUDIO_FILE"
  exit 1
fi

echo ""
echo "Artist name:"
read -r ARTIST

echo ""
echo "Album / track title:"
read -r ALBUM

echo ""
echo "Skip YouTube visualizer? (saves ~80 GB of disk space) [Y/n]:"
read -r SKIP_VIZ
SKIP_VIZ="${SKIP_VIZ:-Y}"

echo ""
echo "Draft mode — faster but lower resolution? [y/N]:"
read -r DRAFT
DRAFT="${DRAFT:-N}"

# Build flags
FLAGS=""
[[ "$SKIP_VIZ" =~ ^[Yy]$ ]] && FLAGS="$FLAGS --no-visualizer"
[[ "$DRAFT"    =~ ^[Yy]$ ]] && FLAGS="$FLAGS --draft"

echo ""
echo "── Generating artwork for \"$ALBUM\" by $ARTIST ──"
echo ""

.venv/bin/artwork-machine generate "$AUDIO_FILE" \
  --artist "$ARTIST" \
  --album  "$ALBUM" \
  $FLAGS

# Open the output folder when done
OUTPUT_DIR="output/$(echo "${ARTIST}_${ALBUM}" | tr ' ' '_' | tr -cd '[:alnum:]_-' | cut -c1-80)"
[[ -d "$OUTPUT_DIR" ]] && open "$OUTPUT_DIR"
