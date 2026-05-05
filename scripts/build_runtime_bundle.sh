#!/usr/bin/env bash
set -euo pipefail
export COPYFILE_DISABLE=1

INCLUDE_TILES=1
for arg in "$@"; do
  case "$arg" in
    --skip-tiles)
      INCLUDE_TILES=0
      ;;
    -h|--help)
      cat <<'EOF'
Usage: ./scripts/build_runtime_bundle.sh [--skip-tiles]

Options:
  --skip-tiles   Do not include the tiles/ directory in the runtime bundle.
EOF
      exit 0
      ;;
    *)
      echo "Unknown argument: $arg" >&2
      exit 1
      ;;
  esac
done

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIST_DIR="${ROOT_DIR}/dist"
STAGE_DIR="${DIST_DIR}/traffic-safety-runtime"
BUNDLE_PATH="${DIST_DIR}/traffic-safety-runtime.tgz"

rm -rf "${STAGE_DIR}" "${BUNDLE_PATH}"
mkdir -p \
  "${STAGE_DIR}/data/processed/weather" \
  "${STAGE_DIR}/data/processed/segments" \
  "${DIST_DIR}"

copy_if_exists() {
  local source_path="$1"
  local target_path="$2"
  if [[ -e "${source_path}" ]]; then
    mkdir -p "$(dirname "${target_path}")"
    cp -R "${source_path}" "${target_path}"
  fi
}

copy_if_exists "${ROOT_DIR}/requirements.txt" "${STAGE_DIR}/requirements.txt"
copy_if_exists "${ROOT_DIR}/README.md" "${STAGE_DIR}/README.md"
copy_if_exists "${ROOT_DIR}/src" "${STAGE_DIR}/src"
copy_if_exists "${ROOT_DIR}/scripts" "${STAGE_DIR}/scripts"
copy_if_exists "${ROOT_DIR}/models" "${STAGE_DIR}/models"
if [[ "${INCLUDE_TILES}" == "1" ]]; then
  copy_if_exists "${ROOT_DIR}/tiles" "${STAGE_DIR}/tiles"
fi

copy_if_exists \
  "${ROOT_DIR}/data/processed/weather/representative_stations.csv.gz" \
  "${STAGE_DIR}/data/processed/weather/representative_stations.csv.gz"
copy_if_exists \
  "${ROOT_DIR}/data/processed/segments/road_segments.parquet" \
  "${STAGE_DIR}/data/processed/segments/road_segments.parquet"
copy_if_exists \
  "${ROOT_DIR}/data/processed/segments/active_road_segments.parquet" \
  "${STAGE_DIR}/data/processed/segments/active_road_segments.parquet"
copy_if_exists \
  "${ROOT_DIR}/data/contact_submissions.jsonl" \
  "${STAGE_DIR}/data/contact_submissions.jsonl"

find "${STAGE_DIR}" \( -name '.DS_Store' -o -name '._*' \) -delete
tar -C "${STAGE_DIR}" -czf "${BUNDLE_PATH}" .

echo "Created ${BUNDLE_PATH}"
if [[ "${INCLUDE_TILES}" == "0" ]]; then
  echo "Bundle excludes tiles/"
fi
du -sh "${BUNDLE_PATH}"
