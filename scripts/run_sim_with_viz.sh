#!/usr/bin/env bash
# Runs a scenario via run_sim.py while follow_ego_cam.py keeps a windowed
# CARLA server's spectator snapped to the ego's camera transform, so the
# window shows the ego's point of view live. Requires the server to already
# be running windowed (./launch_carla.sh -windowed -ResX=1000 -ResY=600).
#
# Usage: same args as run_sim.py, e.g.:
#   ./scripts/run_sim_with_viz.sh --config config/nominal.json
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

python3 scripts/follow_ego_cam.py &
FOLLOW_PID=$!
trap 'kill "$FOLLOW_PID" 2>/dev/null' EXIT

python scripts/run_sim.py "$@"
