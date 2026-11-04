"""An uploaded video must not silently receive a default office inventory."""
import os
from pathlib import Path
import subprocess

import pytest


@pytest.mark.parametrize("success", [False, True])
def test_required_inventory_controls_pipeline_exit(tmp_path: Path, success: bool):
    root = Path(__file__).resolve().parents[1]
    detections = tmp_path / "detections"
    command = r'''set -euo pipefail
source "$1" capture video "$2/out"
DETECTIONS="$2/detections"
fake_inventory() {
  if [ "$INVENTORY_OK" = yes ]; then
    printf 'bucket\ndoor\n' > "$DETECTIONS/prompts.txt"
  else
    return 1
  fi
}
choose_prompts fake_inventory
printf 'SEGMENTATION_REACHED:%s\n' "${PROMPTS[*]}"
'''
    env = {**os.environ, "PIVOT_REQUIRE_INVENTORY": "1", "PIVOT_SCRATCH": str(tmp_path),
           "PIVOT_LOGS": str(tmp_path / "logs"), "INVENTORY_OK": "yes" if success else "no"}
    result = subprocess.run(["bash", "-c", command, "inventory-test",
                             str(root / "scripts/pipeline_common.sh"), str(tmp_path)],
                            env=env, capture_output=True, text=True, check=False)
    assert (result.returncode == 0) is success
    if success:
        assert "SEGMENTATION_REACHED:bucket door" in result.stdout
    else:
        assert "SEGMENTATION_REACHED" not in result.stdout
        assert "refusing to substitute" in result.stderr
        assert not (detections / "prompts.txt").exists()
