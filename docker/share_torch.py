"""Share only the pinned CUDA stack with an otherwise isolated MapAnything venv."""
from pathlib import Path
import site
import sys

source = Path(site.getsitepackages()[0])
target = Path('/opt/map-env/lib') / ('python' + '.'.join(map(str, sys.version_info[:2]))) / 'site-packages'
# No numpy/scipy/SAM metadata is inherited. These shared packages remain pinned
# in the image; installing another torch build is forbidden by both constraints.
names = ('torch', 'torchgen', 'functorch', 'torchvision', 'torchaudio', 'triton', 'nvidia')
for path in source.iterdir():
    if path.name in names or (path.name.startswith('nvidia_') and path.name.endswith('.dist-info')) or any(path.name.startswith(name + '-') and path.name.endswith('.dist-info')
                                 for name in ('torch', 'torchvision', 'torchaudio', 'triton', 'nvidia')):
        (target / path.name).symlink_to(path)
