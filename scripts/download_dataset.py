import thesis.paths as paths

from pathlib import Path

from huggingface_hub import snapshot_download

dir_path = Path(../data)
dir_path.mkdir(parents=True, exist_ok=True)

folder = snapshot_download(
    "cis-lmu/glotlid-corpus", 
    repo_type="dataset",
    local_dir_use_symlinks=False,
    allow_patterns="v3.1/*",
    local_dir=f"{dir_path}/glotlid-corpus/",
)