## Prereqs

1. Install Nix: https://nixos.org/download.html#download-nix
2. Enable Flakes: https://nixos.wiki/wiki/Flakes#Permanent
3. Install Cachix: https://docs.cachix.org/installation
4. Enable cuda-maintainers cache: https://nixos.wiki/wiki/CUDA

## Usage
0. Add/remove dependencies from flake.nix as needed
1. Enter a development shell: `nix develop`
2. (first time) Run the setup script: `run-me-first`
3. Start jupyter: `jupyter-lab`
4. (optional) Install other requirements: `pip install ...`

If you install packages via pip, it's a good idea to track them in a
`requirements.txt` file.  To guarantee an (reproducibly) exact version
of a package, you should have nix download and install the wheel
in `flake.nix`.  See the arviz and jupytext entries.
