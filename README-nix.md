## Prereqs

1. Install Nix: https://nixos.org/download.html#download-nix
2. Enable Flakes: https://nixos.wiki/wiki/Flakes#Permanent

Optional:
3. Install Cachix: https://docs.cachix.org/installation
4. Enable cuda-maintainers cache: https://nixos.wiki/wiki/CUDA
5. `nix flake update` to a recent release to take advantage of caching

Note: Updating to a bleeding edge version could break torch/CUDA integration.
However, using the pinned version could result in cache misses for other packages
triggering recompilation.

## Usage
0. Add/remove dependencies from flake.nix as needed
1. Enter a development shell: `nix develop`
2. (first time) Run the setup script: `run-me-first`
3. Start jupyter: `jupyter-lab`
4. Install other requirements: `pip install -r requirements.txt`
5. Install requirements for your current unit
6. Open unit#.py in jupyter by selecting "Open With > Notebook"

If you install packages via pip, it's a good idea to track them in a
`requirements.txt` file.  To guarantee an (reproducibly) exact version
of a package, you should have nix download and install the wheel
in `flake.nix`.  See the arviz and jupytext entries.
