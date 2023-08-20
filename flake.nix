{
  inputs = {
    nixpkgs.url = "nixpkgs";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config = {
            allowUnfree = true;
            cudaSupport = true;
          };
        };
        py-pkgs = pkgs.python310Packages;
        setup-script = pkgs.writeScriptBin "run-me-first" ''
          #micromamba install --yes -f conda-requirements.txt -c conda-forge -c pytorch -c nvidia
          pip install -r requirements.txt
        '';

        jupytext = with py-pkgs; buildPythonPackage rec {
          pname = "jupytext";
          version = "1.14.7";
          format = "wheel";
          src = fetchPypi rec {
            inherit pname version format;
            sha256 = "sha256-qy/QZtzoESrJWnxPkhjBpA1s76drCTUsjQis8h2tU0k=";
            dist = python;
            python = "py3";
          };
          doCheck = false;
          propagatedBuildInputs = [
            setuptools
            mdit-py-plugins nbformat pyyaml toml markdown-it-py
          ];
        };
        pylibs = ps: with ps; [
          pip
          ale-py
          datasets
          einops
          graphviz
          gymnasium
          huggingface-hub
          ipdb
          ipywidgets
          jupyterlab
          jupyterlab_server
          jupyterlab-widgets
          jupytext
          matplotlib
          moviepy
          numpy
          opencv4
          pandas
          plotly
          seaborn
          tensorflow-bin
          torch-bin
          tqdm
          transformers
        ];

        libraries = with pkgs; [
          boost
          ffmpeg
          fluidsynth
          game-music-emu
          glib
          gtk2
          libGL
          libjpeg
          libstdcxx5
          lua51Packages.lua
          nasm
          openal
          SDL2
          stdenv.cc.cc.lib
          timidity
          wildmidi
          zlib
        ];

        packages = with pkgs; [
          (python310.withPackages pylibs)
          cmake
          curl
          gnutar
          jq
          stgit
          swig
          unzip
        ];
      in
      {
        devShell = pkgs.mkShell {
          buildInputs = packages ++ libraries;
          packages = with pkgs; [
            setup-script
          ];

          shellHook =
            ''
              export LD_LIBRARY_PATH=${pkgs.lib.makeLibraryPath libraries}:$LD_LIBRARY_PATH
              export PIP_PREFIX=$(pwd)/.mypy
              export PYTHONPATH="$PIP_PREFIX/${pkgs.python3.sitePackages}:$PYTHONPATH"
              export PATH="$PIP_PREFIX/bin:$PATH"
              export PIP_DISABLE_PIP_VERSION_CHECK=1
            '';
        };
      });
}
