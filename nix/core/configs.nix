{
  inputs,
  cell,
}: let
  inherit (inputs.std.data) configs;
  inherit (inputs.std.lib.dev) mkNixago;
  inherit (cell.lib) pkgs;
in {
  conform = (mkNixago configs.conform) {
    data = {inherit (inputs) cells;};
  };

  editorconfig = (mkNixago configs.editorconfig) {
    data = {};
  };

  lefthook = (mkNixago configs.lefthook) {
    data = {};
  };

  treefmt = (mkNixago configs.treefmt) {
    packages = [
      pkgs.tex-fmt
      pkgs.bibtex-tidy
    ];
    data = {
      formatter = {
        latex = {
          command = "${pkgs.tex-fmt}/bin/tex-fmt";
          includes = ["*.tex"];
        };
        bibtex = {
          command = "${pkgs.bibtex-tidy}/bin/bibtex-tidy";
          includes = ["*.bib"];
        };
      };
    };
  };

  vscodeSettings = (
    mkNixago {
      # packages = [
      #   pkgs.treefmt
      # ];
      data = {
        "emeraldwalk.runonsave" = {
          autoClearConsole = true;
          commands = [
            {
              match = ".*";
              cmd = "${pkgs.treefmt}/bin/treefmt \${file}";
            }
          ];
        };
        "files.exclude" = {
          "**/.bin" = true;
          "**/.cache" = true;
          "**/.direnv" = true;
          "**/.config" = true;
          "**/.local" = true;
          "**/.run" = true;
        };
        "nixEnvSelector.nixFile" = "\${workspaceRoot}/flake.nix";
        "editor.formatOnSave" = false;
      };
      output = ".vscode/settings.json";
      format = "json";
    }
  );

  vscodeExtensions = (
    mkNixago {
      data = {
        recommendations = [
          "emeraldwalk.runonsave"
          "arrterian.nix-env-selector"
        ];
      };
      output = ".vscode/extensions.json";
      format = "json";
    }
  );
}
