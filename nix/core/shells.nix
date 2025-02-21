{
  cell,
  inputs,
}: let
  inherit (cell.lib) pkgs;
in {
  default = inputs.std.lib.dev.mkShell {
    name = "shell";
    packages = [
      pkgs.tex
    ];

    imports = [
      inputs.std.std.devshellProfiles.default
    ];

    nixago = [
      cell.configs.conform
      cell.configs.editorconfig
      cell.configs.lefthook
      cell.configs.treefmt
      cell.configs.vscodeSettings
      cell.configs.vscodeExtensions
    ];

    commands = [];
  };
}
