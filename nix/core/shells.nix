{ cell, inputs }:
let
  inherit (cell.lib) pkgs;
in
{
  default = inputs.std.lib.dev.mkShell {
    name = "shell";
    packages = [
      pkgs.just
      pkgs.tex
    ];

    imports = [
      inputs.std.std.devshellProfiles.default
    ];

    commands = [ ];
  };
}
