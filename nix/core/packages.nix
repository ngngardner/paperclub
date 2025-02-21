{ cell, inputs }:
let
  inherit (cell.lib) pkgs;
in
{
  annotatedBibliography = pkgs.annotatedBibliography;
}
