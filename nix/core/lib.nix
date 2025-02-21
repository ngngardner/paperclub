{
  inputs,
  cell,
}:
{
  pkgs = inputs.nixpkgs.appendOverlays [
    cell.overlays.default
  ];
}
