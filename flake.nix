{
  description = "A very basic flake";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-24.11";
    std = {
      url = "github:divnix/std";
      inputs.nixpkgs.follows = "nixpkgs";
      inputs.devshell.url = "github:numtide/devshell";
    };
  };

  outputs =
    {
      self,
      std,
      nixpkgs,
      ...
    }@inputs:
    std.growOn
      {
        inherit inputs;
        systems = [ "x86_64-linux" ];
        cellsFrom = ./nix;
        cellBlocks = with std.blockTypes; [
          (functions "lib")
          (functions "overlays")
          (devshells "shells")
          (runnables "packages")
        ];
      }
      {
        devShells = std.harvest inputs.self [
          "core"
          "shells"
        ];
        packages = std.harvest inputs.self [
          "core"
          "packages"
        ];
      };
}
