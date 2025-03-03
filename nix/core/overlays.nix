{
  cell,
  inputs,
}: let
  inherit (inputs) nixpkgs self;
in {
  default = final: prev: {
    annotatedBibliography = nixpkgs.stdenv.mkDerivation {
      name = "annotated-bibliography";
      src = "${self}/typst/annotated_bibliography";

      buildInputs = with final; [
        typst
      ];

      buildPhase = ''
        typst compile --package-cache-path cache main.typ
      '';

      installPhase = ''
        mkdir -p $out/share/annotated-bibliography
        cp main.pdf $out/share/annotated-bibliography/
      '';
    };
  };
}
