{ cell, inputs }:
let
  inherit (inputs) nixpkgs self;
in
{
  default = final: prev: {
    tex = prev.texlive.combine {
      inherit (prev.texlive)
        scheme-medium
        latexmk
        latexindent

        hanging
        biblatex
        ;
    };

    annotatedBibliography = nixpkgs.stdenv.mkDerivation {
      name = "annotated-bibliography";
      src = "${self}/tex/annotated_bibliography";

      buildInputs = with final; [
        tex
        biber
      ];

      buildPhase = ''
        pdflatex main.tex # First LaTeX pass: process main.tex and create auxiliary files.
        biber main # Process the bibliography using Biber, generating a .bbl file.
        pdflatex main.tex # Second LaTeX pass: insert the bibliography and resolve some cross-references.
        pdflatex main.tex # Third LaTeX pass: finalize all cross-references for a consistent document.
      '';

      installPhase = ''
        mkdir -p $out/share/annotated-bibliography
        cp main.pdf $out/share/annotated-bibliography/
      '';
    };
  };
}
