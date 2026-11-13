# XeLaTeX, not pdfLaTeX.
#
# acmart pulls in libertine, which loads the old `fontaxes` shim. That shim
# redefines \figureversions, which newer LaTeX kernels provide themselves, and
# pdflatex dies on the collision:
#
#   ! Undefined control sequence.
#   l.435 ...figureversions_text_str { OsF } { TOsF }
#
# XeLaTeX takes a different font path and is unaffected. If a future TeX
# update fixes fontaxes, `latexmk -pdf` will work again.
$pdf_mode = 5;      # 5 = xelatex
$postscript_mode = 0;
$dvi_mode = 0;

$clean_ext = "bbl nav out snm synctex.gz run.xml bcf";
