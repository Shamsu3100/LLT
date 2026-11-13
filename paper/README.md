# Papers

Two drafts, matching the article plan: one main article and one earlier
workshop paper.

| File | Article | Venue | Class | Length |
|---|---|---|---|---|
| `main.tex` | 1 (main) | ACM TiiS | `acmart`, `acmsmall` | 9 pp |
| `workshop.tex` | 2 (workshop) | LLM-assisted constraint acquisition workshop | `acmart`, `sigconf` | 2 pp |

Both share `references.bib`.

## Build

```bash
latexmk main.tex
```

```bash
latexmk workshop.tex
```

`.latexmkrc` forces **XeLaTeX**. Do not switch to `pdflatex` on this machine:
`acmart` loads `libertine`, which loads the old `fontaxes` shim, which
redefines `\figureversions` against a kernel that already provides it. pdfLaTeX
dies with `Undefined control sequence ... figureversions_text_str`. XeLaTeX
takes a different font path and is unaffected.

Clean:

```bash
latexmk -C
```

## Draft status

**Sections that report measurements carry their heading and nothing else.** No
study has been run, so there is nothing to put under them. Two macros keep that
visible:

- `\result{...}` typesets as **[RESULT: ...]** in bold small caps. Every
  quantity that must come from data is wrapped in it, so no figure anywhere in
  these files can be read as a finding.
- `\awaitingdata{...}` marks a section whose content waits on data.

Written from what exists: introduction, related work, system description,
both study designs, and threats to validity — the structural threats hold
whatever the data turns out to be. Headers only: results, discussion,
conclusion.

## Before submitting

1. **Check every `VERIFY` note in `references.bib`.** Those fields came from
   secondary sources and have not been checked against the publisher's record.
   A wrong page range is a referee's first impression.
2. **Confirm the workshop venue is non-archival**, or permits a later journal
   version. An archival workshop paper would complicate the main article.
3. **Confirm SCHEDBench's venue.** Search results place it in Findings of EMNLP
   2026; the preprint itself states none.
4. Remove `anonymous` and `review` from the `\documentclass` options for a
   camera-ready version, and fill in the author block.
5. Replace every `\result{}` with the measured value, and delete the macro
   once none remain. Grep for it: a leftover placeholder in a submitted paper
   is unrecoverable.

```bash
grep -c 'result{' main.tex workshop.tex
```

## No templates were supplied

The article plan names ACM TiiS, so these use `acmart`, ACM's own class, in
the journal and conference formats respectively. No example Q1 papers were
present in the repository, so the structure follows the conventions of the
venue and of the nearest work, not a specific supplied exemplar. If you have
particular papers whose structure you want matched, add them and the section
ordering can be reworked against them.
