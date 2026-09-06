# Full paper

`PESA_Full_Paper.docx` is the complete draft, built directly on the
official IEEE conference template (`paper-template.docx`), with all
content filled in: title, four authors with ORCIDs and affiliations,
abstract, keywords, eight numbered sections, two tables, five figures
(each embedded as a vector SVG with a PNG fallback), 17 references, and
the conference's authors'-background page on the final sheet.
`PESA_Full_Paper.pdf` is a rendered preview of the same content for
quick review without opening Word.

## How it is generated

`generate_paper.py` builds the docx programmatically:

1. Unpacks `paper-template.docx` (a version of the official IEEE template
   with fragmented text runs pre-merged, so the template's own text is
   reliably findable and replaceable).
2. Fills in the title, author block, abstract, and keywords.
3. Deletes the template's placeholder instructional content and splices
   in the actual paper body, built section by section from Python
   functions in the same file.
4. Every number quoted in the text and both tables is read live from
   `results/metrics/*.json`, the same files `src/train.py`,
   `src/anomaly_detection.py`, and the rigor-study scripts write. Nothing
   is hardcoded, so rerunning the experiments and then this script keeps
   the paper in sync.
5. Embeds the five figures from `results/figures/` (SVG + PNG pairs from
   `src/make_paper_figures.py`) directly into the document.
6. Fills the conference's authors'-background page with the four authors'
   names, roles, and research areas.
7. Repackages everything into `PESA_Full_Paper.docx`.

Reproduce with:

```
cd paper
python generate_paper.py
```

## Still to fill in before submission

- The authors'-background page (last sheet of the docx) has each
  author's name, role, and research field filled in, but email address
  and personal website were left blank since that information was not
  available when this was generated; fill those in directly in Word
  before submitting.
- `analysis.md` and `references.md` remain as the underlying source
  material and citation list the paper text was drafted from.
