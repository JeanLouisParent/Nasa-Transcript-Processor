# Rapport complet - AS11_TEC (run complet --clean)

## Portee

- PDF: `input/AS11_TEC.PDF`
- Execution: pages par batches de 20 en mode `--clean` pour le premier batch, puis sans `--clean`.
- Mode OCR: `ocr_postprocess = "hybrid"` (correct + signal), prompt OCR "plain" par defaut en postprocess.
- Comparaison: `assets/a11tec.csv` (transcription manuelle).

Commandes utilisees (extrait):

```bash
./venv/bin/python main.py process AS11_TEC.PDF --pages 1-20 --clean --timing
./venv/bin/python main.py process AS11_TEC.PDF --pages 21-40 --timing
...
./venv/bin/python main.py process AS11_TEC.PDF --pages 621-626 --timing
```

Sortie console complete avec timings: `pipeline_run_after.log`.

## Ameliorations appliquees (globales, non Apollo-specifiques)

1) OCR prompt
- Ajout d'une consigne explicite sur les 3 colonnes (timestamp / speaker / texte) pour forcer la lecture de la partie droite.
- Objectif: limiter les lignes tronquees ou "timestamp only".

2) Parser speaker
- Support des speakers multi-tokens (ex: `SWIM 1`, `PRESIDENT NIXON`, `CMP/LMP`) via regex et extraction sur ligne prefixee par timestamp.

3) Timestamp corrector
- Tolerance aux petits retours en arriere (OCR slips) avant correction monotone.
- Les grosses incoherences restent corrigees.
- Nouveau tag possible: `timestamp_correction = \"out_of_order\"` (timestamp conserve).

4) Encodage image OCR
- Qualite JPEG augmentee (95) pour preserver les caracteres fins.

5) Robustesse OCR
- OCR vide ne casse plus le pipeline (pas d'exception). Le parsing accepte une page vide.

6) OCR colonne texte (droite)
- Deuxieme passe optionnelle pour remplir les blocs `comm` sans texte.
- Filtrage des lignes speaker/location/page/tape avant injection.

7) Parser blocs & footer
- Reconstruction des sequences "timestamp-only" + colonne speaker/texte (mode timestamp list).
- Footer canonicalise sur la ligne asterisks standard.

## Resultats de comparaison vs CSV

### Avant modifications (run precedent)
- `matched_rows`: 6,372 / 8,460
- `missing_rows`: 2,088
- `avg_similarity`: 0.962
- `median_similarity`: 1.000

### Apres modifications (run actuel)
- `matched_rows`: 7,172 / 8,460
- `missing_rows`: 1,288
- `avg_similarity`: 0.966
- `median_similarity`: 1.000
- `timestamp matched`: 7,397 / 8,460
- `comm_blocks`: 8,465
- `empty_text` (comm sans texte): 284
- `missing_speaker` (comm sans speaker): 393

Amelioration nette sur les correspondances timestamp+speaker et sur la couverture globale.

## Timing (par page, moyenne/mediane/min/max)

Deduplique sur 626 pages (derniere occurence par page):

- `extract`: avg 0.051s / med 0.050s / min 0.045s / max 0.067s
- `process`: avg 0.252s / med 0.220s / min 0.053s / max 3.845s
- `output`: avg 0.067s / med 0.067s / min 0.041s / max 0.096s
- `ocr`: avg 5.455s / med 5.740s / min 1.531s / max 36.262s
- `classify`: avg 7.335s / med 7.633s / min 2.302s / max 43.452s
- `ocr_total`: avg 12.790s / med 13.366s / min 3.844s / max 79.710s

## Observations principales

- Les pages "landing" (ex: around `04 06 44 xx`) recuperent maintenant le texte a droite; l'OCR brut contient les lignes completes.
- Il reste des blocs `comm` vides sur certaines pages, probablement dues a:
  - OCR qui rate encore la partie texte a droite sur quelques pages difficiles,
  - lignes qui ne contiennent effectivement pas de contenu (ou tres peu visible).
- Des ecarts persistent sur des timestamps courts (debut de mission), souvent associes a:
  - confusions OCR de digits,
  - timestamp out-of-order qui sont maintenant conserves (mieux pour la verite terrain).

## Restants (actions possibles)

- Renforcer la detection des textes tres faibles (p. ex. contraste adaptatif, ajustements CLAHE).
- Revoir la logique de correction des speakers manquants lorsque la ligne est en uppercase et contient 2 tokens.

## Fichiers cles

- Log console (timings): `pipeline_run_after.log`
- Output JSON: `output/AS11_TEC/Page_*/AS11_TEC_page_*.json`
- Config prompts: `config/prompts.toml`
- Prompts documentes: `docs/PROMPTS.md`
