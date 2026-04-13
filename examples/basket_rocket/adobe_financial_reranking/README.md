# BASKET/ROCKET Example Bundle

This is a compact, GitHub-friendly export of a legacy BASKET/ROCKET company snapshot
prepared for the CLIFF `basket_rocket_sec` route. The full extractor panel, raw PDFs,
and large intermediate state were intentionally excluded.

## Snapshot

- Company: ADOBE
- Route: basket_rocket_sec
- Source kind: legacy full-panel reranking snapshot
- Reward mode: financial
- Financial horizon: next_year
- Coverage years: 17
- Latest year: 2025
- Statement rows for company: 3545
- Changed statements for company: 281

## Included Files

- `extractor_snapshot.json`: compact metadata from the full-panel BASKET extraction pass
- `panel_reranking_summary.json`: sanitized full-panel financial reranking summary
- `company_summary.json`: company-level reranking and aggregate-plan snapshot
- `top_changed_examples.json`: representative changed statements for this company
- `visualizations/README.md`: GitHub-friendly index for the saved visualizations
- `visualizations/company_reranking.md`: GitHub-renderable company reranking summary
- `visualizations/aggregate_plans.md`: GitHub-renderable aggregate-plan summary
- `visualizations/company_reranking.html`: original company reranking visualizer
- `visualizations/aggregate_plans.html`: original aggregate-plan drilldown
- `visualizations/psr_drilldown.md`: GitHub-renderable PSR summary
- `visualizations/psr_drilldown.html`: original company PSR comparison page
- `images/timeline.png`: company timeline graphic referenced by the PSR drilldown

## Panel Context

- Full-panel extracted workflows: 56498
- Full-panel statement rows: 402037
- Full-panel reranked rows: 56498
- Full-panel changed rows: 11197
- Full-panel mean score gain: 0.0037798410380221396
- Company changed rate: 0.07926657263751763

## Aggregate Plan Highlights

### Top Actions
- realize_revenue: 3543
- optimize: 2246
- digitize: 1181
- expand: 562
- price: 207

### Top Edges
- optimize -> realize_revenue: 2093
- digitize -> realize_revenue: 765
- expand -> realize_revenue: 494
- digitize -> expand: 333
- price -> optimize: 122

## Representative Changed Statements

- adobe:y2008:p0014: innovate -> digitize -> optimize => innovate -> digitize -> optimize -> expand -> realize_revenue (gain=0.10864005889587636)
- adobe:y2011:p0209: acquire -> optimize => acquire -> optimize -> realize_revenue (gain=0.10174259698424237)
- adobe:y2010:p0007: innovate -> digitize => innovate -> digitize -> optimize -> expand -> realize_revenue (gain=0.09925175087922578)
- adobe:y2010:p0143: acquire -> optimize => acquire -> optimize -> realize_revenue (gain=0.09818203542496506)
- adobe:y2010:p0153: acquire -> optimize => acquire -> optimize -> realize_revenue (gain=0.09818203542496506)
