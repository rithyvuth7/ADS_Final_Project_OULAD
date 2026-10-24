# OULAD Early-Warning Project (Advanced Data Science, CADT)

Working question: can early engagement (first 28 days of a course-presentation) identify students at risk of failing or withdrawing, and do the patterns hold across cohorts and student groups? (SDG 4)

## Data

- Open University Learning Analytics Dataset (OULAD), Kuzilek, Hlosta, Zdrahal (2017), Scientific Data 4:170171, doi:10.1038/sdata.2017.171
- Obtained from UCI Machine Learning Repository, dataset id 349, on 2026-10-24
- Licence: CC BY 4.0 as stated by the authors (re-confirm on the source page before submission)
- Raw files live in `data/raw/` and are never edited

## Folder structure

```
data/raw/        original CSVs (7 tables + OULAD.names)
data/processed/  analysis-ready tables
notebooks/       one notebook per phase
outputs/         transformation log, data dictionary, figures
```

## Notebooks

| # | Notebook | Status |
|---|---|---|
| 01 | `01_framing_and_provenance` research framing, provenance, structure | done |
| 02 | `02_master_table` master table at day 28, audit log, data dictionary | done |
| 03 | `03_eda_and_quality_audit` univariate, 7 bivariate pairs, data-quality audit, sensitivity checks | done |
| 04 | `04_feature_engineering_selection` rebuilt scores, relative engagement, selection on 2013 only | done |
| 05 | `05_clustering` k-means, Ward, DBSCAN, GMM on module-relative behaviour | done |
| 06 | `06_association_rules_categorical` FP-growth rules, chi-square and trend tests, CMH, correspondence analysis, MCA | done |
| 07a | `07a_modelling_evaluation_fairness` model, single 2014 test, calibration, fairness, rule and profile replication, cutoff sensitivity | done |
| 07b | `07b_imbalance_leakage_drift` imbalance strategies, leakage audit, drift analysis, monitoring plan | done |
| 07c | `07c_time_series_and_ab_test` activity forecasting, A/B test design with power simulations | done |
| 08 | Final report (Assignment 1 structure plus modelling) and 15-minute presentation | next |

## Reproduce

1. Python 3.13 with: pandas 2.3.2, numpy 2.2.6, scipy 1.16.3, scikit-learn 1.8.0, matplotlib 3.10.6, statsmodels 0.14.6, jupyter (nbclient 0.11.0, nbformat 5.11.0)
   Notebook 06 also needs `mlxtend` 0.23.4 (newer versions pull in pandas 3 and numpy 2.5, which this project was not built on). Correspondence analysis is implemented directly with numpy, so `prince` is not needed.
   Shared item definitions for the rules are in `src/items.py` (notebooks add `../src` to the path). Notebook 07 may use `imbalanced-learn` (optional).
2. Put the 7 OULAD CSVs in `data/raw/`. The two largest raw files (`studentVle.csv`, 422 MB, and `oulad_uci.zip`) are not in this repository because GitHub rejects files over 100 MB. Download the dataset from the UCI Machine Learning Repository (dataset 349, Open University Learning Analytics Dataset) and unzip it into `data/raw/`. The other raw CSVs are already included.
3. Run notebooks in order from the `notebooks/` folder (paths are relative to it).
   `studentVle.csv` has 10.6M rows; notebook 02 uses about 1 GB of RAM at peak.
4. Key parameter: `CUTOFF_DAY = 28` in notebook 02. Change it and rerun to get another prediction time.

## Key design decisions

- Unit of analysis: one student in one course-presentation (28,785 people, 32,593 rows).
- Population: students still registered at the cutoff. 5,040 early leavers are excluded because their outcome is already known.
- Target: `target_at_risk` = Fail or Withdrawn. At-risk rate is 44.2% in the day-28 population.
- All features use only data dated before the cutoff. `date_unregistration` and `final_result` are not features.
- Learned preprocessing (imputation, scaling, encoding, resampling) happens only after the train/test split.
- Planned split: train on 2013 presentations, test on 2014 (also a drift test).

## Main EDA findings (notebook 03)

- No single early feature separates at-risk students well: best AUC is 0.67 (early score, active days, total clicks). Expect a modest day-28 model.
- Modules differ about 10x in activity (median clicks FFF 474 vs GGG 48), which hides the signal: within a module the clicks AUC reaches 0.70 to 0.74 (pooled 0.66). Use within-module relative engagement.
- At-risk students click about half as much even before the course starts, so risk is visible before day 0.
- Deprivation gradient: 56% at risk in the most deprived IMD band vs 35% in the least; the odds ratio per band (0.92) hardly changes after controlling for module, background and engagement.
- Engagement predicts Fail much more than Withdrawn (Fail 41% to 13% from lowest to highest quartile; Withdrawn 22% to 14%).
- Target rate is stable across years (44.4% in 2013, 44.0% in 2014), but February starts run about 7 points higher than October starts.

## Main feature findings (notebook 04)

- 18 features selected from 48 candidates on the 2013 training years only. Grouped 5-fold CV AUC (by student) is 0.784 (logistic) and 0.793 (boosting), the same as using all 48. Selected set is in `outputs/feature_groups.json`.
- Early assessment behaviour is the strongest family (adds about 0.05 AUC), then engagement (0.725 on its own), then background (adds 0.01 to 0.015). Module and season alone reach only 0.57.
- Early scores are now percentile ranks within each assessment. The 100 or 1 coded assessment exists only in 2014 (1,674 test students, none in 2013), so the old score feature would have broken silently on the test set.
- Relative engagement (rank within module-presentation) adds nothing when the module is known, but beats raw engagement for unseen modules (leave-one-module-out AUC 0.738 vs 0.728).
- Test set differs from training: module CCC is 22% of the 2014 rows and absent in 2013; 9.3% of test rows are returning students.

## Main clustering findings (notebook 05)

- Fitted on 2013 only (11,361 active students); 2014 students are only assigned. No 2014 labels used.
- A first attempt on raw behaviours produced clusters that were mostly modules (Cramer's V 0.52, PSI 2013 vs 2014 of 0.45). Ranking every behaviour within its module-presentation fixed it (V 0.17, PSI 0.01).
- Engagement is a continuum, not natural types: silhouettes under 0.25, DBSCAN finds no density gaps, methods disagree on boundaries (ARI 0.3 to 0.6). The pre-stated rules leave k = 3 (GMM stability 0.90; unstable beyond k = 3).
- Profiles: no activity (3.9% of students, 81% at risk), low engagement (63%), intermittent (47%), sustained (29%). Effect holds after controlling for module (odds ratios 4.2 and 2.2 vs sustained); the deprivation gap (13 to 16 points) persists inside every profile.
- Profile shares are the same in 2013 and 2014 (PSI 0.002) and for the unseen module CCC. Clusters add no AUC to the selected features, so they are descriptive only.

## Main rule and categorical findings (notebook 06, 2013 training years only)

- Strongest single fact: missing an assessment that was due before day 28 (12.4% of students) goes with 87.6% at risk (lift 1.97). With a disadvantaged background the confidence reaches 91 to 96% for small groups. 173 rules survive the lift, size, corrected-p and redundancy filters (saved in `outputs/association_rules_2013.csv`).
- Cramer's V with being at risk: early assessment behaviour 0.33, engagement profile 0.31, early score 0.30, prior education 0.17, deprivation 0.15, prior attempts 0.13. Gender 0.01 (no association).
- Background effects are not module artefacts (Mantel-Haenszel odds ratios equal the crude ones), but the engagement effect differs by module (odds ratio low vs sustained: 11.7 in EEE, 1.6 in GGG).
- Equity: the deprivation gap is 13 to 16 points for students who look fine on the platform, but only 5 points among students who missed an assessment (already about 90% at risk).
- Rule lists trade precision for reach: at 70% precision, 8 rules flag 35% of students and reach 56% of the at-risk ones.
- Correspondence analysis: each table reduces to one gradient (91 to 96% of inertia); engagement is about four times as strongly tied to the final result as deprivation.

## Main modelling, data-issue and design findings (notebook 07)

- **Test result (2014, evaluated once):** gradient boosting AUC 0.780 (95% interval 0.773 to 0.786), cross-validated 0.793 on 2013. Calibration is excellent (mean predicted 44.3%, observed 44.0%). Flagging the top 30% gives precision 75.7% and finds 51.6% of at-risk students. Best single baseline (engagement rank) 0.70.
- **Where it works:** unseen module CCC is predicted best (0.81); weakest GGG (0.69) and the small AAA (0.70). The apparent gender gap (women 0.751, men 0.797) is a module-mix effect.
- **Fairness:** with the socio-economic features recall is 58.7% for deprived vs 45.0% for affluent students and false-positive rates differ twofold; without them (cost 0.009 AUC) recall and false-positive rates equalise. Recommended: alert model without education and deprivation.
- **Replication on 2014:** engagement-profile risks within 3 points; all 173 rules keep their direction, 83.8% keep lift 1.25 or more (risk rules 98%, protective rules 60 to 71%).
- **How early:** engagement alone gives AUC 0.64 before the start, 0.70 at day 28; the full model 0.78.
- **Imbalance:** resampling and class weights add no ranking information and inflate probabilities 2 to 3 times; SMOTE hurts; threshold tuning on the plain model matches all fixes.
- **Leakage:** future information inflates test AUC by up to 0.084 and survives a train/test split; random splits 0.009 optimistic; target encoding on all rows 0.18; screening before CV 0.15.
- **Drift:** inputs shifted (module mix, deadlines) but the relationship did not (retraining gain 0.002, coefficient correlation 0.98). The 100-or-1 coded assessment would have cost 0.016 AUC overall (0.074 for affected students). Rank features and course length create false adversarial-validation alarms (AUC 0.89 falls to 0.71 without them). Two fragile features (module_presentation_length, days_before_due_mean) can be removed at a cost of about 0.001 AUC.
- **Time series:** deadlines dominate activity (1.9x to 3.2x the local level the day before); forecasting days 28 to 55, last year's twin profile has sMAPE 17% vs about 29% for the best statistical method.
- **A/B test design:** 4,721 flagged students a year give 2,360 per arm and detect a 3.7-point reduction (80% power); a single module only 10.5 points. A naive contacted vs not-contacted comparison gives +36 points when the true effect is -5; regression discontinuity is unbiased but about four times noisier.

## Known data issues (see `outputs/transformation_log.csv` and `outputs/data_quality_audit.csv`)

- BBB 2014J assessment 15020 is coded only 100 or 1 (a completion marker, not a mark); 671 students have an early mean of exactly 1. To be fixed in notebook 04.
- Missing IMD is informative (29% at risk vs 45%).

- Missing values are coded `"?"`.
- Metadata states 32,953 students in one place and 32,593 elsewhere; the data have 32,593.
- 9 students have an unregistration date but are labelled Fail, not Withdrawn.
- 93 Withdrawn students have no unregistration date, so their exit time is unknown.
- `imd_band` mixes `10-20` and `10-20%` style labels.
- The click log has 787,170 exact duplicate rows; they are kept and summed.
- Modules EEE and GGG have no assessment due before day 28.

## Final presentation (25 minutes: 15 present + 10 Q&A)

| Min | Slide block | Source notebook |
|---|---|---|
| 0 to 2 | Research problem, SDG 4, objectives, RQs | 01 |
| 2 to 4 | Dataset, unit of analysis, cutoff logic | 01, 02 |
| 4 to 6 | Preprocessing and data-quality findings (audit trail) | 02, 03 |
| 6 to 8 | EDA highlights: engagement vs risk, group differences | 03 |
| 8 to 10 | Methods: features, GMM clusters, association rules | 04 to 06 |
| 10 to 13 | Results and evaluation: prediction, drift 2013 to 2014, fairness | 07 |
| 13 to 15 | Conclusions, limitations, recommendations | all |
| 15 to 25 | Q&A: prepare answers on the cutoff choice, leakage controls, early leavers, fairness |  |
