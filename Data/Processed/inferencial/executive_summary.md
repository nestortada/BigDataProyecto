# Executive Summary - SECOP II Inferential Analysis

## Overview
- Input dataset: `C:\Users\nesto\OneDrive - Universidad de la Sabana\Universidad\2026-1\Big data\Proyecto\Data\Processed\Limpieza\datos_feature_engineering.parquet`
- Candidate variables reviewed: 100
- Group comparisons attempted: 25
- Categorical associations attempted: 66
- Numeric associations attempted: 103
- Models estimated successfully: 1

## Confidence Intervals
- Numeric mean confidence intervals generated: 20
- Proportion confidence intervals generated: 40

## Assumptions and Defensive Logic
- Assumption checks generated: 25
- Skipped group comparisons: 0
- Normality was evaluated with Shapiro-Wilk on bounded samples, and homoscedasticity with Levene when applicable.
- For large samples, p-values can become trivially small; effect sizes should be prioritized in interpretation.

## Notable Group Differences
- `duracion_dias` vs `departamento_entidad`: Kruskal-Wallis with p-value 0 and epsilon_squared=0.0428.
- `duracion_dias` vs `justificaci_n_modalidad_de`: Kruskal-Wallis with p-value 0 and epsilon_squared=0.0609.
- `duracion_dias` vs `modalidad_de_contratacion`: Kruskal-Wallis with p-value 0 and epsilon_squared=0.0529.
- `duracion_dias` vs `ordenentidad`: Kruskal-Wallis with p-value 0 and epsilon_squared=0.0155.
- `duracion_dias` vs `tipo_de_contrato`: Kruskal-Wallis with p-value 0 and epsilon_squared=0.0671.

## Notable Categorical Associations
- `competencia_reportada` vs `anomalia_temporal`: Chi-square with p-value 0 and Cramer's V=0.3220.
- `competencia_reportada` vs `fue_duplicado_logico`: Chi-square with p-value 0 and Cramer's V=0.1414.
- `confianza_label` vs `anomalia_temporal`: Chi-square with p-value 0 and Cramer's V=0.1416.
- `confianza_label` vs `competencia_reportada`: Chi-square with p-value 0 and Cramer's V=0.6567.
- `confianza_label` vs `fue_duplicado_logico`: Chi-square with p-value 0 and Cramer's V=0.1032.

## Notable Numeric Associations
- `conteo_de_respuestas_a_ofertas` vs `precio_base`: Pearson r=0.0021 (p=0.164) and Spearman rho=0.0698 (p=0).
- `conteo_de_respuestas_a_ofertas` vs `respuestas_al_procedimiento`: Pearson r=0.1811 (p=0) and Spearman rho=0.1378 (p=0).
- `conteo_de_respuestas_a_ofertas` vs `score_competencia`: Pearson r=0.1019 (p=0) and Spearman rho=0.1353 (p=0).
- `conteo_de_respuestas_a_ofertas` vs `score_temporal`: Pearson r=0.0673 (p=0) and Spearman rho=0.1000 (p=0).
- `conteo_de_respuestas_a_ofertas` vs `visualizaciones_del`: Pearson r=0.0555 (p=6.681e-290) and Spearman rho=0.1076 (p=0).

## Model Diagnostics
- `linear_regression` on `transparency_score` used 120,000 rows with AIC=-224276.8382.

## Post Hoc Signals
- `duracion_dias` by `ordenentidad`: `Nacional` vs `Territorial` remained significant after Holm correction (adjusted p=0).
- `duracion_dias` by `ordenentidad`: `Corporación Autónoma` vs `Territorial` remained significant after Holm correction (adjusted p=6.92e-57).
- `duracion_dias` by `ordenentidad`: `Corporación Autónoma` vs `Nacional` remained significant after Holm correction (adjusted p=6.077e-20).
- `precio_base_log` by `ordenentidad`: `Nacional` vs `Territorial` remained significant after Holm correction (adjusted p=0).
- `precio_base_log` by `ordenentidad`: `Corporación Autónoma` vs `Territorial` remained significant after Holm correction (adjusted p=2.053e-170).

## Interpretation Notes
- Statistical significance does not imply causality.
- Results should be interpreted alongside data quality, missingness patterns, derived-score construction, and the institutional context of SECOP II.
- Variables documented as leakage or post-outcome fields were excluded from inferential modeling to avoid tautological or contaminated conclusions.
- Sparse temporal coverage and highly imbalanced binary outcomes remain important limitations for institutional transparency inference.