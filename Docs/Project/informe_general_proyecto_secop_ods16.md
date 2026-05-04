# Informe general del proyecto SECOP II y ODS 16

**Analitica, modelado y operacionalizacion de riesgo de baja transparencia**

Fecha de generacion: 2026-05-03.

Proyecto academico de Big Data orientado a analizar procesos de contratacion publica de SECOP II y construir evidencia sobre transparencia, trazabilidad, competencia y calidad institucional en el marco del ODS 16.6.

\newpage

## Resumen ejecutivo

El proyecto construye un flujo completo de datos para SECOP II: descarga e ingesta, limpieza, analisis descriptivo, analisis inferencial, ingenieria de variables, modelado predictivo y una aplicacion FastAPI para consumir un modelo de prediccion temprana de riesgo de baja transparencia.

La base cruda parte de 450.000 procesos y 58 columnas. Despues de depuracion y feature engineering se trabaja con 429.541 registros, 106 variables finales y artefactos analiticos en Data, Reports, Models y Docs. El enfoque principal no es predecir adjudicacion, sino construir y explotar senales de transparencia institucional.

El resultado mas importante es una arquitectura reproducible que combina indicadores de completitud, trazabilidad, coherencia temporal y competencia. El modelo temprano usa solo informacion disponible al inicio del proceso contractual, lo que reduce la dependencia de variables posteriores al evento y vuelve el caso de uso mas util para alertas preventivas.

- Datos procesados principales: 429.541 filas y 106 columnas en datos_feature_engineering.parquet.
- Riesgo legacy usado en modelado: 340.715 positivos, equivalentes al 79,32% de los registros.
- Target v2 estricto generado para control: 1.484 positivos, equivalentes al 0,35% de los registros.
- Modelo early seleccionado por el reporte de modelado: LogisticRegression_early con F1 de prueba 0,9941.
- Aplicacion operativa actual: FastAPI con RandomForest_early_pipeline.joblib, umbral 0,645 y endpoints web/API.

## 1. Contexto y proposito

SECOP II concentra informacion publica sobre procesos contractuales del Estado. Para un proyecto alineado con el ODS 16.6, la pregunta central no es solamente cuanto se contrata, sino que tan completa, trazable, coherente y competitiva es la informacion que sustenta esos procesos.

El objetivo general del proyecto es construir un sistema analitico capaz de detectar senales de baja transparencia institucional en etapas tempranas del proceso, usando evidencia cuantitativa y modelos de aprendizaje automatico. El enfoque permite apoyar revision, priorizacion y auditoria de procesos que podrian requerir mayor seguimiento.

**Tabla 1. Alcance funcional del proyecto**

| Componente | Proposito | Artefactos principales |
| --- | --- | --- |
| Ingesta y datos crudos | Obtener y conservar la fuente SECOP II original. | Data/Raw/secop_procesos.parquet; Data/Raw/datos_analisis.parquet |
| Limpieza y preparacion | Normalizar tipos, duplicados, nulos y columnas no aptas para modelado. | Code/DataPrep/limpieza_datos_analisis_secop.py; Data/Processed/Limpieza/ |
| EDA y descriptivo | Caracterizar distribuciones, cobertura, calidad y senales institucionales. | Data/Processed/Descriptivo/; Data/Processed/EDA/; Docs/DataReport/ |
| Inferencial | Evaluar asociaciones, diferencias de grupos e intervalos de confianza. | Data/Processed/inferencial/; analisis_inferencial_secop.py |
| Modelado | Entrenar y comparar modelos predictivos para riesgo de baja transparencia. | Reports/Model/; Models/trained/; pipeline_modelado_secop.py |
| Operacionalizacion | Servir predicciones tempranas mediante formulario web y API. | Code/Operationalization/secop_fastapi/main.py; static/index.html |

## 2. Fuente de datos y evolucion del dataset

El proyecto conserva una trazabilidad clara entre la base cruda, la vista analitica, la base limpia y la base enriquecida. Esta separacion es importante porque permite distinguir datos originales, decisiones de limpieza, variables derivadas y artefactos orientados al modelo.

**Tabla 2. Tamano de los principales conjuntos de datos**

| Archivo | Filas | Columnas | Lectura |
| --- | --- | --- | --- |
| Data/Raw/secop_procesos.parquet | 450.000 | 58 | Base principal cruda. |
| Data/Raw/datos_analisis.parquet | 450.000 | 49 | Vista analitica sin campos posadjudicacion. |
| Data/Processed/Limpieza/datos_analisis_limpio.parquet | 429.541 | 58 | Base depurada para analisis. |
| Data/Processed/Limpieza/datos_feature_engineering.parquet | 429.541 | 106 | Base con variables, scores y targets. |
| Data/Processed/Model/train_early.parquet | 257.724 | 16 | Particion de entrenamiento early. |
| Data/Processed/Model/validation_early.parquet | 85.908 | 16 | Particion de validacion early. |
| Data/Processed/Model/test_early.parquet | 85.909 | 16 | Particion de prueba early. |

A nivel de almacenamiento, el proyecto ocupa aproximadamente 610 MB en Data, 474 MB en Models, 90 MB en Reports, 2,2 MB en Docs y 1,4 MB en Code. Esto confirma que el repositorio contiene tanto pipeline reproducible como artefactos ya materializados.

## 3. Limpieza y calidad de datos

La etapa de limpieza identifica problemas propios de datos administrativos: nulos estructurales, codigos semanticos como 'No definido', duplicados logicos, colas largas en variables monetarias y temporales, y campos posadjudicacion que podrian contaminar un modelo predictivo si se usan antes de tiempo.

El diagnostico inicial encontro 18.910 duplicados exactos y 23.648 filas repetidas por id_del_proceso. En la base descriptiva posterior ya no aparecen duplicados exactos ni duplicados por id_del_proceso, aunque persisten referencias y portafolios repetidos que deben interpretarse como senales de seguimiento y no necesariamente como errores.

**Tabla 3. Principales senales de calidad en la base enriquecida**

| Dimension | Resultado observado | Implicacion |
| --- | --- | --- |
| Cobertura | 429.541 filas y 100 columnas en el perfil descriptivo. | Volumen suficiente para analisis y ML. |
| Nulos altos | categorias_adicionales 83,94%; fecha_de_apertura_de_respuesta 83,04%; fecha_de_recepcion_de 78,06%. | La trazabilidad temporal es la dimension mas fragil. |
| Precio base | Media 403,5 millones; mediana 14,5 millones; p95 496,0 millones. | Distribucion muy asimetrica; conviene usar transformaciones robustas. |
| Duplicados | 0 duplicados exactos y 0 duplicados por id_del_proceso en la base final descriptiva. | La limpieza redujo ruido fisico y logico clave. |
| Riesgo institucional | nivel_riesgo_transparencia mas comun: medio, con 78,62%. | La mayoria de procesos queda en zona intermedia de riesgo. |
| Confianza | confianza_label mas comun: baja, con 61,73%. | La calidad de evidencia exige cautela en lectura institucional. |

Una decision metodologica acertada es no imputar fechas originales de manera indiscriminada. La ausencia de fechas forma parte de la senal de trazabilidad; inventar fechas podria mejorar artificialmente la completitud y debilitar la interpretacion del riesgo.

## 4. Ingenieria de variables y construccion del target

El feature engineering transforma la base limpia en un conjunto con variables derivadas sobre calidad textual, presencia de campos requeridos, ubicacion, trazabilidad, ventanas temporales, competencia reportada y scores agregados. El pipeline procesa 429.541 filas y genera 106 columnas finales.

**Tabla 4. Dimensiones centrales de feature engineering**

| Dimension | Variables o scores | Interpretacion |
| --- | --- | --- |
| Completitud | score_completitud, missing_required_fields_count, flags de presencia. | Mide si el registro contiene campos institucionalmente necesarios. |
| Trazabilidad | score_trazabilidad, flag_id_proceso_valido, flag_tiene_url_publica. | Evalua si el proceso puede seguirse y verificarse. |
| Temporalidad | dias_publicacion_a_recepcion, dias_recepcion_a_apertura, score_temporal. | Resume coherencia y disponibilidad de hitos temporales. |
| Competencia | total_respuestas, total_interes_oferentes, score_competencia. | Aproxima participacion y concurrencia de oferentes. |
| Riesgo | transparency_score, nivel_riesgo_transparencia, riesgo_baja_transparencia. | Integra las dimensiones para etiquetar riesgo. |

El proyecto genera dos lecturas de etiqueta. El target legacy, usado por los reportes de modelado early, marca 340.715 positivos (79,32%). El target v2 es mucho mas estricto: 1.484 positivos (0,35%), con controles de confianza y cobertura de evidencia. Esta doble lectura es valiosa porque permite comparar un criterio amplio de alerta contra un criterio conservador de riesgo extremo.

## 5. Analisis descriptivo

El analisis descriptivo confirma que SECOP II presenta alta heterogeneidad en entidades, modalidades, categorias y objetos contractuales. La variable nombre_del_procedimiento tiene 251.183 categorias y la entidad tiene 1.985 categorias, lo que explica la necesidad de modelos capaces de manejar variables categoricas de alta cardinalidad.

- El precio_base esta fuertemente sesgado: media muy superior a la mediana y presencia de outliers relevantes.
- El transparency_score tiene media 0,6409 y mediana 0,5900, con rango entre 0,3475 y 1,0000.
- score_completitud y score_trazabilidad son altos en promedio, pero score_temporal y score_competencia son bajos.
- La temporalidad disponible se concentra en fecha_de_publicacion_del; recepcion y apertura tienen cobertura parcial.

La conclusion descriptiva es que el problema de transparencia no esta solamente en campos vacios, sino en la combinacion entre informacion incompleta, baja trazabilidad temporal, participacion reducida y registros con alta variabilidad semantica.

## 6. Analisis inferencial

El analisis inferencial revisa 100 variables candidatas, 25 comparaciones de grupos, 66 asociaciones categoricas, 103 asociaciones numericas, 20 intervalos de confianza para medias y 40 intervalos para proporciones. Las pruebas se ejecutan con logica defensiva: Shapiro-Wilk para normalidad en muestras acotadas, Levene para homocedasticidad y enfasis en tamanos de efecto por el gran tamano muestral.

**Tabla 5. Senales inferenciales destacadas**

| Relacion | Metodo | Resultado | Lectura |
| --- | --- | --- | --- |
| duracion_dias vs tipo_de_contrato | Kruskal-Wallis | p=0; epsilon^2=0,0671 | Diferencias pequenas a moderadas segun tipo contractual. |
| duracion_dias vs justificaci_n_modalidad_de | Kruskal-Wallis | p=0; epsilon^2=0,0609 | La modalidad se asocia con duraciones distintas. |
| confianza_label vs competencia_reportada | Chi-cuadrado | p=0; V de Cramer=0,6567 | Asociacion fuerte entre confianza y competencia. |
| competencia_reportada vs anomalia_temporal | Chi-cuadrado | p=0; V de Cramer=0,3220 | Relacion moderada entre competencia y anomalias temporales. |
| conteo_de_respuestas_a_ofertas vs precio_base | Pearson/Spearman | r=0,0021; rho=0,0698 | Relacion practica muy baja aunque estadisticamente detectable. |

La lectura institucional debe evitar causalidad directa. Los resultados muestran patrones y asociaciones, pero deben interpretarse junto con calidad del dato, forma de construccion de los scores y contexto contractual.

## 7. Modelado predictivo

El pipeline de modelado compara regresion logistica, arboles, Random Forest, Gradient Boosting, XGBoost, LightGBM y CatBoost. El preprocesamiento se encapsula en Pipeline y ColumnTransformer, se ajusta solo con entrenamiento/CV, se evita SMOTE y se eliminan variables post-evento, metadata granular y columnas derivadas directamente del target.

**Tabla 6. Resultado del modo early por validacion**

| Modelo | Umbral | Accuracy | Precision | Recall | F1 | ROC AUC | PR AUC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| LogisticRegression_early | 0,270 | 0,9912 | 0,9955 | 0,9934 | 0,9944 | 0,9997 | 0,9999 |
| XGBoost_early | 0,285 | 0,9890 | 0,9928 | 0,9933 | 0,9931 | 0,9996 | 0,9999 |
| LightGBM_early | 0,390 | 0,9890 | 0,9954 | 0,9908 | 0,9931 | 0,9995 | 0,9999 |
| RandomForest_early | 0,645 | 0,9885 | 0,9926 | 0,9929 | 0,9928 | 0,9995 | 0,9999 |
| CatBoost_early | 0,325 | 0,9882 | 0,9925 | 0,9926 | 0,9926 | 0,9995 | 0,9999 |

En prueba, LogisticRegression_early mantiene Accuracy 0,9906, Precision 0,9951, Recall 0,9931, F1 0,9941, ROC AUC 0,9996 y PR AUC 0,9999. La validacion cruzada reporta F1 medio 0,9936 con desviacion 0,0002.

**Tabla 7. Comparacion normal, strict y early**

| Metrica | Normal | Strict | Early | Caida early vs strict |
| --- | --- | --- | --- | --- |
| F1 | 0,9989 | 0,9989 | 0,9944 | 0,0045 |
| Recall | 0,9996 | 0,9990 | 0,9934 | 0,0056 |
| Precision | 0,9983 | 0,9989 | 0,9955 | 0,0033 |
| ROC AUC | 1,0000 | 1,0000 | 0,9997 | 0,0003 |
| PR AUC | 1,0000 | 1,0000 | 0,9999 | 0,0001 |

Las metricas son extremadamente altas y los reportes generan advertencias automaticas de posible leakage por ROC AUC y PR AUC superiores a 0,995. El modo early mitiga parte del riesgo porque restringe variables a informacion inicial, pero aun asi se recomienda validar la etiqueta, los umbrales y el comportamiento fuera de muestra antes de un uso institucional.

### Variables tempranas usadas

- entidad, nit_entidad, departamento_entidad, ciudad_entidad y ordenentidad.
- nombre_del_procedimiento y descripci_n_del_procedimiento.
- fase, precio_base y precio_base_log.
- modalidad_de_contratacion, justificaci_n_modalidad_de, codigo_principal_de_categoria, tipo_de_contrato y categorias_adicionales.

## 8. Operacionalizacion

La operacionalizacion se materializa en una aplicacion FastAPI llamada SECOP Transparencia Temprana. La API carga un pipeline serializado, recibe variables del proceso, calcula precio_base_log, estima probabilidad de riesgo y devuelve probabilidad, etiqueta, clase y factores clave de importancia.

**Tabla 8. Endpoints y comportamiento de la API**

| Endpoint | Metodo | Uso |
| --- | --- | --- |
| / | GET | Sirve el formulario web estatico. |
| /api/health | GET | Reporta estado del servicio, ruta del modelo, umbral y variables. |
| /api/predict | POST | Recibe JSON de proceso SECOP II y retorna prediccion de riesgo. |

El codigo operativo actual carga Models/trained/all_models/RandomForest_early_pipeline.joblib y usa umbral 0,645. Este artefacto coincide con una opcion robusta del experimento early, pero el reporte de seleccion marca LogisticRegression_early como mejor modelo por F1. Antes de cerrar una version productiva conviene documentar si la eleccion de Random Forest fue intencional o alinear la API con best_model_pipeline_early.joblib.

El proyecto tambien incluye docker-compose.yml para PostgreSQL y un flujo de carga que reporta estado uploaded, 429.541 filas cargadas y tabla secop_feature_engineering. Esto permite conectar el pipeline analitico con una capa persistente para consultas o integracion.

## 9. Arquitectura tecnica y reproducibilidad

La organizacion del repositorio es clara: Code contiene scripts de datos, analisis, modelado y API; Data separa Raw, Processed, EDA, Descriptivo, inferencial y Model; Reports agrupa metricas, matrices, curvas, feature importance y validaciones; Models conserva pipelines serializados; Docs contiene informes academicos y tecnicos.

- Dependencias principales: pandas, pyarrow, scipy, scikit-learn, joblib, matplotlib, seaborn, statsmodels, openpyxl, Prefect, FastAPI, Uvicorn y Pydantic.
- Los modelos se guardan como pipelines joblib, lo que reduce errores entre entrenamiento y consumo.
- Los reportes de modelado incluyen validacion cruzada, optimizacion de umbral, curvas ROC/PR, matrices de confusion y feature importance.
- La API evita recalcular features complejas y se concentra en variables tempranas disponibles para el formulario.

## 10. Riesgos, limitaciones y controles necesarios

**Tabla 9. Riesgos principales del proyecto**

| Riesgo | Evidencia | Control recomendado |
| --- | --- | --- |
| Leakage o target reconstruible | Metricas cercanas a 1,0 y advertencias automaticas. | Validar reglas del target, hacer pruebas temporales y evaluar target v2. |
| Desbalance de etiquetas | Target legacy 79,32% positivo; target v2 0,35% positivo. | Elegir target segun caso de uso: alerta amplia o riesgo extremo. |
| Nulos temporales estructurales | Recepcion y apertura tienen alta ausencia. | No imputar fechas originales sin justificacion; usar banderas y ventanas derivadas. |
| Alta cardinalidad categorica | Nombre de procedimiento y descripcion tienen cientos de miles de valores. | Controlar codificacion, regularizacion y generalizacion a entidades nuevas. |
| Diferencia entre modelo seleccionado y modelo servido | Reporte early selecciona LogisticRegression; API sirve RandomForest. | Alinear artefacto operativo o documentar criterio de seleccion. |
| Interpretacion causal indebida | Inferencias son asociaciones con muestras grandes. | Reportar tamanos de efecto y contexto institucional. |

## 11. Recomendaciones

- Definir formalmente que target se usara para la version final: legacy para alertas amplias o v2 para riesgo extremo conservador.
- Ejecutar una validacion temporal estricta por fecha de publicacion para comprobar generalizacion en periodos futuros.
- Alinear la API con el mejor artefacto seleccionado o justificar por escrito la preferencia por RandomForest_early.
- Agregar pruebas automaticas del endpoint /api/predict con payloads representativos y casos borde de precio, categoria y campos faltantes.
- Crear una ficha de modelo con variables permitidas, umbral, distribucion del target, advertencias de leakage y uso previsto.
- Complementar las metricas globales con analisis por departamento, entidad, modalidad y tipo de contrato para revisar sesgos de desempeno.
- Mantener informes PDF/Word como entregables, pero preservar Markdown y CSV como fuentes reproducibles.

## 12. Conclusiones

El proyecto esta bastante avanzado y cubre el ciclo completo de un caso de Big Data aplicado: datos, calidad, analitica, inferencia, modelado y despliegue. Su mayor fortaleza es que traduce la transparencia institucional en dimensiones medibles y conserva artefactos reproducibles para auditar cada etapa.

La principal precaucion es metodologica: las metricas predictivas son tan altas que deben interpretarse como una senal para revisar el target, la separacion temporal y la posible dependencia entre variables tempranas y reglas de etiquetado. Con esa validacion, el proyecto puede sostener un entregable academico robusto y una demostracion operativa clara.

## Anexos. Fuentes internas consultadas

- Docs/DataReport/informe_limpieza_preparacion_ml_secop.md
- Data/Processed/Descriptivo/executive_summary.md
- Data/Processed/inferencial/executive_summary.md
- Reports/FeatureEngineering/feature_engineering_summary.txt
- Reports/FeatureEngineering/target_v2_label_quality_report.txt
- Reports/Model/best_model_report_early.txt
- Reports/Model/metrics_summary_early.csv
- Reports/Model/cross_validation_summary_early.csv
- Reports/Model/early_feature_policy_report.txt
- Code/Operationalization/secop_fastapi/main.py
- Code/Operationalization/secop_fastapi/README.md
- docker-compose.yml y requirements.txt
