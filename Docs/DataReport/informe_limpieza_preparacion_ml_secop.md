# Informe técnico de limpieza y preparación de datos para Machine Learning sobre SECOP II

## Alcance y fuentes

Este informe se construyó a partir de cuatro fuentes del proyecto:

1. `Data/Raw/secop_procesos.parquet`, base principal del proyecto, con 450.000 filas y 58 columnas.
2. `Data/Raw/datos_analisis.parquet`, vista analítica derivada de la base principal, con las mismas 450.000 filas y 49 columnas, usada por el pipeline EDA.
3. Los artefactos generados en `Data/Processed/ods16_secop/`, en especial `null_profile.csv`, `univariate_numeric_summary.csv`, `duplicates_summary.csv`, `temporal_summary.csv`, `temporal_anomalies.csv`, `column_recommendations.csv`, `transparency_quality_indicators.csv`, `entity_quality_summary.csv` y los gráficos asociados.
4. El documento conceptual `Docs/Project/informe_general_secop_ods16.pdf`, que define el encuadre del proyecto en términos de transparencia institucional, trazabilidad, competencia y alineación con el ODS 16.6.

Nota metodológica: el EDA operativo fue ejecutado sobre `datos_analisis.parquet`, que conserva el mismo universo de procesos de `secop_procesos.parquet` y elimina nueve columnas posadjudicación. Además, el pipeline agrega columnas derivadas, por lo que algunos artefactos reportan más columnas que el parquet analítico original. En este informe se distinguen explícitamente columnas originales, columnas excluidas antes del EDA y variables derivadas.

## 1. Introducción

El objetivo del proyecto es analizar procesos de contratación pública en SECOP II para identificar señales de riesgo de baja transparencia institucional, en coherencia con el ODS 16.6, orientado a instituciones eficaces, transparentes y que rindan cuentas. Bajo ese marco, la limpieza y preparación de datos no es una etapa operativa secundaria: define qué tan confiable es la evidencia con la que se construyen indicadores, reglas o modelos de Machine Learning.

El documento conceptual del proyecto plantea que la variable objetivo más consistente no es si un proceso fue adjudicado o no, sino un indicador derivado de riesgo de baja transparencia. Ese enfoque requiere que la base modelable preserve cuatro dimensiones: completitud del registro, trazabilidad del proceso, coherencia temporal y señales de competencia. Por tanto, la preparación de datos debe depurar ruido técnico sin eliminar la información sustantiva que expresa esas dimensiones.

## 2. Diagnóstico de calidad de datos

### 2.1. Cobertura y estructura analítica

- La base principal `secop_procesos.parquet` contiene 450.000 registros y 58 columnas.
- La vista `datos_analisis.parquet` conserva las 450.000 filas y reduce la estructura a 49 columnas tras retirar nueve campos posadjudicación.
- El EDA confirma 18.910 duplicados exactos y 23.648 filas repetidas por `id_del_proceso`, por lo que la preparación debe tratar duplicación física y duplicación lógica.

### 2.2. Valores nulos

Los nulos no están distribuidos de forma homogénea. Se concentran casi por completo en hitos temporales del proceso.

| Columna | % nulos | Lectura técnica |
| --- | ---: | --- |
| `fecha_de_publicacion_fase` | 99,9998% | Variable prácticamente vacía; no es usable para modelado. |
| `fecha_de_publicacion` | 98,6180% | Cobertura insuficiente para usarla como fecha ancla. |
| `fecha_de_publicacion_fase_2` | 96,7240% | Alta ausencia y baja utilidad analítica. |
| `fecha_adjudicacion` | 84,9298% | Muy incompleta y además posterior al resultado. |
| `fecha_de_apertura_de_respuesta` | 79,3738% | Informativa, pero con ausencia estructural por fase. |
| `fecha_de_apertura_efectiva` | 76,8362% | Informativa, con alta ausencia y potencial fuga temporal. |
| `fecha_de_recepcion_de` | 74,5142% | Informativa, pero no apta para imputación indiscriminada. |
| `fecha_de_publicacion_fase_3` | 4,6582% | Cobertura razonable comparada con el resto de fechas auxiliares. |
| `fase` | 0,0111% | Prácticamente completa. |

Hallazgos relevantes:

- Los campos estructurales del proceso están completos: `id_del_proceso`, `referencia_del_proceso`, `entidad`, `nit_entidad`, `departamento_entidad`, `ciudad_entidad`, `precio_base` y la mayoría de variables categóricas tienen 0% de nulos en el corte analítico.
- La completitud promedio a nivel fila es 85,30%, pero esa cifra oculta que la pérdida de información se concentra en trazabilidad temporal.
- Solo 1,3160% de los registros cumple simultáneamente trazabilidad temporal completa y sin anomalías, según `transparency_quality_indicators.csv`. Esto confirma que la temporalidad es la dimensión más frágil del conjunto.

### 2.3. Valores atípicos y distribución

Los gráficos `plots/numeric/*.png` y `plots/bivariate/*.png` muestran distribuciones fuertemente asimétricas a la derecha. El resumen univariado confirma que no todos los outliers deben tratarse como error; varios reflejan colas largas propias de contratación pública.

| Variable | Evidencia del EDA | Implicación |
| --- | --- | --- |
| `precio_base` | Mediana 15.327.319,5; P99 6.074.146.980; máximo 13.212.285.859.000; 15,9773% de outliers IQR; mínimo -193.196.838 | Cola extrema y algunos valores imposibles. Requiere separar error material de valores altos válidos. |
| `duracion` | Mediana 21; P99 365; máximo 50.000.000; 4,2616% de outliers IQR | Hay valores implausibles y la variable debe normalizarse con `unidad_de_duracion`. |
| `visualizaciones_del` | Mediana 0; P99 159; máximo 617; 22,3847% de outliers IQR | Serie muy sesgada y cero-inflada; IQR sobredetecta “outliers”. |
| `respuestas_al_procedimiento` | Mediana 0; P99 28; máximo 231; 18,3471% de outliers IQR | La mayor parte del volumen es cero; cualquier valor positivo queda lejos del IQR. |
| `proveedores_unicos_con` | Mediana 0; P99 24; máximo 231; 18,0060% de outliers IQR | Patrón similar al anterior; no conviene eliminar filas por regla IQR. |
| `numero_de_lotes` | Mediana 0; P99 14; máximo 100; 4,7936% de outliers IQR | Distribución muy concentrada en cero. |

Verificaciones adicionales sobre el parquet analítico:

- `precio_base` presenta 5 valores negativos y 19.845 valores en cero.
- `duracion` tiene 27.726 registros en cero, 20 registros por encima de 3.650 y 10 por encima de 100.000.
- `proveedores_invitados` supera 100 en 11.776 filas; esto no es necesariamente error, pero sí indica una cola larga que debe estabilizarse antes del modelado.

Una observación clave es que varias variables de conteo tienen `Q1 = Q3 = 0`. En esos casos, la regla IQR marca como “outlier” casi cualquier valor positivo; por tanto, el IQR sirve como alerta de asimetría, no como criterio automático de eliminación.

### 2.4. Inconsistencias

#### Inconsistencias temporales

El pipeline reporta 18.468 registros con al menos una anomalía temporal, equivalentes a 4,1040% del total. Todas corresponden a la misma regla:

- `fecha_de_apertura_efectiva < fecha_de_recepcion_de`

Profundizando en esos casos:

- La diferencia entre recepción y apertura efectiva es de 1 día en el mínimo, 5 días en la mediana, 17 días en P95 y 307 días en el máximo.
- Las anomalías se concentran en `Evaluación` (11.910 casos) y `Seleccionado` (6.508 casos).

Esto sugiere que no se trata de ruido aleatorio, sino de una inconsistencia sistemática en la secuencia temporal o en la semántica de alguna fecha.

#### Inconsistencias semánticas

No todos los “faltantes” se expresan como nulos. El EDA muestra varios tokens semánticos que deben tratarse explícitamente:

- `subtipo_de_contrato`: 100,0000% `No Definido`
- `categorias_adicionales`: 81,1453% `No definido`
- `ciudad_de_la_unidad_de`: 29,0607% `No Aplica`
- `estado_resumen`: 50 registros con `No Definido`

Además, en la base cruda excluida del EDA se observa el mismo patrón:

- `adjudicado`: 84,5218% `No`
- `id_adjudicacion`: 84,5218% `No Adjudicado`
- `valor_total_adjudicacion`: 84,6122% igual a `0`
- Las columnas del proveedor adjudicado concentran entre 84,5224% y 93,9469% de `No Definido`

Estos valores no deben mezclarse indiscriminadamente con nulos. Algunos significan “desconocido”, otros “no aplica” y otros “todavía no adjudicado”.

#### Tipos de dato

La conversión realizada por el pipeline fue exitosa para las columnas tratadas:

- 100% de éxito de parseo en todas las fechas reconocidas por el EDA.
- 100% de éxito en la conversión numérica de `precio_base`, `duracion` y las variables de conteo.

La calidad del casteo no elimina la necesidad de depuración semántica: una variable puede convertirse correctamente y seguir conteniendo valores imposibles o poco informativos.

### 2.5. Duplicados

| Tipo de duplicado | Evidencia |
| --- | --- |
| Duplicado exacto | 18.910 filas duplicadas exactas |
| Duplicado por `id_del_proceso` | 23.648 filas, agrupadas en 3.189 claves |
| Duplicado por `referencia_del_proceso` | 75.688 filas, agrupadas en 16.575 referencias |

Hallazgos adicionales dentro de los grupos duplicados por `id_del_proceso`:

- `referencia_del_proceso`, `estado_del_procedimiento`, `fase`, `fecha_de_publicacion_del`, `fecha_de_publicacion` y `fecha_de_ultima_publicaci` son constantes en 100% de los grupos repetidos.
- `estado_resumen` varía en 43,52% de los grupos.
- `fecha_adjudicacion` varía en 43,59% de los grupos.

Conclusión técnica: `id_del_proceso` debe ser la clave primaria lógica; `referencia_del_proceso` no es única y solo debe conservarse como identificador de referencia. Además, una parte de los repetidos por `id_del_proceso` parece corresponder a versiones alternativas del mismo proceso, no a procesos independientes.

## 3. Estrategia de limpieza de datos

### 3.1. Principio general

La limpieza debe priorizar dos objetivos simultáneos:

1. Preservar señales útiles para medir transparencia.
2. Evitar que el modelo aprenda el resultado administrativo final o artefactos de captura.

Por eso la estrategia recomendada no es “llenar todo y eliminar extremos”, sino separar cuatro tipos de problema: ausencia estructural, error material, información posresultado y redundancia.

### 3.2. Tratamiento de valores nulos

#### a. Eliminación por inutilidad estructural

Se recomienda eliminar del set modelable:

- `fecha_de_publicacion_fase`
- `fecha_de_publicacion`
- `fecha_de_publicacion_fase_2`

Justificación:

- Superan 96% de ausencia.
- En `column_recommendations.csv` aparecen como `high_nulls`.
- Dos de ellas también aparecen como `near_constant`.
- Su imputación fabricaría cronología sin respaldo empírico.

#### b. Conservación con banderas de presencia

Se recomienda conservar como variables de trazabilidad, pero no imputar directamente, las siguientes columnas:

- `fecha_de_recepcion_de`
- `fecha_de_apertura_de_respuesta`
- `fecha_de_apertura_efectiva`
- `fecha_adjudicacion`

Acción propuesta:

- Crear `tiene_fecha_recepcion`, `tiene_fecha_apertura_respuesta`, `tiene_fecha_apertura_efectiva` y `tiene_fecha_adjudicacion`.
- Derivar ventanas temporales solo cuando existan ambos extremos.
- Si el algoritmo final no tolera nulos, imputar únicamente las variables derivadas numéricas, nunca las fechas originales, y acompañar esa imputación con su respectiva bandera de ausencia.

Razón técnica:

- La ausencia está ligada al estado o a la fase del proceso, no a un mecanismo de pérdida aleatorio.
- Imputar una fecha inventada alteraría precisamente la señal de transparencia que se quiere medir.

#### c. Fecha ancla recomendada

Para el modelado es preferible usar `fecha_de_publicacion_del` como fecha de inicio del proceso:

- Tiene 0% de nulos.
- Comparte el mismo rango temporal general que las fechas de publicación reportadas por el EDA.
- Evita depender de `fecha_de_publicacion`, que está ausente en 98,6180% de los casos.

#### d. Nulos semánticos

Se recomienda homologar tokens semánticos a un esquema de tres estados:

- `missing`: información esperada pero no reportada, por ejemplo `No definido`.
- `not_applicable`: información no aplicable, por ejemplo `No Aplica`.
- `not_yet_available`: información aún no ocurrida, por ejemplo `No Adjudicado`.

Esto evita perder semántica al convertir todos los placeholders en `NaN`.

### 3.3. Manejo de outliers

#### a. Detección

Usar dos niveles de detección:

1. Reglas de negocio para valores imposibles.
2. Reglas estadísticas para colas largas válidas.

Reglas de negocio recomendadas:

- `precio_base < 0` se marca como inválido y se convierte en `NaN`.
- `duracion < 0` se marca como inválido.
- `duracion` debe normalizarse a días antes de analizar extremos, usando `unidad_de_duracion`.

Reglas estadísticas recomendadas:

- IQR como alerta exploratoria.
- Percentiles altos, preferiblemente P99 o P99,5, como umbral de winsorización.
- Transformación `log1p` para montos y conteos muy sesgados.

#### b. Tratamiento propuesto por tipo de variable

**`precio_base`**

- Conservar una copia limpia `precio_base_clean`.
- Pasar los 5 negativos a `NaN` y crear `flag_precio_invalido`.
- Conservar los ceros, porque en el corte existen modalidades y tipos de contrato con mediana 0 y no todos esos casos son errores.
- Crear `log_precio_base = log1p(precio_base_clean)`.
- Winsorizar la cola superior por tipo o modalidad de contratación, no de forma global, porque los cruces `precio_base_vs_modalidad_de_contratacion.csv` y `precio_base_vs_tipo_de_contrato.csv` muestran heterogeneidad sustantiva entre categorías.

**`duracion`**

- Convertir a `duracion_dias` a partir de `unidad_de_duracion`.
- Mantener `unidad_de_duracion` como insumo auxiliar para auditoría.
- Marcar valores implausibles tras la conversión y recortar la cola superior con percentiles, no por eliminación masiva.

**Conteos de competencia y atención**

- `proveedores_invitados`
- `proveedores_con_invitacion`
- `respuestas_al_procedimiento`
- `respuestas_externas`
- `conteo_de_respuestas_a_ofertas`
- `proveedores_unicos_con`
- `visualizaciones_del`
- `numero_de_lotes`

Tratamiento recomendado:

- No eliminar filas solo porque sean outliers IQR.
- Aplicar `log1p` a las variables con cola larga.
- Reemplazar las versiones crudas casi constantes por banderas binarias o agregados interpretables.

Justificación:

- En varias de estas columnas el IQR es cero porque la mayoría de registros tiene valor 0.
- El EDA muestra que `respuestas_al_procedimiento` y `proveedores_unicos_con` tienen correlación de 0,9846, por lo que no aporta usar ambas versiones crudas al mismo tiempo.

### 3.4. Corrección de errores y estandarización

#### a. Tipos de datos

- Mantener la conversión temporal y numérica ya validada por el EDA.
- Normalizar `urlproceso` a una cadena URL simple, como ya hace el pipeline, pero usarla solo para trazabilidad, no como predictor bruto.

#### b. Estandarización textual

Se recomienda normalizar mayúsculas, tildes, espacios dobles y variantes triviales en:

- `entidad`
- `nombre_de_la_unidad_de`
- `ciudad_entidad`
- `ciudad_de_la_unidad_de`
- `departamento_entidad`
- `estado_resumen`

La justificación no es que el EDA ya pruebe errores ortográficos concretos, sino que la cardinalidad es alta y sensible a pequeñas variantes:

- `entidad`: 1.985 categorías
- `nombre_de_la_unidad_de`: 3.454 categorías
- `ciudad_entidad`: 410 categorías
- `ciudad_de_la_unidad_de`: 454 categorías

En variables con alta cardinalidad, pequeñas discrepancias de escritura generan dispersión artificial.

#### c. Identificadores y redundancias

Se recomienda:

- Conservar `codigo_entidad` como identificador estable de la entidad.
- Excluir `ppi` del modelado porque es idéntico a `codigo_entidad` en 100% de los registros.
- No usar simultáneamente `codigo_entidad`, `entidad` y `nit_entidad` como predictores. `codigo_entidad` determina a `entidad` y `nit_entidad` en 100% de los casos observados, mientras que `nit_entidad` no siempre determina una sola entidad.

### 3.5. Eliminación de duplicados

Regla recomendada:

1. Eliminar primero los 18.910 duplicados exactos.
2. Tomar `id_del_proceso` como clave primaria lógica.
3. Para grupos repetidos por `id_del_proceso`, conservar un solo registro por proceso usando una jerarquía de desempate:
   - mayor completitud en columnas no filtradas por leakage,
   - preferencia por `estado_resumen` distinto de `No Definido`,
   - mayor `fecha_de_ultima_publicaci` cuando exista desempate real,
   - criterio determinístico final por `id_del_portafolio` o por el orden original.
4. Mantener un `flag_id_del_proceso_duplicado` para análisis posteriores.

Esta regla es coherente con el diagnóstico: los repetidos por `id_del_proceso` comparten la mayoría de sus columnas estructurales y difieren sobre todo en atributos de seguimiento del mismo proceso.

## 4. Eliminación de variables irrelevantes

### 4.1. Variables a eliminar de forma definitiva del set modelable

| Grupo | Columnas | Justificación |
| --- | --- | --- |
| Posadjudicación y fuga directa | `adjudicado`, `id_adjudicacion`, `valor_total_adjudicacion`, `codigoproveedor`, `departamento_proveedor`, `ciudad_proveedor`, `nombre_del_adjudicador`, `nombre_del_proveedor`, `nit_del_proveedor_adjudicado` | Son variables conocidas después del resultado del proceso. Además, en la base cruda concentran entre 84,52% y 93,95% de valores por defecto (`No`, `No Adjudicado`, `No Definido` o `0`). |
| Identificadores de fila o de enlace | `id_del_proceso`, `referencia_del_proceso`, `id_del_portafolio`, `urlproceso` | Tienen altísima cardinalidad y sirven para trazabilidad, no para generalización. Deben conservarse fuera de la matriz de entrenamiento. |
| Redundancia exacta | `ppi` | Es idéntica a `codigo_entidad` en 100% del corte observado. |
| Variable constante | `subtipo_de_contrato` | Tiene un solo valor no nulo: `No Definido` en 100% de las filas. |
| Variable numérica sin señal | `proveedores_que_manifestaron` | Es 0 en 100% de las filas; no agrega varianza ni información. |
| Variables casi vacías | `fecha_de_publicacion_fase`, `fecha_de_publicacion`, `fecha_de_publicacion_fase_2` | Entre 96,72% y 99,9998% de ausencia; dos además son casi constantes. |

### 4.2. Variables que no deben usarse como predictores si el objetivo es detección temprana

| Columnas | Justificación |
| --- | --- |
| `estado_del_procedimiento`, `id_estado_del_procedimiento`, `estado_de_apertura_del_proceso`, `estado_resumen` | El EDA las marca como `potential_leakage`. Son variables de resultado o seguimiento avanzado del proceso. |
| `fecha_de_ultima_publicaci`, `fecha_de_apertura_efectiva`, `fecha_adjudicacion` | Son hitos posteriores o demasiado cercanos al resultado administrativo; contaminan un modelo orientado a señales tempranas de transparencia. |

### 4.3. Variables que conviene reemplazar por versiones derivadas

| Variable original | Problema | Reemplazo recomendado |
| --- | --- | --- |
| `duracion` + `unidad_de_duracion` | Mezcla de escalas y cola extrema | `duracion_dias`, `log_duracion_dias`, `flag_duracion_implausible` |
| `respuestas_al_procedimiento` y `proveedores_unicos_con` | Casi colineales y cero-infladas | `competencia_reportada`, `intensidad_competencia_log` o una sola de las dos |
| `respuestas_externas`, `conteo_de_respuestas_a_ofertas`, `proveedores_con_invitacion`, `numero_de_lotes` | Baja variabilidad en crudo | Banderas binarias, percentiles o agregados |
| `entidad` y `nit_entidad` | Redundantes frente a `codigo_entidad` para modelado tabular | Mantener para reporting, no para la matriz de entrenamiento si ya se usa `codigo_entidad` |
| `urlproceso` | Texto de alta cardinalidad | `flag_tiene_url` para auditoría; no usar la URL cruda |

### 4.4. Nota crítica sobre leakage adicional

Si la etiqueta final `riesgo_baja_transparencia` se construye a partir de completitud, trazabilidad, coherencia temporal y competencia, esas mismas variables no deben reutilizarse sin control como predictores del mismo modelo supervisado. En ese escenario hay dos opciones metodológicamente correctas:

- usar esos indicadores para construir la etiqueta y entrenar con variables anteriores al momento de etiquetado, o
- prescindir del modelo supervisado y tratar el puntaje como un índice directo de riesgo.

## 5. Variables derivadas (Feature Engineering)

Las variables derivadas deben aproximar transparencia sin depender de resultados posadjudicación.

| Variable derivada | Construcción | Justificación frente a transparencia |
| --- | --- | --- |
| `duracion_dias` | Normalizar `duracion` usando `unidad_de_duracion` | Permite comparar procesos en una sola escala temporal. |
| `dias_publicacion_a_recepcion` | `fecha_de_recepcion_de - fecha_de_publicacion_del` | Mide ventana efectiva para recibir respuestas u ofertas. |
| `dias_publicacion_a_apertura` | `fecha_de_apertura_de_respuesta - fecha_de_publicacion_del` | Aproxima oportunidad y secuencia del proceso. |
| `missing_temporal_count` | Conteo de hitos temporales ausentes | Resume debilidad de trazabilidad documental. |
| `anomalia_temporal` | Bandera ya derivada por el EDA | Captura incoherencia cronológica explícita. |
| `completitud_critica` | Proporción de campos clave presentes: entidad, código, categoría, valor, descripción, fecha ancla y hitos tempranos | Mide calidad mínima del registro para control ciudadano y auditoría. |
| `tokens_semanticos_faltantes` | Conteo de `No definido`, `No Aplica`, `No Adjudicado` u otros placeholders homologados | Separa ausencia real de falta de estandarización. |
| `precio_base_log` | `log1p(precio_base_clean)` | Estabiliza la fuerte asimetría de montos. |
| `flag_precio_cero` | `precio_base == 0` | Diferencia procesos sin monto informado o sin base monetaria explícita. |
| `competencia_reportada` | 1 si alguna variable de competencia es mayor que 0 | El EDA muestra que solo 18,7849% de los procesos reporta competencia observable. |
| `intensidad_competencia_log` | `log1p(proveedores_unicos_con + respuestas_al_procedimiento + conteo_de_respuestas_a_ofertas)` o una variante depurada | Resume apertura competitiva sin multiplicar variables casi redundantes. |
| `ratio_proveedores_vs_invitados` | `proveedores_unicos_con / max(proveedores_invitados, 1)` | Ayuda a distinguir invitación formal de participación efectiva. |
| `longitud_nombre` y `longitud_descripcion` | Longitud de `nombre_del_procedimiento` y `descripci_n_del_procedimiento` | El documento conceptual propone usar longitud textual como señal de calidad descriptiva. |
| `flag_id_proceso_duplicado` | 1 si el registro proviene de un grupo repetido por `id_del_proceso` | Conserva memoria de un problema de calidad relevante. |

Observaciones para uso prudente:

- `consistencia_estados` ya está en 99,9889% de los casos; sirve más como control de calidad que como predictor útil.
- `tiene_url`, `tiene_descripcion` y `tiene_ids_clave` son 100% verdaderas en el corte actual; no aportan discriminación en este dataset y deben quedar fuera del modelo, aunque sí son útiles como chequeos de pipeline.

## 7. Conclusiones

La calidad de la base SECOP II en este proyecto no está comprometida por una falta masiva de identificadores o campos estructurales, sino por tres focos concretos: trazabilidad temporal incompleta, duplicación lógica por proceso y presencia de variables posresultado que inducen fuga de información. Por eso, la limpieza útil para Machine Learning no consiste en “rellenar nulos” de manera general, sino en separar claramente qué columnas describen el proceso y cuáles revelan su desenlace.

La preparación recomendada deja una base modelable más coherente con el objetivo del proyecto: detectar riesgo de baja transparencia y no simplemente reconstruir adjudicaciones. En términos prácticos, esto implica eliminar variables de adjudicación y proveedor adjudicado, consolidar registros por `id_del_proceso`, normalizar duración y montos, convertir placeholders semánticos en señales explícitas y reemplazar variables casi constantes por indicadores derivados de completitud, trazabilidad y competencia.

Aplicada de esta manera, la limpieza fortalece la alineación con el ODS 16.6: mejora la validez de la evidencia, reduce sesgos por captura administrativa tardía y deja una estructura analítica defendible para auditoría, monitoreo institucional y modelado predictivo o semisupervisado.

## Referencias internas del proyecto

- `Docs/Project/informe_general_secop_ods16.pdf`
- `Data/Processed/ods16_secop/null_profile.csv`
- `Data/Processed/ods16_secop/univariate_numeric_summary.csv`
- `Data/Processed/ods16_secop/duplicates_summary.csv`
- `Data/Processed/ods16_secop/temporal_summary.csv`
- `Data/Processed/ods16_secop/temporal_anomalies.csv`
- `Data/Processed/ods16_secop/column_recommendations.csv`
- `Data/Processed/ods16_secop/transparency_quality_indicators.csv`
- `Data/Processed/ods16_secop/correlation_matrix.csv`
- `Data/Processed/ods16_secop/plots/numeric/precio_base_box.png`
- `Data/Processed/ods16_secop/plots/numeric/duracion_box.png`
- `Data/Processed/ods16_secop/plots/temporal/variables_temporales_vs_estado.png`
