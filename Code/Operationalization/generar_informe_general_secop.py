from __future__ import annotations

import os
import textwrap
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from xml.sax.saxutils import escape

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "Docs" / "Project"
BASE_NAME = "informe_general_proyecto_secop_ods16"

REPORT_DATE = "2026-05-03"
REPORT_TITLE = "Informe general del proyecto SECOP II y ODS 16"
REPORT_SUBTITLE = "Analitica, modelado y operacionalizacion de riesgo de baja transparencia"


def p(text: str) -> dict:
    return {"type": "paragraph", "text": text}


def h1(text: str) -> dict:
    return {"type": "heading1", "text": text}


def h2(text: str) -> dict:
    return {"type": "heading2", "text": text}


def bullets(items: list[str]) -> dict:
    return {"type": "bullets", "items": items}


def table(title: str, headers: list[str], rows: list[list[str]]) -> dict:
    return {"type": "table", "title": title, "headers": headers, "rows": rows}


def page_break() -> dict:
    return {"type": "page_break"}


REPORT_ELEMENTS = [
    {"type": "title", "title": REPORT_TITLE, "subtitle": REPORT_SUBTITLE},
    p(f"Fecha de generacion: {REPORT_DATE}."),
    p(
        "Proyecto academico de Big Data orientado a analizar procesos de contratacion publica "
        "de SECOP II y construir evidencia sobre transparencia, trazabilidad, competencia y "
        "calidad institucional en el marco del ODS 16.6."
    ),
    page_break(),
    h1("Resumen ejecutivo"),
    p(
        "El proyecto construye un flujo completo de datos para SECOP II: descarga e ingesta, "
        "limpieza, analisis descriptivo, analisis inferencial, ingenieria de variables, "
        "modelado predictivo y una aplicacion FastAPI para consumir un modelo de prediccion "
        "temprana de riesgo de baja transparencia."
    ),
    p(
        "La base cruda parte de 450.000 procesos y 58 columnas. Despues de depuracion y "
        "feature engineering se trabaja con 429.541 registros, 106 variables finales y "
        "artefactos analiticos en Data, Reports, Models y Docs. El enfoque principal no es "
        "predecir adjudicacion, sino construir y explotar senales de transparencia institucional."
    ),
    p(
        "El resultado mas importante es una arquitectura reproducible que combina indicadores "
        "de completitud, trazabilidad, coherencia temporal y competencia. El modelo temprano "
        "usa solo informacion disponible al inicio del proceso contractual, lo que reduce la "
        "dependencia de variables posteriores al evento y vuelve el caso de uso mas util para "
        "alertas preventivas."
    ),
    bullets(
        [
            "Datos procesados principales: 429.541 filas y 106 columnas en datos_feature_engineering.parquet.",
            "Riesgo legacy usado en modelado: 340.715 positivos, equivalentes al 79,32% de los registros.",
            "Target v2 estricto generado para control: 1.484 positivos, equivalentes al 0,35% de los registros.",
            "Modelo early seleccionado por el reporte de modelado: LogisticRegression_early con F1 de prueba 0,9941.",
            "Aplicacion operativa actual: FastAPI con RandomForest_early_pipeline.joblib, umbral 0,645 y endpoints web/API.",
        ]
    ),
    h1("1. Contexto y proposito"),
    p(
        "SECOP II concentra informacion publica sobre procesos contractuales del Estado. Para "
        "un proyecto alineado con el ODS 16.6, la pregunta central no es solamente cuanto se "
        "contrata, sino que tan completa, trazable, coherente y competitiva es la informacion "
        "que sustenta esos procesos."
    ),
    p(
        "El objetivo general del proyecto es construir un sistema analitico capaz de detectar "
        "senales de baja transparencia institucional en etapas tempranas del proceso, usando "
        "evidencia cuantitativa y modelos de aprendizaje automatico. El enfoque permite apoyar "
        "revision, priorizacion y auditoria de procesos que podrian requerir mayor seguimiento."
    ),
    table(
        "Tabla 1. Alcance funcional del proyecto",
        ["Componente", "Proposito", "Artefactos principales"],
        [
            [
                "Ingesta y datos crudos",
                "Obtener y conservar la fuente SECOP II original.",
                "Data/Raw/secop_procesos.parquet; Data/Raw/datos_analisis.parquet",
            ],
            [
                "Limpieza y preparacion",
                "Normalizar tipos, duplicados, nulos y columnas no aptas para modelado.",
                "Code/DataPrep/limpieza_datos_analisis_secop.py; Data/Processed/Limpieza/",
            ],
            [
                "EDA y descriptivo",
                "Caracterizar distribuciones, cobertura, calidad y senales institucionales.",
                "Data/Processed/Descriptivo/; Data/Processed/EDA/; Docs/DataReport/",
            ],
            [
                "Inferencial",
                "Evaluar asociaciones, diferencias de grupos e intervalos de confianza.",
                "Data/Processed/inferencial/; analisis_inferencial_secop.py",
            ],
            [
                "Modelado",
                "Entrenar y comparar modelos predictivos para riesgo de baja transparencia.",
                "Reports/Model/; Models/trained/; pipeline_modelado_secop.py",
            ],
            [
                "Operacionalizacion",
                "Servir predicciones tempranas mediante formulario web y API.",
                "Code/Operationalization/secop_fastapi/main.py; static/index.html",
            ],
        ],
    ),
    h1("2. Fuente de datos y evolucion del dataset"),
    p(
        "El proyecto conserva una trazabilidad clara entre la base cruda, la vista analitica, "
        "la base limpia y la base enriquecida. Esta separacion es importante porque permite "
        "distinguir datos originales, decisiones de limpieza, variables derivadas y artefactos "
        "orientados al modelo."
    ),
    table(
        "Tabla 2. Tamano de los principales conjuntos de datos",
        ["Archivo", "Filas", "Columnas", "Lectura"],
        [
            ["Data/Raw/secop_procesos.parquet", "450.000", "58", "Base principal cruda."],
            ["Data/Raw/datos_analisis.parquet", "450.000", "49", "Vista analitica sin campos posadjudicacion."],
            ["Data/Processed/Limpieza/datos_analisis_limpio.parquet", "429.541", "58", "Base depurada para analisis."],
            ["Data/Processed/Limpieza/datos_feature_engineering.parquet", "429.541", "106", "Base con variables, scores y targets."],
            ["Data/Processed/Model/train_early.parquet", "257.724", "16", "Particion de entrenamiento early."],
            ["Data/Processed/Model/validation_early.parquet", "85.908", "16", "Particion de validacion early."],
            ["Data/Processed/Model/test_early.parquet", "85.909", "16", "Particion de prueba early."],
        ],
    ),
    p(
        "A nivel de almacenamiento, el proyecto ocupa aproximadamente 610 MB en Data, 474 MB "
        "en Models, 90 MB en Reports, 2,2 MB en Docs y 1,4 MB en Code. Esto confirma que el "
        "repositorio contiene tanto pipeline reproducible como artefactos ya materializados."
    ),
    h1("3. Limpieza y calidad de datos"),
    p(
        "La etapa de limpieza identifica problemas propios de datos administrativos: nulos "
        "estructurales, codigos semanticos como 'No definido', duplicados logicos, colas largas "
        "en variables monetarias y temporales, y campos posadjudicacion que podrian contaminar "
        "un modelo predictivo si se usan antes de tiempo."
    ),
    p(
        "El diagnostico inicial encontro 18.910 duplicados exactos y 23.648 filas repetidas por "
        "id_del_proceso. En la base descriptiva posterior ya no aparecen duplicados exactos ni "
        "duplicados por id_del_proceso, aunque persisten referencias y portafolios repetidos "
        "que deben interpretarse como senales de seguimiento y no necesariamente como errores."
    ),
    table(
        "Tabla 3. Principales senales de calidad en la base enriquecida",
        ["Dimension", "Resultado observado", "Implicacion"],
        [
            ["Cobertura", "429.541 filas y 100 columnas en el perfil descriptivo.", "Volumen suficiente para analisis y ML."],
            ["Nulos altos", "categorias_adicionales 83,94%; fecha_de_apertura_de_respuesta 83,04%; fecha_de_recepcion_de 78,06%.", "La trazabilidad temporal es la dimension mas fragil."],
            ["Precio base", "Media 403,5 millones; mediana 14,5 millones; p95 496,0 millones.", "Distribucion muy asimetrica; conviene usar transformaciones robustas."],
            ["Duplicados", "0 duplicados exactos y 0 duplicados por id_del_proceso en la base final descriptiva.", "La limpieza redujo ruido fisico y logico clave."],
            ["Riesgo institucional", "nivel_riesgo_transparencia mas comun: medio, con 78,62%.", "La mayoria de procesos queda en zona intermedia de riesgo."],
            ["Confianza", "confianza_label mas comun: baja, con 61,73%.", "La calidad de evidencia exige cautela en lectura institucional."],
        ],
    ),
    p(
        "Una decision metodologica acertada es no imputar fechas originales de manera indiscriminada. "
        "La ausencia de fechas forma parte de la senal de trazabilidad; inventar fechas podria "
        "mejorar artificialmente la completitud y debilitar la interpretacion del riesgo."
    ),
    h1("4. Ingenieria de variables y construccion del target"),
    p(
        "El feature engineering transforma la base limpia en un conjunto con variables derivadas "
        "sobre calidad textual, presencia de campos requeridos, ubicacion, trazabilidad, ventanas "
        "temporales, competencia reportada y scores agregados. El pipeline procesa 429.541 filas "
        "y genera 106 columnas finales."
    ),
    table(
        "Tabla 4. Dimensiones centrales de feature engineering",
        ["Dimension", "Variables o scores", "Interpretacion"],
        [
            ["Completitud", "score_completitud, missing_required_fields_count, flags de presencia.", "Mide si el registro contiene campos institucionalmente necesarios."],
            ["Trazabilidad", "score_trazabilidad, flag_id_proceso_valido, flag_tiene_url_publica.", "Evalua si el proceso puede seguirse y verificarse."],
            ["Temporalidad", "dias_publicacion_a_recepcion, dias_recepcion_a_apertura, score_temporal.", "Resume coherencia y disponibilidad de hitos temporales."],
            ["Competencia", "total_respuestas, total_interes_oferentes, score_competencia.", "Aproxima participacion y concurrencia de oferentes."],
            ["Riesgo", "transparency_score, nivel_riesgo_transparencia, riesgo_baja_transparencia.", "Integra las dimensiones para etiquetar riesgo."],
        ],
    ),
    p(
        "El proyecto genera dos lecturas de etiqueta. El target legacy, usado por los reportes de "
        "modelado early, marca 340.715 positivos (79,32%). El target v2 es mucho mas estricto: "
        "1.484 positivos (0,35%), con controles de confianza y cobertura de evidencia. Esta "
        "doble lectura es valiosa porque permite comparar un criterio amplio de alerta contra "
        "un criterio conservador de riesgo extremo."
    ),
    h1("5. Analisis descriptivo"),
    p(
        "El analisis descriptivo confirma que SECOP II presenta alta heterogeneidad en entidades, "
        "modalidades, categorias y objetos contractuales. La variable nombre_del_procedimiento "
        "tiene 251.183 categorias y la entidad tiene 1.985 categorias, lo que explica la necesidad "
        "de modelos capaces de manejar variables categoricas de alta cardinalidad."
    ),
    bullets(
        [
            "El precio_base esta fuertemente sesgado: media muy superior a la mediana y presencia de outliers relevantes.",
            "El transparency_score tiene media 0,6409 y mediana 0,5900, con rango entre 0,3475 y 1,0000.",
            "score_completitud y score_trazabilidad son altos en promedio, pero score_temporal y score_competencia son bajos.",
            "La temporalidad disponible se concentra en fecha_de_publicacion_del; recepcion y apertura tienen cobertura parcial.",
        ]
    ),
    p(
        "La conclusion descriptiva es que el problema de transparencia no esta solamente en campos "
        "vacios, sino en la combinacion entre informacion incompleta, baja trazabilidad temporal, "
        "participacion reducida y registros con alta variabilidad semantica."
    ),
    h1("6. Analisis inferencial"),
    p(
        "El analisis inferencial revisa 100 variables candidatas, 25 comparaciones de grupos, "
        "66 asociaciones categoricas, 103 asociaciones numericas, 20 intervalos de confianza "
        "para medias y 40 intervalos para proporciones. Las pruebas se ejecutan con logica "
        "defensiva: Shapiro-Wilk para normalidad en muestras acotadas, Levene para homocedasticidad "
        "y enfasis en tamanos de efecto por el gran tamano muestral."
    ),
    table(
        "Tabla 5. Senales inferenciales destacadas",
        ["Relacion", "Metodo", "Resultado", "Lectura"],
        [
            ["duracion_dias vs tipo_de_contrato", "Kruskal-Wallis", "p=0; epsilon^2=0,0671", "Diferencias pequenas a moderadas segun tipo contractual."],
            ["duracion_dias vs justificaci_n_modalidad_de", "Kruskal-Wallis", "p=0; epsilon^2=0,0609", "La modalidad se asocia con duraciones distintas."],
            ["confianza_label vs competencia_reportada", "Chi-cuadrado", "p=0; V de Cramer=0,6567", "Asociacion fuerte entre confianza y competencia."],
            ["competencia_reportada vs anomalia_temporal", "Chi-cuadrado", "p=0; V de Cramer=0,3220", "Relacion moderada entre competencia y anomalias temporales."],
            ["conteo_de_respuestas_a_ofertas vs precio_base", "Pearson/Spearman", "r=0,0021; rho=0,0698", "Relacion practica muy baja aunque estadisticamente detectable."],
        ],
    ),
    p(
        "La lectura institucional debe evitar causalidad directa. Los resultados muestran patrones "
        "y asociaciones, pero deben interpretarse junto con calidad del dato, forma de construccion "
        "de los scores y contexto contractual."
    ),
    h1("7. Modelado predictivo"),
    p(
        "El pipeline de modelado compara regresion logistica, arboles, Random Forest, Gradient "
        "Boosting, XGBoost, LightGBM y CatBoost. El preprocesamiento se encapsula en Pipeline y "
        "ColumnTransformer, se ajusta solo con entrenamiento/CV, se evita SMOTE y se eliminan "
        "variables post-evento, metadata granular y columnas derivadas directamente del target."
    ),
    table(
        "Tabla 6. Resultado del modo early por validacion",
        ["Modelo", "Umbral", "Accuracy", "Precision", "Recall", "F1", "ROC AUC", "PR AUC"],
        [
            ["LogisticRegression_early", "0,270", "0,9912", "0,9955", "0,9934", "0,9944", "0,9997", "0,9999"],
            ["XGBoost_early", "0,285", "0,9890", "0,9928", "0,9933", "0,9931", "0,9996", "0,9999"],
            ["LightGBM_early", "0,390", "0,9890", "0,9954", "0,9908", "0,9931", "0,9995", "0,9999"],
            ["RandomForest_early", "0,645", "0,9885", "0,9926", "0,9929", "0,9928", "0,9995", "0,9999"],
            ["CatBoost_early", "0,325", "0,9882", "0,9925", "0,9926", "0,9926", "0,9995", "0,9999"],
        ],
    ),
    p(
        "En prueba, LogisticRegression_early mantiene Accuracy 0,9906, Precision 0,9951, Recall "
        "0,9931, F1 0,9941, ROC AUC 0,9996 y PR AUC 0,9999. La validacion cruzada reporta "
        "F1 medio 0,9936 con desviacion 0,0002."
    ),
    table(
        "Tabla 7. Comparacion normal, strict y early",
        ["Metrica", "Normal", "Strict", "Early", "Caida early vs strict"],
        [
            ["F1", "0,9989", "0,9989", "0,9944", "0,0045"],
            ["Recall", "0,9996", "0,9990", "0,9934", "0,0056"],
            ["Precision", "0,9983", "0,9989", "0,9955", "0,0033"],
            ["ROC AUC", "1,0000", "1,0000", "0,9997", "0,0003"],
            ["PR AUC", "1,0000", "1,0000", "0,9999", "0,0001"],
        ],
    ),
    p(
        "Las metricas son extremadamente altas y los reportes generan advertencias automaticas "
        "de posible leakage por ROC AUC y PR AUC superiores a 0,995. El modo early mitiga parte "
        "del riesgo porque restringe variables a informacion inicial, pero aun asi se recomienda "
        "validar la etiqueta, los umbrales y el comportamiento fuera de muestra antes de un uso "
        "institucional."
    ),
    h2("Variables tempranas usadas"),
    bullets(
        [
            "entidad, nit_entidad, departamento_entidad, ciudad_entidad y ordenentidad.",
            "nombre_del_procedimiento y descripci_n_del_procedimiento.",
            "fase, precio_base y precio_base_log.",
            "modalidad_de_contratacion, justificaci_n_modalidad_de, codigo_principal_de_categoria, tipo_de_contrato y categorias_adicionales.",
        ]
    ),
    h1("8. Operacionalizacion"),
    p(
        "La operacionalizacion se materializa en una aplicacion FastAPI llamada SECOP Transparencia "
        "Temprana. La API carga un pipeline serializado, recibe variables del proceso, calcula "
        "precio_base_log, estima probabilidad de riesgo y devuelve probabilidad, etiqueta, clase "
        "y factores clave de importancia."
    ),
    table(
        "Tabla 8. Endpoints y comportamiento de la API",
        ["Endpoint", "Metodo", "Uso"],
        [
            ["/", "GET", "Sirve el formulario web estatico."],
            ["/api/health", "GET", "Reporta estado del servicio, ruta del modelo, umbral y variables."],
            ["/api/predict", "POST", "Recibe JSON de proceso SECOP II y retorna prediccion de riesgo."],
        ],
    ),
    p(
        "El codigo operativo actual carga Models/trained/all_models/RandomForest_early_pipeline.joblib "
        "y usa umbral 0,645. Este artefacto coincide con una opcion robusta del experimento early, "
        "pero el reporte de seleccion marca LogisticRegression_early como mejor modelo por F1. "
        "Antes de cerrar una version productiva conviene documentar si la eleccion de Random Forest "
        "fue intencional o alinear la API con best_model_pipeline_early.joblib."
    ),
    p(
        "El proyecto tambien incluye docker-compose.yml para PostgreSQL y un flujo de carga que "
        "reporta estado uploaded, 429.541 filas cargadas y tabla secop_feature_engineering. Esto "
        "permite conectar el pipeline analitico con una capa persistente para consultas o integracion."
    ),
    h1("9. Arquitectura tecnica y reproducibilidad"),
    p(
        "La organizacion del repositorio es clara: Code contiene scripts de datos, analisis, modelado "
        "y API; Data separa Raw, Processed, EDA, Descriptivo, inferencial y Model; Reports agrupa "
        "metricas, matrices, curvas, feature importance y validaciones; Models conserva pipelines "
        "serializados; Docs contiene informes academicos y tecnicos."
    ),
    bullets(
        [
            "Dependencias principales: pandas, pyarrow, scipy, scikit-learn, joblib, matplotlib, seaborn, statsmodels, openpyxl, Prefect, FastAPI, Uvicorn y Pydantic.",
            "Los modelos se guardan como pipelines joblib, lo que reduce errores entre entrenamiento y consumo.",
            "Los reportes de modelado incluyen validacion cruzada, optimizacion de umbral, curvas ROC/PR, matrices de confusion y feature importance.",
            "La API evita recalcular features complejas y se concentra en variables tempranas disponibles para el formulario.",
        ]
    ),
    h1("10. Riesgos, limitaciones y controles necesarios"),
    table(
        "Tabla 9. Riesgos principales del proyecto",
        ["Riesgo", "Evidencia", "Control recomendado"],
        [
            ["Leakage o target reconstruible", "Metricas cercanas a 1,0 y advertencias automaticas.", "Validar reglas del target, hacer pruebas temporales y evaluar target v2."],
            ["Desbalance de etiquetas", "Target legacy 79,32% positivo; target v2 0,35% positivo.", "Elegir target segun caso de uso: alerta amplia o riesgo extremo."],
            ["Nulos temporales estructurales", "Recepcion y apertura tienen alta ausencia.", "No imputar fechas originales sin justificacion; usar banderas y ventanas derivadas."],
            ["Alta cardinalidad categorica", "Nombre de procedimiento y descripcion tienen cientos de miles de valores.", "Controlar codificacion, regularizacion y generalizacion a entidades nuevas."],
            ["Diferencia entre modelo seleccionado y modelo servido", "Reporte early selecciona LogisticRegression; API sirve RandomForest.", "Alinear artefacto operativo o documentar criterio de seleccion."],
            ["Interpretacion causal indebida", "Inferencias son asociaciones con muestras grandes.", "Reportar tamanos de efecto y contexto institucional."],
        ],
    ),
    h1("11. Recomendaciones"),
    bullets(
        [
            "Definir formalmente que target se usara para la version final: legacy para alertas amplias o v2 para riesgo extremo conservador.",
            "Ejecutar una validacion temporal estricta por fecha de publicacion para comprobar generalizacion en periodos futuros.",
            "Alinear la API con el mejor artefacto seleccionado o justificar por escrito la preferencia por RandomForest_early.",
            "Agregar pruebas automaticas del endpoint /api/predict con payloads representativos y casos borde de precio, categoria y campos faltantes.",
            "Crear una ficha de modelo con variables permitidas, umbral, distribucion del target, advertencias de leakage y uso previsto.",
            "Complementar las metricas globales con analisis por departamento, entidad, modalidad y tipo de contrato para revisar sesgos de desempeno.",
            "Mantener informes PDF/Word como entregables, pero preservar Markdown y CSV como fuentes reproducibles.",
        ]
    ),
    h1("12. Conclusiones"),
    p(
        "El proyecto esta bastante avanzado y cubre el ciclo completo de un caso de Big Data aplicado: "
        "datos, calidad, analitica, inferencia, modelado y despliegue. Su mayor fortaleza es que "
        "traduce la transparencia institucional en dimensiones medibles y conserva artefactos "
        "reproducibles para auditar cada etapa."
    ),
    p(
        "La principal precaucion es metodologica: las metricas predictivas son tan altas que deben "
        "interpretarse como una senal para revisar el target, la separacion temporal y la posible "
        "dependencia entre variables tempranas y reglas de etiquetado. Con esa validacion, el "
        "proyecto puede sostener un entregable academico robusto y una demostracion operativa clara."
    ),
    h1("Anexos. Fuentes internas consultadas"),
    bullets(
        [
            "Docs/DataReport/informe_limpieza_preparacion_ml_secop.md",
            "Data/Processed/Descriptivo/executive_summary.md",
            "Data/Processed/inferencial/executive_summary.md",
            "Reports/FeatureEngineering/feature_engineering_summary.txt",
            "Reports/FeatureEngineering/target_v2_label_quality_report.txt",
            "Reports/Model/best_model_report_early.txt",
            "Reports/Model/metrics_summary_early.csv",
            "Reports/Model/cross_validation_summary_early.csv",
            "Reports/Model/early_feature_policy_report.txt",
            "Code/Operationalization/secop_fastapi/main.py",
            "Code/Operationalization/secop_fastapi/README.md",
            "docker-compose.yml y requirements.txt",
        ]
    ),
]


def write_markdown(path: Path) -> None:
    lines: list[str] = []
    for element in REPORT_ELEMENTS:
        kind = element["type"]
        if kind == "title":
            lines.append(f"# {element['title']}")
            lines.append("")
            lines.append(f"**{element['subtitle']}**")
            lines.append("")
        elif kind == "heading1":
            lines.append(f"## {element['text']}")
            lines.append("")
        elif kind == "heading2":
            lines.append(f"### {element['text']}")
            lines.append("")
        elif kind == "paragraph":
            lines.append(element["text"])
            lines.append("")
        elif kind == "bullets":
            lines.extend(f"- {item}" for item in element["items"])
            lines.append("")
        elif kind == "table":
            lines.append(f"**{element['title']}**")
            lines.append("")
            headers = element["headers"]
            lines.append("| " + " | ".join(headers) + " |")
            lines.append("| " + " | ".join("---" for _ in headers) + " |")
            for row in element["rows"]:
                lines.append("| " + " | ".join(row) + " |")
            lines.append("")
        elif kind == "page_break":
            lines.append("\\newpage")
            lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def w_text(text: str) -> str:
    return escape(text)


def w_run(text: str, bold: bool = False, size: int | None = None) -> str:
    props = ""
    if bold or size:
        prop_parts = []
        if bold:
            prop_parts.append("<w:b/>")
        if size:
            prop_parts.append(f'<w:sz w:val="{size}"/><w:szCs w:val="{size}"/>')
        props = "<w:rPr>" + "".join(prop_parts) + "</w:rPr>"
    return f"<w:r>{props}<w:t xml:space=\"preserve\">{w_text(text)}</w:t></w:r>"


def w_paragraph(text: str, style: str = "Normal", bold: bool = False) -> str:
    return (
        "<w:p>"
        f"<w:pPr><w:pStyle w:val=\"{style}\"/></w:pPr>"
        f"{w_run(text, bold=bold)}"
        "</w:p>"
    )


def w_page_break() -> str:
    return '<w:p><w:r><w:br w:type="page"/></w:r></w:p>'


def w_table(headers: list[str], rows: list[list[str]]) -> str:
    border = (
        '<w:tblBorders><w:top w:val="single" w:sz="4" w:space="0" w:color="999999"/>'
        '<w:left w:val="single" w:sz="4" w:space="0" w:color="999999"/>'
        '<w:bottom w:val="single" w:sz="4" w:space="0" w:color="999999"/>'
        '<w:right w:val="single" w:sz="4" w:space="0" w:color="999999"/>'
        '<w:insideH w:val="single" w:sz="4" w:space="0" w:color="999999"/>'
        '<w:insideV w:val="single" w:sz="4" w:space="0" w:color="999999"/></w:tblBorders>'
    )
    parts = [f"<w:tbl><w:tblPr><w:tblStyle w:val=\"TableGrid\"/>{border}</w:tblPr>"]
    for idx, row in enumerate([headers] + rows):
        parts.append("<w:tr>")
        for cell in row:
            shading = '<w:shd w:fill="D9EAF7"/>' if idx == 0 else ""
            parts.append(
                "<w:tc>"
                f"<w:tcPr>{shading}<w:tcW w:w=\"2400\" w:type=\"dxa\"/></w:tcPr>"
                f"{w_paragraph(str(cell), 'TableText', bold=(idx == 0))}"
                "</w:tc>"
            )
        parts.append("</w:tr>")
    parts.append("</w:tbl>")
    return "".join(parts)


def document_xml() -> str:
    body_parts: list[str] = []
    for element in REPORT_ELEMENTS:
        kind = element["type"]
        if kind == "title":
            body_parts.append(w_paragraph(element["title"], "Title"))
            body_parts.append(w_paragraph(element["subtitle"], "Subtitle"))
        elif kind == "heading1":
            body_parts.append(w_paragraph(element["text"], "Heading1"))
        elif kind == "heading2":
            body_parts.append(w_paragraph(element["text"], "Heading2"))
        elif kind == "paragraph":
            body_parts.append(w_paragraph(element["text"]))
        elif kind == "bullets":
            for item in element["items"]:
                body_parts.append(w_paragraph(f"- {item}", "ListParagraph"))
        elif kind == "table":
            body_parts.append(w_paragraph(element["title"], "Caption", bold=True))
            body_parts.append(w_table(element["headers"], element["rows"]))
            body_parts.append(w_paragraph(""))
        elif kind == "page_break":
            body_parts.append(w_page_break())
    sect = (
        '<w:sectPr><w:pgSz w:w="11906" w:h="16838"/>'
        '<w:pgMar w:top="1134" w:right="1134" w:bottom="1134" w:left="1134" '
        'w:header="708" w:footer="708" w:gutter="0"/></w:sectPr>'
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:wpc="http://schemas.microsoft.com/office/word/2010/wordprocessingCanvas" '
        'xmlns:mc="http://schemas.openxmlformats.org/markup-compatibility/2006" '
        'xmlns:o="urn:schemas-microsoft-com:office:office" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
        'xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math" '
        'xmlns:v="urn:schemas-microsoft-com:vml" '
        'xmlns:wp14="http://schemas.microsoft.com/office/word/2010/wordprocessingDrawing" '
        'xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing" '
        'xmlns:w10="urn:schemas-microsoft-com:office:word" '
        'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
        'xmlns:w14="http://schemas.microsoft.com/office/word/2010/wordml" '
        'xmlns:wpg="http://schemas.microsoft.com/office/word/2010/wordprocessingGroup" '
        'xmlns:wpi="http://schemas.microsoft.com/office/word/2010/wordprocessingInk" '
        'xmlns:wne="http://schemas.microsoft.com/office/word/2006/wordml" '
        'xmlns:wps="http://schemas.microsoft.com/office/word/2010/wordprocessingShape" '
        'mc:Ignorable="w14 wp14"><w:body>'
        + "".join(body_parts)
        + sect
        + "</w:body></w:document>"
    )


def styles_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:docDefaults><w:rPrDefault><w:rPr><w:rFonts w:ascii="Calibri" w:hAnsi="Calibri"/><w:sz w:val="22"/></w:rPr></w:rPrDefault></w:docDefaults>
  <w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/><w:qFormat/><w:pPr><w:spacing w:after="160" w:line="276" w:lineRule="auto"/></w:pPr></w:style>
  <w:style w:type="paragraph" w:styleId="Title"><w:name w:val="Title"/><w:basedOn w:val="Normal"/><w:qFormat/><w:pPr><w:spacing w:after="240"/></w:pPr><w:rPr><w:b/><w:sz w:val="36"/></w:rPr></w:style>
  <w:style w:type="paragraph" w:styleId="Subtitle"><w:name w:val="Subtitle"/><w:basedOn w:val="Normal"/><w:qFormat/><w:rPr><w:color w:val="555555"/><w:sz w:val="24"/></w:rPr></w:style>
  <w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/><w:basedOn w:val="Normal"/><w:qFormat/><w:pPr><w:spacing w:before="320" w:after="160"/></w:pPr><w:rPr><w:b/><w:color w:val="1F4E79"/><w:sz w:val="30"/></w:rPr></w:style>
  <w:style w:type="paragraph" w:styleId="Heading2"><w:name w:val="heading 2"/><w:basedOn w:val="Normal"/><w:qFormat/><w:pPr><w:spacing w:before="240" w:after="120"/></w:pPr><w:rPr><w:b/><w:color w:val="2F75B5"/><w:sz w:val="25"/></w:rPr></w:style>
  <w:style w:type="paragraph" w:styleId="Caption"><w:name w:val="Caption"/><w:basedOn w:val="Normal"/><w:rPr><w:b/><w:color w:val="404040"/></w:rPr></w:style>
  <w:style w:type="paragraph" w:styleId="ListParagraph"><w:name w:val="List Paragraph"/><w:basedOn w:val="Normal"/><w:pPr><w:ind w:left="360"/></w:pPr></w:style>
  <w:style w:type="paragraph" w:styleId="TableText"><w:name w:val="Table Text"/><w:basedOn w:val="Normal"/><w:pPr><w:spacing w:after="40" w:line="240" w:lineRule="auto"/></w:pPr><w:rPr><w:sz w:val="18"/></w:rPr></w:style>
  <w:style w:type="table" w:styleId="TableGrid"><w:name w:val="Table Grid"/><w:tblPr><w:tblBorders><w:top w:val="single" w:sz="4" w:color="auto"/><w:left w:val="single" w:sz="4" w:color="auto"/><w:bottom w:val="single" w:sz="4" w:color="auto"/><w:right w:val="single" w:sz="4" w:color="auto"/><w:insideH w:val="single" w:sz="4" w:color="auto"/><w:insideV w:val="single" w:sz="4" w:color="auto"/></w:tblBorders></w:tblPr></w:style>
</w:styles>
"""


def write_docx(path: Path) -> None:
    created = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    content_types = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
</Types>
"""
    root_rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>
"""
    doc_rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>
"""
    core = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:dcmitype="http://purl.org/dc/dcmitype/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:title>{w_text(REPORT_TITLE)}</dc:title>
  <dc:creator>Codex</dc:creator>
  <cp:lastModifiedBy>Codex</cp:lastModifiedBy>
  <dcterms:created xsi:type="dcterms:W3CDTF">{created}</dcterms:created>
  <dcterms:modified xsi:type="dcterms:W3CDTF">{created}</dcterms:modified>
</cp:coreProperties>
"""
    app = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <Application>Codex</Application>
</Properties>
"""
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as docx:
        docx.writestr("[Content_Types].xml", content_types)
        docx.writestr("_rels/.rels", root_rels)
        docx.writestr("word/_rels/document.xml.rels", doc_rels)
        docx.writestr("word/document.xml", document_xml())
        docx.writestr("word/styles.xml", styles_xml())
        docx.writestr("docProps/core.xml", core)
        docx.writestr("docProps/app.xml", app)


def pdf_lines_for_table(headers: list[str], rows: list[list[str]], width: int = 110) -> list[str]:
    cols = len(headers)
    col_width = max(12, width // cols - 3)
    lines = []
    all_rows = [headers] + rows
    for idx, row in enumerate(all_rows):
        wrapped_cells = [textwrap.wrap(str(cell), col_width) or [""] for cell in row]
        max_lines = max(len(cell_lines) for cell_lines in wrapped_cells)
        for line_idx in range(max_lines):
            parts = []
            for cell_lines in wrapped_cells:
                value = cell_lines[line_idx] if line_idx < len(cell_lines) else ""
                parts.append(value[:col_width].ljust(col_width))
            lines.append(" | ".join(parts).rstrip())
        if idx == 0:
            lines.append("-" * min(width, cols * (col_width + 3)))
    return lines


class PdfWriter:
    def __init__(self, path: Path):
        self.path = path
        self.pdf = PdfPages(path)
        self.fig = None
        self.y = 0.92
        self.page = 0
        self.new_page()

    def new_page(self):
        if self.fig is not None:
            self.pdf.savefig(self.fig, bbox_inches="tight")
            plt.close(self.fig)
        self.fig = plt.figure(figsize=(8.27, 11.69))
        self.fig.patch.set_facecolor("white")
        self.y = 0.92
        self.page += 1
        self.fig.text(0.08, 0.965, REPORT_TITLE, fontsize=8, color="#666666")
        self.fig.text(0.90, 0.035, str(self.page), fontsize=8, color="#666666", ha="right")

    def ensure_space(self, needed: float):
        if self.y - needed < 0.08:
            self.new_page()

    def text(self, text: str, size: float = 9.5, bold: bool = False, color: str = "#222222", wrap: int = 102, indent: float = 0.0):
        lines = textwrap.wrap(text, wrap) or [""]
        line_height = size / 900
        self.ensure_space(line_height * len(lines) + 0.02)
        for line in lines:
            self.fig.text(
                0.08 + indent,
                self.y,
                line,
                fontsize=size,
                color=color,
                weight="bold" if bold else "normal",
                ha="left",
                va="top",
            )
            self.y -= line_height
        self.y -= 0.012

    def title(self, title: str, subtitle: str):
        self.fig.text(0.08, 0.70, title, fontsize=22, weight="bold", color="#1F4E79", ha="left", va="top")
        self.fig.text(0.08, 0.655, subtitle, fontsize=13, color="#444444", ha="left", va="top")
        self.fig.text(0.08, 0.615, f"Fecha de generacion: {REPORT_DATE}", fontsize=9.5, color="#666666", ha="left", va="top")
        self.y = 0.55

    def h1(self, text: str):
        self.ensure_space(0.06)
        self.y -= 0.01
        self.fig.text(0.08, self.y, text, fontsize=15, weight="bold", color="#1F4E79", ha="left", va="top")
        self.y -= 0.038

    def h2(self, text: str):
        self.ensure_space(0.05)
        self.fig.text(0.08, self.y, text, fontsize=12, weight="bold", color="#2F75B5", ha="left", va="top")
        self.y -= 0.032

    def bullet(self, item: str):
        self.text(f"- {item}", size=9.2, wrap=96, indent=0.02)

    def table(self, title: str, headers: list[str], rows: list[list[str]]):
        self.text(title, size=9, bold=True, color="#404040", wrap=100)
        lines = pdf_lines_for_table(headers, rows)
        line_height = 0.011
        for line in lines:
            self.ensure_space(line_height + 0.006)
            self.fig.text(0.08, self.y, line, fontsize=6.8, family="DejaVu Sans Mono", color="#222222", ha="left", va="top")
            self.y -= line_height
        self.y -= 0.012

    def close(self):
        if self.fig is not None:
            self.pdf.savefig(self.fig, bbox_inches="tight")
            plt.close(self.fig)
        self.pdf.close()


def write_pdf(path: Path) -> None:
    writer = PdfWriter(path)
    try:
        for element in REPORT_ELEMENTS:
            kind = element["type"]
            if kind == "title":
                writer.title(element["title"], element["subtitle"])
            elif kind == "heading1":
                writer.h1(element["text"])
            elif kind == "heading2":
                writer.h2(element["text"])
            elif kind == "paragraph":
                writer.text(element["text"])
            elif kind == "bullets":
                for item in element["items"]:
                    writer.bullet(item)
            elif kind == "table":
                writer.table(element["title"], element["headers"], element["rows"])
            elif kind == "page_break":
                writer.new_page()
    finally:
        writer.close()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    md_path = OUTPUT_DIR / f"{BASE_NAME}.md"
    docx_path = OUTPUT_DIR / f"{BASE_NAME}.docx"
    pdf_path = OUTPUT_DIR / f"{BASE_NAME}.pdf"
    write_markdown(md_path)
    write_docx(docx_path)
    write_pdf(pdf_path)
    print(md_path)
    print(docx_path)
    print(pdf_path)


if __name__ == "__main__":
    main()
