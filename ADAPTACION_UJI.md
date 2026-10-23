# Adaptación al corpus UJI (inglés → catalán/valenciano) con la recompensa IPD

Este documento explica todo lo que se ha cambiado en el código para poder entrenar y
evaluar con el corpus UJI valenciano–inglés
(`data/va_en_uji/va-en-sentence.length.ner.curated.jsonl`) y la recompensa IPD de
`src/ipd_reward.py`, en lugar de FLORES-200.

Los cambios son **retrocompatibles**: sin `--dataset`, `main.py`, `generate_translations.py`
y `decoding_diagnostic.py` se comportan exactamente igual que antes (FLORES-200). No se ha
tocado `src/trainer.py`, `src/reward.py`, `src/ipd_reward.py` ni `metrics/` en las secciones
1–6; la sección 7 describe un cambio posterior (métricas de seguimiento durante el
entrenamiento) que sí los toca.

## 1. Planteamiento del experimento

| | |
|---|---|
| Modelo | NLLB-200 (600M o 1.3B), igual que antes |
| Dirección | **inglés (`eng_Latn`) → catalán (`cat_Latn`)** |
| Entrada (`src`) | la frase en inglés (`EN` del jsonl) |
| Referencia (`ref`) | la frase en valenciano (`VA` del jsonl) |
| Recompensa | `--reward ipd` (IPD de AVLizador sobre la hipótesis; no mira el origen) |

**Interpretación que he tomado.** El dataset se describe como «valenciano a inglés», pero
la recompensa IPD puntúa texto *catalán* generado, y NLLB solo tiene un código para esa
lengua (`cat_Latn`, variante oriental/IEC). Por tanto el modelo traduce **de inglés a
catalán** y el lado valenciano del corpus se usa como referencia. Así la adaptación
«catalán → valenciano» ocurre en el lado de salida: el IPD empuja la salida hacia formas
valencianas y el chrF++ contra referencias valencianas (que antes, con FLORES, penalizaba
justo lo contrario, ver REPRODUCTION.md) ahora sí premia lo mismo que el IPD. Si en
realidad se quería otra dirección (p. ej. VA → EN), habría que cambiarlo, pero la recompensa
IPD no tendría sentido con salida en inglés.

## 2. El formato del dataset y por qué hacía falta convertirlo

FLORES se usaba como `data/<dataset>/<lang>/{dev,devtest}.parquet` con columnas `src`, `ref`,
y `main.py` entrenaba **y** evaluaba sobre `dev.parquet` (997 frases). El corpus UJI es
distinto:

- Es un `.jsonl` (una línea = un objeto JSON) con campos `VA`, `EN` y `source`
  (el fichero de la noticia de la que sale la frase, p. ej. `noticia2116.jsonl`).
- 10 836 pares, 4 078 noticias distintas (≈2,7 frases por noticia), sin pares vacíos ni
  `VA` duplicados.
- **No trae particiones** train/dev/test, mientras que `main.py` necesitaba un único
  `dev.parquet`, y `generate_translations.py` un `dev.parquet` + `devtest.parquet`.
- Frases bastante más largas que FLORES (media ≈ 31 palabras VA / 29 EN, máximo 129/144).
- Casi la mitad de las frases (4 853 de 10 836 en el lado VA) acaban en `..` (doble punto),
  un artefacto de construcción del corpus que no está en FLORES.

## 3. Cambios, fichero a fichero

### 3.1 `prepare_uji.py` (nuevo)

Convierte el jsonl al mismo formato que usa el resto del código:

```bash
uv run python prepare_uji.py            # opciones: --n_devtest 1000 --n_dev 500 --seed 42 ...
```

Escribe `data/uji_eng/cat_Latn/{train,dev,devtest}.parquet` con columnas `src` (EN), `ref`
(VA) y `article` (la noticia de origen, solo informativa; el código ignora esa columna).
La estructura espeja `data/flores200_eng/cat_Latn/`.

- **Partición por noticia**, no por frase: todas las frases de una noticia van al mismo
  split. Si se partiera por frase, frases contiguas de un mismo texto caerían en train y
  en test (fuga de datos). Se barajan las noticias con `--seed` (42 por defecto, reproducible),
  se llena primero `devtest`, luego `dev` y el resto va a `train`. Comprobado: 0 noticias
  compartidas entre splits.
- **Tamaños** (por defecto): `devtest` 1 001 frases (similar a las 1 012 de FLORES devtest),
  `dev` 500, `train` 9 335. Cada split puede pasarse del objetivo por la última noticia.
  Dev se usa para (a) elegir el ajuste de decodificación (primeras 100 frases) y (b) la
  evaluación intermedia que escoge `adapters_best`; 500 sobran para eso, y deja más datos
  para entrenar.
- Las frases de cada split se barajan (con la misma semilla). Importa porque el entrenamiento
  recorre el parquet en orden, una frase por paso: sin barajar vería noticias enteras
  seguidas, y las 100 primeras frases de dev (para ajustar la decodificación) saldrían de
  muy pocas noticias.
- **Limpieza**: el `..` final se colapsa a `.` en ambos lados (se mantiene `...`, que es
  una elipsis real). `--keep_double_period` desactiva esto y deja el texto tal cual.
  La razón es que el modelo generará un solo `.` y el chrF++ contra una referencia con `..`
  lo penalizaría sin motivo (y el `..` en la entrada es ruido). Se hace `strip()` de los
  espacios. Se descartan pares con algún lado vacío (en este fichero: 0).
- Lectura en UTF-8 (`utf-8-sig`): el fichero tiene tildes, `ç`, `l·l`, etc.

Los parquet generados (`data/uji_eng/`) ya están creados en el árbol de trabajo (sin
commit). Se pueden regenerar en cualquier momento con el comando de arriba.

### 3.2 `src/data.py` (nuevo)

Registro pequeño de datasets que usan los tres scripts, en vez de tener la tabla
`SOURCE_TO_DATA_DIR` copiada en cada uno:

```python
DATASETS = {
    "flores": dict(data_dirs={"en": "data/flores200_eng", "es": "data/flores200_spa"},
                   train_split="dev", eval_split="dev"),
    "uji":    dict(data_dirs={"en": "data/uji_eng"}, train_split="train", eval_split="dev"),
}
resolve_data_dir(dataset, source, data_dir=None)   # --data_dir si se da; si no, el del dataset
```

`uji` solo tiene origen `en`; `--dataset uji --source es` falla con un mensaje claro.

### 3.3 `main.py` (entrenamiento)

Argumentos nuevos:

| Argumento | Por defecto | Qué hace |
|---|---|---|
| `--dataset {flores,uji}` | `flores` | Elige directorio de datos y splits por defecto |
| `--train_split` | `dev` (flores) / `train` (uji) | Split sobre el que se entrena |
| `--eval_split` | `dev` | Split del que la evaluación intermedia saca su muestra |

Otros cambios:

- Antes: `df_train = dev.parquet` y `df_eval = df_train`. Ahora se cargan por separado.
  Si `eval_split == train_split` (caso FLORES) se reutiliza `df_train` tal cual, incluido el
  recorte de `--n_segments`, así que **el comportamiento de FLORES no cambia**. Con UJI, la
  evaluación intermedia usa `dev.parquet`, que no se ha visto en entrenamiento, así que
  `adapters_best` no se elige sobre datos de entrenamiento.
- `--n_segments` sigue recortando solo los datos de entrenamiento.
- Directorio de salida por defecto: `results/{lang}_{scale}_{source}_{reward}` para FLORES
  (como antes) y `results/{lang}_{scale}_{source}_{reward}_uji` para UJI, para que no se
  pisen resultados de runs distintos.
- `config.txt` guarda además `resolved_data_dir`, `resolved_train_split` y
  `resolved_eval_split`; la cabecera impresa muestra el dataset, los splits y el nº de
  filas de evaluación.
- El mensaje final con el comando de evaluación incluye `--dataset` y, si el destino es
  `cat_Latn`, `--ipd`.

### 3.4 `generate_translations.py` (evaluación)

- Nuevo `--dataset {flores,uji}` (resuelve el directorio con `src/data.py`). Sigue usando
  `dev.parquet` (primeras 100 frases, para ajustar la decodificación) y `devtest.parquet`
  (test), que `prepare_uji.py` genera con el mismo significado.
- `--output_dir` ahora por defecto es `results/translations` para FLORES (como antes) y
  `results/translations_uji` para UJI, para no sobrescribir `cat_Latn_summary.json` de
  FLORES.
- `{lang}_summary.json` incluye el campo `dataset`.
- `--ipd` ya funcionaba y no se ha tocado: calcula el IPD corpus-level de la referencia, del
  vanilla y del adaptador. Con UJI la referencia es valenciana, así que `reference_ipd`
  pasa a ser el techo de referencia razonable (con FLORES era ≈ 0,19).

### 3.5 `decoding_diagnostic.py`

Solo `--dataset` y el sufijo `_uji` en el directorio de salida por defecto, para poder
calcular el vanilla sobre el devtest de UJI.

### 3.6 No modificado

`src/trainer.py` (el bucle itera `df_train.itertuples()` con `row.src`, válido para
cualquier parquet con `src`/`ref`), `src/models.py`, `src/reward.py`, `src/ipd_reward.py`,
`metrics/*`, `prepare_data.py` (sigue siendo solo FLORES) y `slurm/run.sbatch`.

## 4. Cómo ejecutarlo

```bash
# 0) Datos (ya generados; solo si se quieren regenerar)
uv run python prepare_uji.py

# 1) Baseline: NLLB vanilla en en->cat_Latn, con el IPD de la referencia valenciana
uv run python generate_translations.py --lang cat_Latn --scale 600m --dataset uji \
    --setting "default(beam5,lp1.0)" --ipd

# 2) Entrenamiento con la recompensa IPD
uv run python main.py --lang cat_Latn --scale 600m --dataset uji --reward ipd \
    --n_epochs 1 --seed 42 --max_new_tokens_train 96

# 3) Evaluación del mejor checkpoint (vs vanilla), chrF++ e IPD
uv run python generate_translations.py \
    --adapter_path results/cat_Latn_600m_en_ipd_uji/adapters_best/cat_Latn \
    --lang cat_Latn --scale 600m --dataset uji --setting "default(beam5,lp1.0)" --ipd
```

En SLURM, lo mismo con `sbatch --job-name=... slurm/run.sbatch main.py ...` como en el
README. Los flags de `--dataset uji` son los únicos nuevos; el resto de hiperparámetros
(`--k`, `--lr`, `--kl_beta`, LoRA, temperatura...) funcionan igual.

## 5. Cosas a tener en cuenta (no son cambios de código)

1. **Longitud máxima de generación en entrenamiento.** Medido con el tokenizador de NLLB
   sobre `train`: la referencia catalana tiene de media 52 tokens; el 25 % supera 64 (el
   valor por defecto de `--max_new_tokens_train`), el 5 % supera 96 y el 0,7 % supera 128.
   Con 64, una cuarta parte de las hipótesis muestreadas se cortan a mitad de frase durante
   el entrenamiento (las frases de FLORES son más cortas; no lo he medido). Por eso el ejemplo usa 96. No lo he
   cambiado por defecto porque aumenta la memoria del paso (logits de 12 × tokens × 256k de
   vocabulario) y las GPU de 11 GB del clúster están justas; hay que probarlo ahí y bajar
   a 80 o a `--k` menor si hay OOM. `--max_new_tokens_eval` (128) cubre el 99 %.
2. **Tiempo de entrenamiento.** `train` tiene 9 335 frases frente a ~1 000 de FLORES dev.
   Con ~6–7 s/paso (medido en T4 con la recompensa híbrida, ver REPRODUCTION.md; el IPD es
   probablemente más barato que LaBSE+COMET-Kiwi, pero sin medir) una época son ~16–18 h
   y los 3 epochs por defecto superan el límite de 24 h de `run.sbatch`. Usar
   `--n_epochs 1` y/o `--n_segments N` (recorta a las primeras N frases del train ya
   barajado, que es una muestra aleatoria válida).
3. **La recompensa IPD ignora el origen.** Nada en la recompensa obliga a que la salida sea
   una traducción de la entrada; solo el término KL (β = 0,001, muy pequeño) y el pequeño
   lr lo frenan. Con UJI se puede vigilar esto, porque la evaluación intermedia mide
   chrF++ contra la referencia valenciana sobre dev (una caída fuerte indica deriva), y
   `generate_translations.py --ipd` da el IPD final. Las frases sin ninguna palabra
   dialectalmente marcada reciben recompensa 0 (`none_value`), no se premian.
4. **Selección de checkpoint.** `adapters_best` sigue siendo el de mayor chrF++ intermedio.
   Con referencias valencianas esto ya no choca con el IPD (la advertencia pendiente en
   REPRODUCTION.md aplicaba a FLORES), aunque sigue siendo una métrica distinta de la
   recompensa. Si se quisiera seleccionar por IPD, habría que cambiar la comparación
   `chrf_val > best_chrf` en `run_training` (`src/trainer.py`); la sección 7 ya registra el
   IPD en cada evaluación, pero no lo usa para elegir.
5. **`third_party/AVLizador`.** Está vacío en el checkout actual: antes de usar
   `--reward ipd` o `--ipd` hay que hacer `git submodule update --init` y `uv sync` (modelo
   spaCy `ca_core_news_md`), como indica el README.
6. **Comillas.** El lado VA mezcla `'` y `’`. No se ha normalizado, para no alterar el texto
   de referencia; si el IPD o el chrF++ se vieran afectados, es un punto a revisar.

## 6. Verificación hecha

Entorno local sin GPU ni el submódulo, así que **no se ha ejecutado entrenamiento ni
evaluación**. Sí se ha comprobado:

- `prepare_uji.py` se ejecuta sobre el jsonl real: 10 836 pares leídos, 0 descartados,
  splits de 9 335 / 500 / 1 001 filas, 0 noticias compartidas entre splits, ningún `..`
  final restante, `src`/`ref` como `object` (str).
- Todos los `.py` modificados compilan (`py_compile`) y `resolve_data_dir` devuelve las
  rutas esperadas (y falla con `uji` + `es`).
- La lógica nueva de `main.py` mantiene el comportamiento de FLORES por construcción
  (mismo `dev.parquet`, mismo `df_eval = df_train`, mismo directorio de salida), pero
  conviene un smoke test en el clúster, p. ej.:

  ```bash
  sbatch --job-name=smoke-uji slurm/run.sbatch main.py --lang cat_Latn --dataset uji \
      --reward ipd --n_segments 20 --n_epochs 1 --eval_every 10 --seed 42
  ```

## 7. Métricas de seguimiento durante el entrenamiento

Para ver cómo evoluciona el modelo (y poder hacer gráficas), cada `--eval_every` pasos
(50 por defecto) la evaluación intermedia calcula, sobre el mismo subconjunto fijo de
`--eval_n_samples` frases del split de evaluación (`dev`), estas métricas y las guarda en
`eval_metrics.csv`:

| Columna | Métrica | Para qué |
|---|---|---|
| `chrf` | chrF++ contra la referencia | Es la única que elige `adapters_best` (como antes) |
| `bleu` | BLEU contra la referencia (sacreBLEU, tokenización 13a) | Seguimiento |
| `ipd` | IPD corpus-level: occidental / (occidental + oriental) sobre todas las hipótesis | Seguimiento de lo que optimiza la recompensa |
| `ipd_mean` | IPD medio por frase con las no marcadas a 0 | Es la misma cantidad que `--reward ipd` da en entrenamiento, comparable con `reward_raw` |
| `ipd_western`, `ipd_eastern`, `ipd_unmarked` | Nº de palabras occidentales / orientales y de frases sin ninguna marcada | Diagnóstico del IPD |
| `comet_kiwi` | COMET-Kiwi (origen + hipótesis, sin referencia) | Comprobar que el modelo no desaprende a traducir |

**Ninguna entra en el GRPO**: son solo información. `chrf` sigue siendo lo único que decide
el checkpoint.

Detalles:

- **Qué COMET.** En el código no había BLEU, y el único COMET definido es COMET-Kiwi (el de
  la recompensa híbrida, `wmt22-cometkiwi-da`). Uso ese: es sin referencia, así que sirve
  igual con cualquier dataset. No he añadido un COMET con referencia (`wmt22-comet-da`).
  BLEU lo he añadido en `metrics/bleu.py` (sacreBLEU ya era dependencia por el chrF++).
- **Filas extra.** Además de las evaluaciones cada `--eval_every` pasos, se evalúa el modelo
  **antes de entrenar** (`step=0`, `epoch=0`; con LoRA recién inicializada es el vanilla, la
  línea base de las gráficas) y **tras el último paso** si este no coincidía con una
  evaluación. Estas dos filas son solo de seguimiento y nunca eligen `adapters_best`.
  Convención de `step`: es el índice del paso de entrenamiento que acaba de ejecutarse (como
  en la columna `chrf` de `training_metrics.csv`), así que una fila `step=50` lleva 51
  actualizaciones; la fila base es la de 0 actualizaciones.
- **Techo de referencia.** `eval_reference.json` guarda el IPD y el COMET-Kiwi de las propias
  referencias valencianas sobre ese mismo subconjunto, para dibujarlos como línea horizontal.
- **Flags.** `--eval_metrics` acepta `bleu ipd comet`. Por defecto: `bleu` y `comet`, más `ipd`
  si `--lang cat_Latn`. `--eval_metrics` sin valores deja solo chrF++. Con `ipd` y otro idioma
  da error. Las métricas con referencia necesitan corpus paralelo (columna `ref`); tanto UJI
  como FLORES lo son, y `main.py` falla con un mensaje claro si falta, porque el chrF++
  también elige el checkpoint. `--eval_batch_size` fija también el tamaño de lote de COMET.
- **Se aplica también a FLORES** (el CSV se escribe siempre); con `--reward hybrid` se reutiliza
  el COMET-Kiwi ya cargado para la recompensa, y con `--reward ipd` se carga uno aparte
  (necesita `HF_TOKEN`, es un modelo gated, y unos GB de GPU más: en tarjetas de 11 GB conviene
  vigilarlo y usar `--eval_metrics bleu ipd` si hay OOM).
- **Coste.** Cada evaluación son 100 frases con beam 4 más COMET sobre otras 100: del orden de
  un minuto o menos por evaluación (sin medir en el clúster). Con `--eval_every 50` es poco
  frente a 50 pasos de entrenamiento, pero con 100 frases el IPD es ruidoso; si se quiere una
  curva más limpia, subir `--eval_n_samples` (dev tiene 500).

Cambios de código: `src/eval_metrics.py` (nuevo, calcula las métricas), `metrics/bleu.py`
(nuevo), `src/trainer.py` (`intermediate_eval` devuelve ahora un diccionario de métricas;
`run_training` acepta `evaluator` y `eval_logger`; nueva clase `EvalLogger`),
`src/reward.py` (`load_comet_kiwi()` extraído de `RewardResources`, sin cambio de comportamiento),
`src/ipd_reward.py` (`corpus_ipd` añade la clave `mean_sentence_ipd`; el resto del diccionario
no cambia, así que `generate_translations.py --ipd` sigue igual salvo esa clave nueva en el
`_summary.json`) y `main.py` (flag, construcción del evaluador, `eval_metrics.csv`,
`eval_reference.json`). `training_metrics.csv` no cambia (sigue teniendo su columna `chrf`).

**Verificación.** Probado con un actor/tokenizador/COMET/IPD simulados (sin GPU ni modelos
reales): el CSV sale con la fila base, las filas periódicas y la final, `adapters_best` se
guarda, y sin `evaluator`/`eval_logger` el flujo antiguo es idéntico. Las métricas reales
(COMET-Kiwi, IPD con spaCy) no se han ejecutado aquí; conviene el smoke test del clúster con
`--eval_every 10 --n_segments 20`.
