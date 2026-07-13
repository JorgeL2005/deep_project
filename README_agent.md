# Agente para Overcooked-AI: red entrenada por destilación (BC + DAgger)

**Entregable: [policies/student_agent_bc.py](policies/student_agent_bc.py) + `policies/bc_weights.npz`**
— una CNN entrenada; en evaluación solo corre la red (inferencia numpy pura,
~6.5 ms/acción, sin torch ni heurísticas). Config:
[configs/evaluate_student_bc.yaml](configs/evaluate_student_bc.yaml).

## Método de entrenamiento

Destilación de política (BC + DAgger) sobre un experto planificador:

1. **Experto** ([policies/student_agent.py](policies/student_agent.py)): agente
   de planificación ad-hoc (solo se usa offline para generar etiquetas).
2. **Datos**: ~800k pares (observación, acción experta) en 57 layouts
   (14 oficiales + 43 custom del dataset), self-play y contra 3 baselines,
   ambos roles, con ruido de ejecución para cobertura de estados.
3. **Observación**: encoding lossless de Overcooked (26 canales/celda) con
   **frame stacking ×3** (78 canales) — necesario porque el experto tiene
   estado interno y una sola observación no es Markoviana — padded a 16×24.
4. **Red**: Conv(78→96)→Conv(96→96)→Conv(96→64)→FC(256)→FC(6), ~6.7M
   parámetros, CE con pesos por clase (inversa^0.3) y label smoothing.
5. **DAgger ×3**: la red juega, el experto re-etiqueta cada estado visitado y
   se reentrena sobre el agregado (corrige el distribution shift del BC puro).
6. **Selección de checkpoint por gameplay real** (no val_acc): cada
   checkpoint juega 20 partidas de prueba; se exporta el mejor (iter3,
   probe 206.8; progresión 190→175→190→207).
7. **Export**: pesos a `bc_weights.npz`; paridad torch↔numpy verificada
   (error máx 4.8e-06).

Reproducir: `C:\...\vt\Scripts\python.exe -m training.run_pipeline`
(ver `training/`; logs en `training/logs/pipeline.log`).

## Resultados del agente ENTRENADO (retorno sparse medio, horizonte 400, seeds 0-2, ambos roles)

| Layout | self-play | greedy_full_task | random_motion |
|---|---|---|---|
| cramped_room | 100 | 50 | 153 |
| coordination_ring | 160 | 210 | 70 |
| counter_circuit | 408 | 123 | 133 |
| forced_coordination | 60 | 100 | 0* |
| asymmetric_advantages | 180 | 190 | 140 |
| large_room | 180 | 140 | 127 |
| simple_o | 160 | 220 | 170 |
| simple_tomato | 80 | 0* | 157 |
| small_corridor | 20 | 20 | 13 |
| soup_coordination | 494 | 216 | 386 |
| tutorial_0 | 504 | 473 | 315 |
| tutorial_1 | 476 | 85 | 45 |
| tutorial_2 | 468 | 475 | 230 |
| tutorial_3 | 468 | 475 | 230 |

\* Casos demostrablemente imposibles por el compañero: en forced_coordination
`random_motion` nunca interactúa (nadie puede cocinar); en simple_tomato
`greedy_full_task` envenena la única olla con cebollas en ciclo infinito (el
experto también saca 0 ahí).

## v3: entrenamiento continuado anti-congelamiento (`bc_weights_v3.npz`)

`training/run_pipeline_v3.py` continuó el entrenamiento desde el mejor
checkpoint v2 con: sobremuestreo ×4 de estados donde el alumno lleva ≥4 pasos
quieto y el experto indica actuar, ×3 episodios en los mapas débiles, fine-
tuning a lr 3e-4 y selección de checkpoint penalizando la racha de inactividad
(el modelo v2 compitió como baseline en la selección).

Resultado (probe de gameplay): retorno 198.2 → **212.6** y racha máxima de
inactividad media 69 → **31.4** pasos. En la matriz completa: promedio
207.2 → **228.9**; mejoras grandes en soup_coordination vs greedy (+165),
forced_coordination self (+120), simple_tomato self (+120), cramped_room self
(+100); regresiones en counter_circuit self (−272) y large_room self (−80).
Comparación completa: `training/logs/final_eval_v2.json` vs
`training/logs/final_eval.json` (v3).

Los pesos v2 originales están intactos en `bc_weights.npz` (backup:
`bc_weights_v2_backup.npz`). Para usar v3, en el YAML del agente:

```yaml
config:
  weights: bc_weights_v3.npz
```

## v4 (ronda corta) y evaluación en mapas custom

`training/run_pipeline_v4.py` intentó recuperar counter_circuit/large_room con
2 iteraciones enfocadas; ninguna superó al baseline v3 en el probe, así que la
guardia exportó los pesos de v3 como `bc_weights_v4.npz` (v3 == v4).

Evaluación en los 43 layouts custom del dataset (self-play + greedy, ambos
roles, seeds 0-2), resultados en `training/logs/custom_eval.json`:

- Promedio: self-play 176.5, con greedy 137.3.
- Mejores: dual_pots 307/303, dos_ollas 307/247, isla_central 320/227,
  m_room 313/229, duelo_1v1 353 self.
- El agente anota >0 en TODOS los mapas custom jugables (37/43). Los 6 con
  cero (guillermo_custom_03/04/05, salinas_custom_02, onion_hard_1,
  tomato_hard_1) son **imposibles por construcción**: tienen la olla, la
  entrega o los dispensers sellados por counters sin casilla caminable
  adyacente (verificado; el experto planificador también saca 0).

## v5: especialización para los mapas de competencia (`bc_weights_v5.npz`) ⭐ RECOMENDADO

El profesor anunció que la competencia se juega en asymmetric_advantages,
coordination_ring y counter_circuit. `training/run_pipeline_v5.py` continuó
el fine-tuning desde v3 con sobremuestreo ×6 de esos mapas (los 57 layouts
siguen en la mezcla), boost de estados de atasco, y selección de checkpoint
por score compuesto (competencia con penalización de idle + 0.4×generalidad),
con v3 como baseline de guardia. Ganó iter11.

Mapas de competencia (promedio sobre self/greedy/random_motion/stay, ambos
roles, seeds 0-2): **v3 181.4 → v5 206.7 (+14%)**. Destacado: counter_circuit
vs random_motion +128 (189→317), vs greedy +84 (168→253); coordination_ring
self +47, vs greedy +40; asymmetric_advantages self +13 con idle igual de
bajo. Generalidad intacta: matriz oficial v5 promedio 244.5, sin ceros nuevos
(solo los 2 pareos demostrablemente imposibles). Detalle:
`training/logs/final_eval_v5.json`.

## Agente alternativo probado: PPO self-play desde BC (`training_rl/`)

Segundo paradigma evaluado: fine-tuning con RL (PPO) partiendo de la red BC
v3, con población de compañeros estilo FCP (self-play 60%, BC congelada 20%,
greedy 10%, random 10%), recompensa sparse + shaped, GAE, ancla KL a la
política BC y warmup del crítico. Tres intentos documentados en
`training_rl/logs/`:

1. Sin estabilización → colapso (213 → 67 en 30 iters).
2. Warmup 15 + KL 0.05 → deriva lenta (213 → 184) y crash por bug NumPy 2 del
   fork (acumulador int64; parcheado en overcooked_env.py).
3. Warmup 25 + KL 0.15 + lr 2e-5 → estable; pico en iter60 con retorno 227.5
   (supera el 212.6 de BC) pero score 148.0 vs 149.9 por idle; luego meseta
   oscilante sin superar al baseline en 400 iteraciones (~1M pasos).

Conclusión: con este presupuesto de cómputo, **la destilación BC+DAgger
supera al fine-tuning PPO**; la guardia de checkpoint exportó los pesos v3
como `policies/rl_weights.npz` (idénticos a v3). La comparación completa de
curvas está en `training_rl/logs/rl_attempt*.log` y `rl.log`.

---

# Apéndice: el experto planificador (usado solo para entrenar)

## Por qué destilar un planificador y no RL por layout (estado del arte)

Los métodos aprendidos de zero-shot coordination (PPO_BC/Human-Aware RL, FCP,
MEP, TrajeDi, COLE, E3T) entrenan **una política por layout**. Los benchmarks
recientes muestran que no generalizan a layouts nunca vistos:

- *The Overcooked Generalisation Challenge* (TMLR 2025): incluso los mejores
  métodos de curriculum (DCD/UED) fallan al cooperar con compañeros nuevos en
  layouts nuevos; solo PAIRED+SoftMoE logra generalización parcial.
- *OvercookedV2* (ICLR 2025): gran parte del "éxito" de ZSC en Overcooked
  clásico es cobertura de estados, no coordinación; los métodos actuales no se
  adaptan en tiempo de evaluación.
- Los agentes LLM (ProAgent) sí se adaptan, pero no cumplen el límite de
  100 ms por acción.

Para esta competencia (layouts custom desconocidos, compañero desconocido por
ronda, cambio de rol, 100 ms/acción y score dominado por #sopas), la solución
más fuerte disponible es un **agente de ad-hoc teamwork basado en
planificación**, que generaliza por construcción y se adapta al compañero
en línea. Es también lo que hace competitivo a un agente en OGC: el problema
duro no es "aprender a cocinar", es *coordinarse con cualquiera en cualquier
cocina*.

## Diseño

Jerárquico: análisis del mundo → selección de subtarea → navegación BFS.

1. **Generalización estructural**: lee el layout del estado crudo (ollas,
   dispensers, counters, zonas de entrega) y planifica con BFS. Funciona en
   cualquier grid, con recetas arbitrarias (cebolla/tomate mixtas), tiempos y
   valores de cocción custom (`onion_time`, `recipe_times`, bonus orders) y
   cualquier número de ollas. Soporta ambos roles (índice 0/1).
2. **Razonamiento de recetas**: elige la orden más rápida de completar
   (prioriza #sopas; el valor solo desempata), solo agrega ingredientes que
   mantienen el contenido de la olla como sub-multiconjunto de alguna orden, e
   inicia la cocción (INTERACT) cuando el contenido coincide exactamente.
3. **Adaptación al compañero**:
   - No duplica trabajo: descuenta platos/ingredientes que el compañero lleva.
   - *Prefetch* predictivo del plato cuando al pot le falta exactamente el
     ingrediente que trae el compañero.
   - **Ollas envenenadas**: si el compañero mete un ingrediente que no cabe en
     ninguna orden, cocina la olla de inmediato para liberarla y descarta la
     sopa sin valor en un counter (no la entrega).
4. **Layouts desconectados (forced_coordination)**: detecta regiones
   inalcanzables y hace *handoff* por counters compartidos: abastece
   ingredientes/platos al lado del compañero o los recoge, eligiendo el counter
   que minimiza el costo conjunto.
5. **Protocolo anti-deadlock en corredores** (crítico en small_corridor,
   coordination_ring):
   - Prioridad de paso determinista: agente cargado > vacío; empate → índice 0.
   - El que cede se **compromete** a retirarse a un "bolsillo" verificado por
     BFS (una casilla desde la cual el compañero aún puede pasar).
   - Si el compañero no se mueve tras cederle el paso, se lo trata como
     **pared** y se replanifica alrededor (se limpia si vuelve a moverse).
   - Detección de atasco con sidestep estocástico y regla de cortesía si
     bloqueo a un compañero estancado.
6. **Seguridad**: nunca lanza excepciones al runner (fallback `stay`);
   latencia medida 0.15 ms media / 0.65 ms máx por acción (límite: 100 ms →
   riesgo de penalización por timeout: cero).

## Requisito de configuración (IMPORTANTE)

El agente necesita la observación de estado crudo. En el YAML:

```yaml
observation:
  type: state
  include_agent_index: true
```

Config de ejemplo: [configs/evaluate_student.yaml](configs/evaluate_student.yaml).

```
python -m src.run_game --config configs/evaluate_student.yaml
```

## Resultados (retorno sparse medio, horizonte 400, seeds 0-2, ambos roles)

| Layout | self-play | greedy_full_task | random |
|---|---|---|---|
| cramped_room | 180 | 210 | 90 |
| coordination_ring | 180 | 240 | 70 |
| counter_circuit | 612 | 280 | 0* |
| forced_coordination | 240 | 100* | 0* |
| asymmetric_advantages | 260 | 190 | 170 |
| large_room | 180 | 150 | 70 |
| simple_o | 210 | 290 | 270 |
| simple_tomato | 190 | 0* | 180 |
| small_corridor | 80 | 60 | 0* |
| soup_coordination | 546 | 400 | 494 |
| tutorial_0 | 630 | 630 | 315 |
| tutorial_1 | 544 | 119* | 272 |
| tutorial_2 | 494 | 475 | 247 |
| tutorial_3 | 494 | 475 | 247 |

(20 puntos ≈ 1 sopa estándar; en tutoriales/counter_circuit las recetas valen
distinto.) Los casos * son límites del compañero, no del agente:
`greedy_full_task` está cableado a cebollas y envenena las ollas en layouts de
tomate (simple_tomato, tutorial_1); en forced_coordination se congela con la
cebolla en la mano y nadie puede compensarlo; `random`/`stay` a veces quedan
parados sobre la única casilla de acceso al dispenser de platos.

También probado en layouts custom del dataset (recetas mixtas, dos ollas,
islas): 200–360 en self-play.

## Notas de entorno (Windows, este repo)

- Python 3.13 (Store) con `gym`, `pyyaml`, `imageio`, `ipywidgets` instalados
  (`py -3.13 -m pip install --user ...`) y `PYTHONPATH` apuntando al repo
  hermano `..\Overcooked-AI` que contiene `overcooked_ai_py`.
- Se parchearon en ese repo: `np.Inf → np.inf` (NumPy 2.x) y
  `OvercookedState.from_dict` (aceptar `objects` como lista — necesario para
  `tutorial_1`).
- El layout custom `Attention_t_Layouts_maze_kitchen.layout` es inválido para
  el framework (declara `recipe_values` y `onion_value` a la vez).
