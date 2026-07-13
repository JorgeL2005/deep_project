# Puesta en marcha en un dispositivo nuevo

## 1. Python
Usa **Python 3.13** (probado en Windows 11). Verifica: `python --version`.

## 2. Dependencias pip

```bash
python -m pip install -r requirements.txt          # ejecutar / evaluar el agente
python -m pip install -r requirements-train.txt    # SOLO si vas a entrenar (ver el archivo: torch va aparte)
```

## 3. Overcooked-AI (IMPORTANTE: no viene de pip)

Este proyecto usa un **fork parcheado** de Overcooked-AI que vive en una carpeta
hermana: `../Overcooked-AI` (contiene el paquete `overcooked_ai_py`). No se
instala con pip; se encuentra vía `PYTHONPATH`.

Copia la carpeta `Overcooked-AI` junto a `proyecto_deep` de modo que quede:

```
   <algun_dir>/
   ├── proyecto_deep/        <- este repo
   └── Overcooked-AI/        <- el fork con overcooked_ai_py
```

Parches ya aplicados en ese fork (necesarios para NumPy 2.x; si clonas el
Overcooked-AI oficial en vez de copiar este, hay que re-aplicarlos):
- `np.Inf -> np.inf` en `planning/planners.py`, `agents/agent.py`,
  `mdp/overcooked_env.py`, `mdp/layout_generator.py` (y `np.int -> int`).
- `mdp/overcooked_mdp.py`: `OvercookedState.from_dict` acepta `objects` como
  lista o dict (arregla el layout `tutorial_1`).
- `mdp/overcooked_env.py`: acumuladores de recompensa en `float` en lugar de
  `int64` (evita crash con recetas de valor no entero).

## 4. Configurar PYTHONPATH y ejecutar

Windows (PowerShell):
```powershell
cd <ruta>\proyecto_deep
$env:PYTHONPATH = "<ruta>\Overcooked-AI"
python -m src.run_game --config configs/evaluate_student_bc.yaml
```

Linux/macOS (bash):
```bash
cd <ruta>/proyecto_deep
export PYTHONPATH=<ruta>/Overcooked-AI
python -m src.run_game --config configs/evaluate_student_bc.yaml
```

## 5. Agente entregable

El agente es `policies/student_agent_bc.py` + un archivo de pesos `.npz`
(recomendado: `policies/bc_weights_v5.npz`). En inferencia SOLO usa `numpy`.
Selecciona los pesos en el YAML del config:

```yaml
policies:
  agent_0:
    config:
      weights: bc_weights_v5.npz
```

Sin la clave `weights`, el agente carga `bc_weights.npz` por defecto.
