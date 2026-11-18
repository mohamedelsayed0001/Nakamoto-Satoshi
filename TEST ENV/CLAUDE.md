# FL lab — instructions for AI assistants

Event-driven (virtual-time) simulator for asynchronous federated learning research
(device heterogeneity, stragglers, fairness, non-IID data, throughput). **Everything is a Jupyter notebook.**

## Files
- `settings.yaml`: the shared experiment conditions. All approaches run under the same settings.
- `simulator.ipynb`: everything shared (data, clients, event loop, evaluation, saving).
  Section **6. Approach API** documents everything an approach can use.
- `approaches/<name>.ipynb`: one notebook per approach (see the existing ones for the exact layout).
- `run_all.ipynb`: executes the approach notebooks (outputs are saved into each one).
- `compare.ipynb`: reads ALL results made with the current settings and plots them together.
- Results: `results/<run>_seed<k>_metrics.csv` and `_updates.csv`.

## When the user asks to try a new approach
1. Read `settings.yaml`, section 6 of `simulator.ipynb`, and the most similar notebook in `approaches/`.
2. Create `approaches/<short_snake_name>.ipynb` with the **same cell layout** as the existing ones:
   1. Markdown: title, 1–2 sentence idea, paper (if any), a numbered **How it works** list, then
      `> **Note:**` / `> **Caution:**` callouts for the key insight and any simplification you made.
   2. Markdown "## 1. Load the simulator" + code `%run ../simulator.ipynb`
   3. Markdown "## 2. The approach" + code: ONE `Approach` subclass with `name = "<short_snake_name>"`,
      `label = "<readable legend name>"` (in the style of the existing labels, e.g. "Async staleness weighted K=5"),
      and its parameters as class attributes at the top, each with a short comment.
   4. Markdown "## 3. Run it" (with a `> **Tip:**` showing a variant) + code `results = run(MyApproach())`
   5. Markdown "## 4. Quick look" + the same final-results table code as the other notebooks.
   Write the notebook with `nbformat` and keep the code short and readable, like the existing ones.
   Style: professional and plain. **No emojis** anywhere. Callouts are only `> **Note:**`,
   `> **Important:**`, `> **Caution:**`, `> **Tip:**` and `> **Correction:**`.
3. Execute it: `.venv/bin/jupyter nbconvert --to notebook --execute --inplace approaches/<name>.ipynb`
   (run from the project folder). Fix errors until it runs cleanly.
4. Re-execute `compare.ipynb` the same way and report its summary table to the user. Its plots keep the
   style of the original comparison notebook (plain matplotlib, `marker="o", markersize=3`, grid alpha 0.3,
   legend fontsize 8, dpi 150). Do not restyle them. To add a new approach to the legend order, append its
   name to `ORDER` in compare.ipynb.

## Rules
- Do not edit `settings.yaml` unless the user asks: a change makes new results incomparable with old
  ones (compare.ipynb automatically skips results made with different settings).
- Approach-specific parameters belong in the approach notebook, never in `settings.yaml`.
- Do not change `simulator.ipynb` for one approach. If an approach really needs something the API
  can't express (e.g. clients receiving a non-global model, latency depending on payload size,
  device dropout), tell the user and propose the smallest possible hook. Existing approaches must
  give identical results afterwards.
- Never delete files in `results/`. Rerunning the same approach and seed overwrites its own files.
