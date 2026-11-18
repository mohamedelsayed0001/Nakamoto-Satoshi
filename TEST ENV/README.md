# FL lab

Everything is a notebook.

```
fl_lab/
├── settings.yaml            shared conditions for ALL approaches: CIFAR-10 baseline (data, clients, devices, budget, seeds)
├── settings_mnist_reference.yaml   the original MNIST setup (reproduces the old notebooks exactly)
├── tuning.ipynb             tunes lr, batch size, local epochs, clients for the baseline; writes them into settings.yaml
├── simulator.ipynb          the shared engine; every approach runs it with  %run ../simulator.ipynb
├── approaches/              one notebook per approach: open it and Run All
│   ├── sync_fedavg.ipynb          notebook 01
│   ├── fedasync.ipynb             notebook 02
│   ├── fedasync_staleness.ipynb   notebook 03, fixed (staleness now really scales the update)
│   ├── buffered_fedavg.ipynb      notebook 04
│   └── fedbuff.ipynb              notebook 05
├── run_all.ipynb            runs every approach notebook (saves each with its outputs)
├── compare.ipynb            reads ALL results and plots them together, with notes on each plot
├── CLAUDE.md                instructions an AI assistant follows to add a new approach
└── results/                 <approach>_seed<k>_metrics.csv / _updates.csv
```

How they connect:

```
settings.yaml ──► simulator.ipynb ◄── %run ── approaches/<name>.ipynb ── run(...) ──► results/*.csv
                                                                                         │
                                                         compare.ipynb ◄── reads all ────┘
```

## Use
1. **One approach:** open `approaches/<name>.ipynb` and run all cells.
2. **All approaches:** open `run_all.ipynb` and run it.
3. **Compare:** open `compare.ipynb` and run it. Old and new results appear together.

Select the `.venv` Python as the notebook kernel (VS Code: "Select Kernel" → `fl_lab/.venv`).

## Adding an approach
Ask the AI, e.g. *"add FedAT with 3 tiers"*. It follows `CLAUDE.md`: it reads `settings.yaml` and
the **Approach API** section of `simulator.ipynb`, creates `approaches/fedat.ipynb` in the same
style, runs it, and refreshes `compare.ipynb`.

By hand: duplicate the closest approach notebook, rename the class and `name`, and change `on_update`.

## Comparing fairly
`compare.ipynb` only compares results made with the **current** `settings.yaml`, and lists any it
skipped. Use 3+ seeds (`seeds: [42, 43, 44]` in settings.yaml); one seed can't separate approaches.
