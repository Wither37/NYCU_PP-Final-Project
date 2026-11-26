# Memory-Scalable Kernel K-means on a Single GPU

NYCU Parallel Programming (PP-f26) final project, group 6. See `CLAUDE.md`
for the project guide and `proposal/` for the proposal.

```bash
make && make test
./bin/kkm --help

# experiments: what each script does, how to run, what to change -> experiments/README.md
python3 scripts/sweep.py experiments/smoke.json
python3 scripts/collect.py results/smoke
python3 scripts/plot.py results/smoke/runs.csv --x n --logx --logy -o plots/smoke.png
```
