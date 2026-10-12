.PHONY: db seed test eval eval-cli judge regress labels calibrate

export DATABASE_URL ?= postgresql://postgres:postgres@localhost:5433/support
RUNS ?= 3

db:        ; docker compose up -d --wait && python tools/seed.py
seed:      ; python tools/seed.py
test:      ; pytest -q                                   # layer 1: offline, seconds
eval:      ; python evals/run_evals.py --runs $(RUNS)    # layer 2: N runs per case, gate in evals/gate.yaml
judge:     ; python judge/run_judge.py                   # layer 3: judge the latest layer 2 run
regress:   ; python tools/regress.py --runs 2            # break the agent on purpose, see who notices
labels:    ; python tools/calibrate_judge.py export --n 30
calibrate: ; python tools/calibrate_judge.py score labels/to_label.csv

# The stock ADK CLI on one suite, for comparison with the runner
eval-cli:
	python tools/seed.py
	adk eval agents/support_assistant evals/orders/orders.evalset.json --config_file_path evals/orders/test_config.json
