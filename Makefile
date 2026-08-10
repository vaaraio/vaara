.PHONY: bench repro-v031-bench help

PY := .venv/bin/python
ADV := tests/adversarial
V031 := $(ADV)/v031

help:
	@echo "Targets:"
	@echo "  bench              reproduce the current bench doc numbers (v0.39)"
	@echo "  repro-v031-bench   reproduce the historical v0.31 bench numbers"
	@echo ""
	@echo "bench needs the ml extra: pip install 'vaara[ml]'"
	@echo "The first run downloads the MiniLM embedding model, so it needs"
	@echo "network access once. Everything after that is offline."

# Current bench reproduction, against bench/vaara-bench-v0.39.md and the v8
# and v9 bundles that actually ship in src/vaara/data/.
#
# This target used to evaluate adversarial_classifier_v6 and _v3 against the
# v0.35 split. Neither bundle is in the tree any more, so the target failed on
# a missing file, while the README told a reader every published figure was
# reproducible by running it. Pointing it at the script that produced the
# published artifact keeps the two in step.
bench:
	@echo "[1/2] verify corpus integrity"
	cd $(ADV) && sha256sum -c MANIFEST.sha256 > /dev/null
	@echo "[2/2] calibrate and evaluate v9 across the four v0.39 surfaces"
	$(PY) scripts/eval_v039_v9.py --json-out bench/v039_v9_eval.json
	@echo "done. compare against bench/vaara-bench-v0.39.md."

# End-to-end reproduction of bench/vaara-bench-v0.31.md. Anyone cloning
# the repo at a tagged commit can run this and get the same SHAs and
# the same numbers. Fails fast if the corpus integrity check fails.
repro-v031-bench:
	@echo "[1/8] verify corpus integrity"
	cd $(ADV) && sha256sum -c MANIFEST.sha256 > /dev/null
	@echo "[2/8] build deterministic 70/15/15 split"
	$(PY) scripts/build_train_val_test_split.py
	@echo "[3/8] train classifier bundle on TRAIN fold"
	$(PY) scripts/save_classifier_bundle.py \
		--version v0.31 --threshold 0.90 \
		--split-manifest $(ADV)/v031_split.json \
		--bundle-out src/vaara/data/adversarial_classifier_v2.joblib
	@echo "[4/8] score full corpus through Pipeline.intercept + classifier"
	$(PY) scripts/eval_pipeline_attribution.py --fold all
	@echo "[5/8] three-way variants on VAL"
	$(PY) scripts/three_way_variants.py --fold val --classifier-threshold 0.90 \
		--out $(V031)/three_way_variants_val_v031.json
	@echo "[6/8] threshold sweep on VAL"
	$(PY) scripts/threshold_sweep_val.py --fold val
	@echo "[7/8] held-out TEST eval at picked threshold"
	$(PY) scripts/three_way_variants.py --fold test --classifier-threshold 0.90 \
		--out $(V031)/test_final_eval_v031.json
	@echo "[8/8] Wilson 95% intervals on TEST headline"
	$(PY) scripts/wilson_intervals.py
	@echo "done. compare SHAs printed above to bench/vaara-bench-v0.31.md."
