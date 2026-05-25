.PHONY: bench repro-v031-bench help

PY := .venv/bin/python
ADV := tests/adversarial
V031 := $(ADV)/v031

help:
	@echo "Targets:"
	@echo "  bench              reproduce the current bench doc numbers (v0.34)"
	@echo "  repro-v031-bench   reproduce the historical v0.31 bench numbers"

# Current bench reproduction. Always points at the latest released
# methodology (currently bench/vaara-bench-v0.34.md). Anyone cloning
# at a tagged commit can run this and get the same SHAs and numbers.
bench:
	@echo "[1/4] verify corpus integrity (includes v0.34 additions)"
	cd $(ADV) && sha256sum -c MANIFEST.sha256 > /dev/null
	@echo "[2/4] evaluate production v3 bundle on v031_split TEST"
	$(PY) scripts/eval_v032.py \
		--bundle src/vaara/data/adversarial_classifier_v3.joblib \
		--target-fpr 0.05 \
		--json-out bench/v033_eval_final.json
	@echo "[3/4] cross-eval v3 on v034_split TEST"
	$(PY) scripts/eval_v032.py \
		--bundle src/vaara/data/adversarial_classifier_v3.joblib \
		--split-manifest $(ADV)/v034_split.json \
		--target-fpr 0.05 \
		--json-out bench/v034_eval_v3_cross.json
	@echo "[4/4] cross-eval v5 (A/B) on v034_split TEST"
	$(PY) scripts/eval_v032.py \
		--bundle src/vaara/data/adversarial_classifier_v5.joblib \
		--split-manifest $(ADV)/v034_split.json \
		--target-fpr 0.05 \
		--json-out bench/v034_eval_v5.json
	@echo "done. compare SHAs to bench/vaara-bench-v0.34.md."

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
