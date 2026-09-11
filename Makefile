.PHONY: bench repro-v031-bench help

PY := .venv/bin/python
ADV := tests/adversarial
V031 := $(ADV)/v031

help:
	@echo "Targets:"
	@echo "  bench              reproduce the classifier figures published in README.md"
	@echo "  repro-v031-bench   reproduce the historical v0.31 bench numbers"
	@echo ""
	@echo "bench needs the ml extra: pip install 'vaara[ml]'"
	@echo "The first run downloads the MiniLM embedding model, so it needs"
	@echo "network access once. Everything after that is offline."

# Reproduces the classifier figures in README.md against the bundles that ship
# in src/vaara/data/. The README tells every reader those numbers are
# reproducible by running this, so the two have to move together.
#
# THIS TARGET HAS NOW DRIFTED TWICE. It first evaluated v6 and v3 against the
# v0.35 split after both bundles left the tree, so it failed on a missing file.
# It was then pointed at v9 on the v0.39 surfaces, and stayed there when v11
# shipped on the v0.40 split, so it reproduced numbers the README no longer
# published. Both times the README kept promising reproduction and both times
# the promise was false. Whoever changes the shipped bundle changes this target
# in the same commit.
#
# The candidate is held at its SHIPPED threshold rather than recalibrated.
# Calibration answers what a model could do at a chosen FPR; a published figure
# answers what the model in the package actually does.
SHIPPED := src/vaara/data/adversarial_classifier_v11.joblib
BASELINE := src/vaara/data/adversarial_classifier_v9.joblib
SHIPPED_T := 0.8800
SPLIT := $(ADV)/v040_split.json

bench:
	@echo "[1/4] verify corpus integrity, including files on disk that no manifest line covers"
	$(PY) scripts/check_corpus_manifest.py --quiet-unlisted
	@echo "[2/4] shipped bundle against the baseline across the v0.40 surfaces"
	$(PY) scripts/eval_v039_v9.py --split $(SPLIT) \
		--baseline $(BASELINE) --candidate $(SHIPPED) \
		--candidate-threshold $(SHIPPED_T) \
		--json-out bench/v040_shipped_eval.json
	@echo "[3/4] confirmation set, generated after the operating point was fixed"
	$(PY) scripts/eval_confirmation_set.py \
		--glob 'tests/adversarial/generated/*-v041-llama33-*.jsonl' \
		--glob 'tests/adversarial/benign_generated/BT-v041-*.jsonl' \
		--bundle $(BASELINE):0.9150 --bundle $(SHIPPED):$(SHIPPED_T) \
		--json-out bench/v041_confirmation_full.json
	@echo "[4/4] cross-model holdout, attacks from a model absent from TRAIN"
	$(PY) scripts/eval_confirmation_set.py \
		--glob 'tests/adversarial/generated/*-v041-qwen25-*.jsonl' \
		--glob 'tests/adversarial/benign_generated/BT-v041-*.jsonl' \
		--bundle $(BASELINE):0.9150 --bundle $(SHIPPED):$(SHIPPED_T) \
		--json-out bench/v041_crossmodel_qwen25.json
	@echo "done. compare against the classifier bullets in README.md and bench/V11-CANDIDATE.md."

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
