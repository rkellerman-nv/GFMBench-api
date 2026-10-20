# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""E2E smoke tests: task → mock model → metrics → (optional) CSV report."""

from __future__ import annotations

import types

from gfmbench_api.benchmark_report import BenchmarkReport
from gfmbench_api.tasks.concrete.gue_promoter_all_task import GuePromoterAllTask
from tests.e2e.conftest import MockGFMModel
from tests.e2e.fixture_tasks import (
    FixtureSupervisedVariantTask,
    FixtureZeroShotIndelTask,
    FixtureZeroShotSNVTask,
)

CLASSIFICATION_METRICS = {
    "classification_accuracy",
    "classification_mcc",
    "classification_auroc",
    "classification_auprc",
}

ZERO_SHOT_COMMON_METRICS = {
    "sum_probs_llr_auroc",
    "sum_probs_llr_auprc",
    "sequence_embeddings_cosinesim_auroc",
    "sequence_embeddings_cosinesim_auprc",
    "sequence_embeddings_l2_auroc",
    "sequence_embeddings_l2_auprc",
}

ZERO_SHOT_AFFECTED_REGION_METRICS = {
    f"affected_region_k{k}_{dist}_{score}"
    for k in (0, 2)
    for dist in ("cosinesim", "l2")
    for score in ("auroc", "auprc")
} | {"affected_region_length_auroc", "affected_region_mean_tokens"}

ZERO_SHOT_SNV_METRICS = {
    "snv_variant_effect_cosinesim_auroc",
    "snv_variant_effect_cosinesim_auprc",
    "snv_variant_effect_prediction_masked_llr_auroc",
    "snv_variant_effect_prediction_masked_llr_auprc",
}


def test_e2e_supervised_single_seq(fixtures_root, smoke_task_config, mock_model):
    task = GuePromoterAllTask(
        root_data_dir_path=str(fixtures_root),
        task_config=smoke_task_config,
    )
    results = task.eval_test_set(mock_model)
    assert set(results.keys()) == CLASSIFICATION_METRICS
    assert all(score is not None for score in results.values())


def test_e2e_supervised_variant(fixtures_root, smoke_task_config, mock_model):
    task = FixtureSupervisedVariantTask(
        root_data_dir_path=str(fixtures_root),
        task_config=smoke_task_config,
    )
    results = task.eval_test_set(mock_model)
    assert set(results.keys()) == CLASSIFICATION_METRICS
    assert all(score is not None for score in results.values())


def test_e2e_zero_shot_snv(fixtures_root, smoke_task_config, mock_model):
    task = FixtureZeroShotSNVTask(
        root_data_dir_path=str(fixtures_root),
        task_config=smoke_task_config,
    )
    results = task.eval_test_set(mock_model)
    expected = ZERO_SHOT_COMMON_METRICS | ZERO_SHOT_SNV_METRICS | ZERO_SHOT_AFFECTED_REGION_METRICS
    assert set(results.keys()) == expected
    assert all(score is not None for score in results.values())


def test_e2e_zero_shot_indel(fixtures_root, smoke_task_config, mock_model):
    task = FixtureZeroShotIndelTask(
        root_data_dir_path=str(fixtures_root),
        task_config=smoke_task_config,
    )
    results = task.eval_test_set(mock_model)
    assert set(results.keys()) == ZERO_SHOT_COMMON_METRICS | ZERO_SHOT_AFFECTED_REGION_METRICS
    assert ZERO_SHOT_SNV_METRICS.isdisjoint(results.keys())
    assert all(score is not None for score in results.values())


def test_e2e_report_pipeline(fixtures_root, smoke_task_config, mock_model, tmp_path):
    task = GuePromoterAllTask(
        root_data_dir_path=str(fixtures_root),
        task_config=smoke_task_config,
    )
    results = task.eval_test_set(mock_model)
    csv_path = tmp_path / "smoke_results.csv"
    report = BenchmarkReport(str(csv_path))
    report.add_scores(task.get_task_name(), "mock_model", results)
    report.save_csv()

    reloaded = BenchmarkReport(str(csv_path))
    df = reloaded.get_dataframe()
    assert len(df) == len(CLASSIFICATION_METRICS)
    assert "mock_model" in df.columns
    for metric in CLASSIFICATION_METRICS:
        row = df[(df["task"] == task.get_task_name()) & (df["metric"] == metric)]
        assert len(row) == 1
        assert row["mock_model"].iloc[0] is not None


def test_e2e_missing_model_method(fixtures_root, smoke_task_config):
    model = MockGFMModel()
    model.infer_sequence_to_labels_probs = types.MethodType(
        lambda self, sequences, conditional_input=None: None,
        model,
    )
    task = GuePromoterAllTask(
        root_data_dir_path=str(fixtures_root),
        task_config=smoke_task_config,
    )
    results = task.eval_test_set(model)
    assert set(results.keys()) == CLASSIFICATION_METRICS
    assert all(score is None for score in results.values())
