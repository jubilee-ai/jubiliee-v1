/**
 * Demo mock data for Model Risk / registry profiles.
 * Toggle off to use live `/api/model-risk/*` when implemented.
 */
import type {
  ModelRiskDashboard,
  ModelRiskInventoryItem,
  ModelRiskPortalRole,
  ModelRiskProfileDetail,
} from "@/lib/api"

export const USE_MODEL_RISK_MOCK = true

/** Matches registry UI: Development in progress, monitoring healthy, next review aligned. */
const NEXT_REVIEW_ISO = "2026-06-29"

const LIFECYCLE_DEVELOPMENT_IN_FLIGHT: ModelRiskProfileDetail["lifecycle_stages"] = [
  { id: "identification", label: "Identification", status: "complete", entered_at: "2026-01-10T10:00:00Z", notes: "Captured in linked experiment / inventory." },
  { id: "development", label: "Development", status: "current", entered_at: "2026-02-01T09:00:00Z", notes: "Training, testing, and documentation in progress." },
  { id: "validation", label: "Validation", status: "pending", notes: "Independent review not yet started." },
  { id: "approval", label: "Approval", status: "pending", notes: "—" },
  { id: "implementation", label: "Implementation", status: "pending", notes: "—" },
  { id: "monitoring", label: "Monitoring", status: "pending", notes: "Ongoing performance tracking after deployment." },
  { id: "periodic_review", label: "Periodic review", status: "pending", notes: "Scheduled revalidation." },
  { id: "retirement", label: "Retirement", status: "pending", notes: "—" },
]

const MONITORING_HEALTHY_DEV: ModelRiskProfileDetail["monitoring"] = {
  status: "healthy",
  as_of: "2026-04-12T07:00:00Z",
  psi_csi: [
    { feature: "top_feature_1", psi: 0.05, csi: 0.1 },
    { feature: "top_feature_2", psi: 0.07, csi: 0.12 },
    { feature: "top_feature_3", psi: 0.04, csi: 0.08 },
  ],
  backtest: [
    { period: "Latest holdout", metric: "Policy check", value: 1, benchmark: 1, pass: true },
  ],
  narrative: "No drift breach. Continue development / pre-production checks.",
}

const REVIEWS_STANDARD: ModelRiskProfileDetail["reviews"] = [
  {
    id: "r-next",
    review_type: "Scheduled periodic review",
    scheduled_for: NEXT_REVIEW_ISO,
    completed_at: null,
    outcome: null,
    owner: "MRM",
  },
]

/** Per trained-model registry row: copy aligned with experiment goals in the lab. */
type TrainedRegistryEntry = {
  model_type: string
  intended_use: string
  /** Shorter line for the model list card (optional). */
  intended_use_summary?: string
  usage_guidance?: string
  performance_kpis?: ModelRiskProfileDetail["performance_kpis"]
}

const TRAINED_REGISTRY: Record<string, TrainedRegistryEntry> = {
  pit_nextlap_rf_v1: {
    model_type: "sklearn_RandomForestClassifier",
    intended_use:
      "We want to build a model that can predict when a Formula 1 pit stop will occur on the next lap, using telemetry and race context from the linked experiment.",
    intended_use_summary:
      "We want to build a model that can predict when a Formula 1 pit stop will occur on the next lap (telemetry / race context).",
    usage_guidance:
      "Run on the same feature schema as training (pit history, tire age, gap to car ahead). Do not use outside single-car telemetry domains covered in the report.",
    performance_kpis: [
      { label: "Val ROC AUC", value: 100, unit: "%", window: "Validation", trend: "flat" },
      { label: "Val Accuracy", value: 98.2, unit: "%", window: "Validation", trend: "flat" },
    ],
  },
  rf_v5: {
    model_type: "sklearn_RandomForestRegressor",
    intended_use:
      "Predict Quantity with a random forest regressor on engineered tabular features from the linked experiment (inventory row: 16 features, ~10.3k samples).",
    usage_guidance: "Validate input ranges against training; flag out-of-range rows.",
  },
  hgb_v1: {
    model_type: "sklearn_HistGradientBoostingRegressor",
    intended_use:
      "Predict charges (HistGradientBoosting, v2) from policyholder and risk attributes — aligns with insurance pricing experiments.",
    performance_kpis: [
      { label: "Val ROC AUC", value: 99.5, unit: "%", window: "Validation", trend: "flat" },
      { label: "Val Accuracy", value: 97.9, unit: "%", window: "Validation", trend: "flat" },
    ],
  },
  kmeans_v2_k3: {
    model_type: "sklearn_unsupervised_KMeans",
    intended_use: "Unsupervised clustering (k=3) for segmentation; labels are cluster ids, not a supervised target.",
    usage_guidance: "Interpret clusters with business descriptors before using in decisions.",
  },
  kmeans_v9_holdout_k4: {
    model_type: "sklearn_unsupervised_KMeans",
    intended_use: "Volatility clustering: partition observations into volatility regimes (k=4) for exploratory monitoring.",
    intended_use_summary: "Volatility Clustering — KMeans holdout segmentation.",
    usage_guidance: "Use cluster assignments as inputs to downstream models or dashboards, not as sole decision.",
  },
  rf_v2: {
    model_type: "sklearn_RandomForestRegressor",
    intended_use: "Predict charges with random forest regression on the pricing dataset (9 features, ~17.4k samples in registry).",
  },
  vol_cluster_kmeans_holdout_v1: {
    model_type: "sklearn_unsupervised_KMeans",
    intended_use: "Volatility Clustering — assign each row to a volatility regime for monitoring and downstream features.",
    intended_use_summary: "Volatility Clustering (holdout).",
  },
  kmeans_k5_v1: {
    model_type: "sklearn_unsupervised_KMeans",
    intended_use: "Large-sample clustering (k=5) for customer or transaction segmentation at scale.",
  },
  rf_v3: {
    model_type: "sklearn_RandomForestClassifier",
    intended_use:
      "Classify PitNextLap (v3): lap-level features for next-lap pit window prediction — same problem family as pit_nextlap_rf_v1 with an iterated RF architecture.",
    performance_kpis: [
      { label: "Val ROC AUC", value: 99.9, unit: "%", window: "Validation", trend: "flat" },
      { label: "Val Accuracy", value: 98.2, unit: "%", window: "Validation", trend: "flat" },
    ],
  },
  hgb_v4: {
    model_type: "sklearn_HistGradientBoostingRegressor",
    intended_use: "Train an insurance pricing model: predict charges from policy and risk inputs.",
    intended_use_summary: "train an insurance pricing model",
  },
  rf_probe2: {
    model_type: "sklearn_RandomForestClassifier",
    intended_use: "Predict binary Default from loan application features (probe / wider feature set).",
    performance_kpis: [
      { label: "Val ROC AUC", value: 74.6, unit: "%", window: "Validation", trend: "flat" },
      { label: "Val Accuracy", value: 88.4, unit: "%", window: "Validation", trend: "flat" },
    ],
  },
  loan_default_demo: {
    model_type: "sklearn_logistic_regression",
    intended_use: "Demonstrate default prediction on the loan default demo dataset (small-n sanity check).",
    performance_kpis: [
      { label: "ROC AUC", value: 1, window: "Train/val (demo)", trend: "flat" },
      { label: "Accuracy", value: 100, unit: "%", window: "Train/val (demo)", trend: "flat" },
    ],
  },
  nn_baseline_v1_fixbool: {
    model_type: "pytorch_nn",
    intended_use: "Neural network baseline with boolean feature fixes; use for comparison against sklearn baselines.",
  },
  ridge_v1: {
    model_type: "sklearn_Ridge",
    intended_use: "Linear regression baseline for charges (ridge regularization).",
    performance_kpis: [
      { label: "Val R²", value: 78.3, unit: "%", window: "Validation", trend: "flat" },
      { label: "Val RMSE", value: 5536.9641, window: "Validation", trend: "flat" },
    ],
  },
  gbc_v1: {
    model_type: "sklearn_GradientBoostingClassifier",
    intended_use: "Predict Default with gradient boosting on the engineered feature set.",
    performance_kpis: [
      { label: "Val ROC AUC", value: 73.1, unit: "%", window: "Validation", trend: "flat" },
      { label: "Val Accuracy", value: 89.4, unit: "%", window: "Validation", trend: "flat" },
    ],
  },
  kmeans_k4_v1: {
    model_type: "sklearn_unsupervised_KMeans",
    intended_use: "Four-cluster segmentation on large tabular data (255k+ rows) for exploratory analytics.",
  },
  skill_val_metrics_model: {
    model_type: "sklearn_LogisticRegression",
    intended_use: "Minimal validation harness: predict target with two features (skill / metrics check).",
    performance_kpis: [
      { label: "Val ROC AUC", value: 87.5, unit: "%", window: "Validation", trend: "flat" },
      { label: "Val Accuracy", value: 75, unit: "%", window: "Validation", trend: "flat" },
    ],
  },
  gbr_v3: {
    model_type: "sklearn_GradientBoostingRegressor",
    intended_use: "Train a pricing model: predict charges with gradient boosting regression.",
    intended_use_summary: "train a pricing model",
  },
  rf_v1: {
    model_type: "sklearn_RandomForestRegressor",
    intended_use: "Predict charges via random forest regression (pricing line).",
    performance_kpis: [
      { label: "Val ROC AUC", value: 99.9, unit: "%", window: "Validation", trend: "flat" },
      { label: "Val Accuracy", value: 98.2, unit: "%", window: "Validation", trend: "flat" },
    ],
  },
  ridge_baseline_v1: {
    model_type: "sklearn_RidgeClassifier",
    intended_use: "Linear baseline classifier for Default on the wide loan feature set.",
  },
  logreg_l2_balanced_v1: {
    model_type: "sklearn_LogisticRegression",
    intended_use: "L2-balanced logistic regression for Default; class-weighted for imbalance.",
  },
  nn_wider_v2: {
    model_type: "pytorch_nn",
    intended_use: "Wider neural network architecture experiment vs nn_baseline; medium-risk tier until validated.",
  },
}

const LIFECYCLE_FULL: ModelRiskProfileDetail["lifecycle_stages"] = [
  { id: "identification", label: "Identification", status: "complete", entered_at: "2025-06-01T10:00:00Z", notes: "Business case: CECL reserve estimation for commercial portfolio." },
  { id: "development", label: "Development", status: "complete", entered_at: "2025-07-15T14:00:00Z", notes: "Trained on 24mo history; hold-out validation completed." },
  { id: "validation", label: "Validation", status: "complete", entered_at: "2025-08-20T09:00:00Z", notes: "Independent review: conceptual soundness + data integrity." },
  { id: "approval", label: "Approval", status: "complete", entered_at: "2025-09-01T16:00:00Z", notes: "MRM committee sign-off (small-bank path)." },
  { id: "implementation", label: "Implementation", status: "complete", entered_at: "2025-09-15T11:00:00Z", notes: "Deployed to loss forecasting workflow; UAT passed." },
  { id: "monitoring", label: "Monitoring", status: "current", entered_at: "2025-10-01T08:00:00Z", notes: "Monthly PSI/CSI and quarterly back-test." },
  { id: "periodic_review", label: "Periodic review", status: "pending", notes: "Next full revalidation scheduled." },
  { id: "retirement", label: "Retirement", status: "pending", notes: "Not applicable until replacement model selected." },
]

const LIFECYCLE_CREDIT: ModelRiskProfileDetail["lifecycle_stages"] = [
  { id: "identification", label: "Identification", status: "complete", entered_at: "2025-04-10T10:00:00Z", notes: "Commercial credit risk rating for C&I loans." },
  { id: "development", label: "Development", status: "complete", entered_at: "2025-05-22T14:00:00Z", notes: "Feature set aligned with policy 4.2." },
  { id: "validation", label: "Validation", status: "complete", entered_at: "2025-07-01T09:00:00Z", notes: "Third-party model validation memo on file." },
  { id: "approval", label: "Approval", status: "complete", entered_at: "2025-07-20T16:00:00Z", notes: "High-risk model: dual approval + committee." },
  { id: "implementation", label: "Implementation", status: "complete", entered_at: "2025-08-01T11:00:00Z", notes: "Production scoring integration; shadow period completed." },
  { id: "monitoring", label: "Monitoring", status: "current", entered_at: "2025-08-15T08:00:00Z", notes: "Watch on DTI drift; remediation plan active." },
  { id: "periodic_review", label: "Periodic review", status: "pending", notes: "Annual review Q2." },
  { id: "retirement", label: "Retirement", status: "pending", notes: "—" },
]

function baseDetail(
  modelName: string,
  overrides: Partial<ModelRiskProfileDetail>,
): ModelRiskProfileDetail {
  const now = new Date().toISOString()
  const merged: ModelRiskProfileDetail = {
    model_name: modelName,
    model_type: overrides.model_type ?? "gradient_boosting",
    risk_tier: overrides.risk_tier ?? "medium",
    governance_profile: overrides.governance_profile ?? "Small bank",
    lifecycle_stage: overrides.lifecycle_stage ?? "monitoring",
    lifecycle_stages: overrides.lifecycle_stages ?? LIFECYCLE_FULL,
    monitoring: overrides.monitoring ?? {
      status: "healthy",
      as_of: now,
      psi_csi: [
        { feature: "loan_to_value", psi: 0.06, csi: 0.12 },
        { feature: "fico_score", psi: 0.04, csi: 0.08 },
        { feature: "dti_ratio", psi: 0.11, csi: 0.21 },
      ],
      backtest: [
        { period: "2025-Q4", metric: "MAPE", value: 0.042, benchmark: 0.05, pass: true },
        { period: "2025-Q3", metric: "MAPE", value: 0.048, benchmark: 0.05, pass: true },
      ],
      narrative: "Population drift within policy thresholds. Continue monthly monitoring.",
    },
    approvals: overrides.approvals ?? [
      { id: "a1", stage: "Model owner attestation", status: "approved", requested_at: "2025-09-01T12:00:00Z", decided_at: "2025-09-02T09:00:00Z", actor_role: "Model owner", comment: "Artifact package complete." },
      { id: "a2", stage: "MRM review", status: "approved", requested_at: "2025-09-02T10:00:00Z", decided_at: "2025-09-05T15:00:00Z", actor_role: "MRM", comment: "No material findings." },
      { id: "a3", stage: "Internal audit sample", status: "approved", requested_at: "2025-09-10T08:00:00Z", decided_at: "2025-09-18T11:00:00Z", actor_role: "Audit", comment: "Controls effective; minor documentation update noted." },
    ],
    documents: overrides.documents ?? [],
    reviews: overrides.reviews ?? [
      { id: "r1", review_type: "Scheduled periodic review", scheduled_for: "2026-06-30", completed_at: null, outcome: null, owner: "MRM" },
    ],
    metadata: overrides.metadata,
  }
  return { ...merged, ...overrides, model_name: modelName }
}

export const MOCK_CECL_NAME = "cecl_reserve_forecast_demo"
export const MOCK_CREDIT_NAME = "commercial_credit_risk_demo"

const CECL_DETAIL: ModelRiskProfileDetail = baseDetail(MOCK_CECL_NAME, {
  model_type: "glm",
  risk_tier: "high",
  governance_profile: "Small bank",
  lifecycle_stage: "monitoring",
  lifecycle_stages: LIFECYCLE_FULL,
  intended_use: "Estimate expected credit loss (ECL) for the commercial loan portfolio under CECL, for allowance reporting and management review.",
  usage_guidance: "Run monthly after core loan trial balance close. Inputs must come from the approved warehouse snapshot (T+2). Compare outputs to prior month and investigate variances above 5%. Escalate to MRM if PSI for any top-5 feature exceeds 0.25.",
  monitoring: {
    status: "healthy",
    as_of: "2026-04-01T06:00:00Z",
    psi_csi: [
      { feature: "pd_segment", psi: 0.07, csi: 0.14 },
      { feature: "lgd_proxy", psi: 0.05, csi: 0.09 },
      { feature: "ead_balance", psi: 0.09, csi: 0.11 },
    ],
    backtest: [
      { period: "2026-Q1", metric: "Weighted MAPE (ECL)", value: 0.031, benchmark: 0.045, pass: true },
      { period: "2025-Q4", metric: "Weighted MAPE (ECL)", value: 0.038, benchmark: 0.045, pass: true },
    ],
    narrative: "Back-testing within tolerance. No breach of drift policy this period.",
  },
  monitoring_pipelines: [
    { id: "pipe-cecl-psi", name: "Monthly PSI / CSI batch", schedule_cron: "0 6 3 * *", last_run_at: "2026-04-03T06:05:00Z", status: "ok" },
    { id: "pipe-cecl-bt", name: "Quarterly ECL back-test", schedule_cron: "0 7 5 1,4,7,10 *", last_run_at: "2026-04-05T07:00:00Z", status: "ok" },
    { id: "pipe-cecl-lineage", name: "Input lineage & quality checks", schedule_cron: "0 5 * * *", last_run_at: "2026-04-12T05:00:00Z", status: "ok" },
  ],
  performance_kpis: [
    { label: "Production MAPE (ECL)", value: 3.1, unit: "%", window: "Rolling 4Q", trend: "flat" },
    { label: "ROC-AUC (discrimination)", value: 0.84, window: "Validation", trend: "up" },
    { label: "Calls / month", value: 1, unit: "batch runs", window: "Apr 2026", trend: "flat" },
  ],
  inference_log: [
    {
      id: "inf-001",
      at: "2026-04-03T06:12:04Z",
      input_preview: { portfolio: "C&I", as_of_date: "2026-03-31", loan_count: 12840 },
      output_preview: { total_ecl_usd: 42_300_000, prior_month_ecl_usd: 41_100_000, delta_pct: 2.9 },
      latency_ms: 1840,
    },
    {
      id: "inf-002",
      at: "2026-03-03T06:11:58Z",
      input_preview: { portfolio: "C&I", as_of_date: "2026-02-28", loan_count: 12720 },
      output_preview: { total_ecl_usd: 41_100_000, prior_month_ecl_usd: 40_800_000, delta_pct: 0.7 },
      latency_ms: 1920,
    },
  ],
  line_defense_docs: {
    owner: {
      id: "doc-owner-cecl",
      title: "Owner quarterly model summary — Q1 2026",
      kind: "owner_report",
      updated_at: "2026-04-02T15:00:00Z",
    },
    mrm: {
      id: "doc-mrm-cecl",
      title: "MRM validation opinion & monitoring sign-off",
      kind: "mrm_report",
      updated_at: "2026-03-28T10:00:00Z",
    },
    audit: {
      id: "doc-audit-cecl",
      title: "Internal audit testing sample — CECL model",
      kind: "audit_report",
      updated_at: "2025-11-15T14:30:00Z",
    },
  },
  documents: [
    { id: "d1", title: "Business case & model inventory entry", kind: "identification", updated_at: "2025-06-01T12:00:00Z" },
    { id: "d2", title: "Assumptions & sensitivity (ECL drivers)", kind: "assumptions", updated_at: "2025-08-10T09:00:00Z" },
    { id: "d3", title: "Independent validation memo", kind: "validation", updated_at: "2025-08-20T16:00:00Z" },
  ],
  reviews: [
    { id: "r-cecl-1", review_type: "Annual full revalidation", scheduled_for: "2026-09-30", completed_at: null, outcome: null, owner: "MRM" },
    { id: "r-cecl-2", review_type: "Vendor model review (core)", scheduled_for: "2026-12-15", completed_at: null, outcome: null, owner: "Procurement + MRM" },
  ],
})

const CREDIT_DETAIL: ModelRiskProfileDetail = baseDetail(MOCK_CREDIT_NAME, {
  model_type: "xgboost",
  risk_tier: "high",
  governance_profile: "Regional bank",
  lifecycle_stage: "monitoring",
  lifecycle_stages: LIFECYCLE_CREDIT,
  intended_use: "Assign internal risk grades (1–10) for commercial & industrial loans to support underwriting workflow and portfolio reporting.",
  usage_guidance: "Invoke at booking and at annual review. Required fields: financial spreads, DTI, industry code. If any input is imputed, flag the case for manual underwriter review.",
  monitoring: {
    status: "watch",
    as_of: "2026-04-10T07:00:00Z",
    psi_csi: [
      { feature: "debt_to_income", psi: 0.22, csi: 0.41 },
      { feature: "years_in_business", psi: 0.08, csi: 0.15 },
      { feature: "industry_risk_score", psi: 0.06, csi: 0.1 },
    ],
    backtest: [
      { period: "2026-Q1", metric: "Accuracy (grade)", value: 0.78, benchmark: 0.75, pass: true },
      { period: "2026-Q1", metric: "Kappa", value: 0.62, benchmark: 0.58, pass: true },
    ],
    narrative: "Elevated PSI on DTI — characteristic shift vs training. MRM watching; no production halt.",
  },
  monitoring_pipelines: [
    { id: "pipe-cr-drift", name: "Weekly drift monitoring", schedule_cron: "0 8 * * 1", last_run_at: "2026-04-07T08:00:05Z", status: "warning" },
    { id: "pipe-cr-api", name: "Inference audit log export", schedule_cron: "0 2 * * *", last_run_at: "2026-04-12T02:00:00Z", status: "ok" },
  ],
  performance_kpis: [
    { label: "Accuracy (hold-out)", value: 79.2, unit: "%", window: "2025 validation", trend: "flat" },
    { label: "Avg inference latency", value: 45, unit: "ms", window: "7d p95", trend: "down" },
    { label: "API calls (30d)", value: 14_200, unit: "calls", window: "Mar–Apr 2026", trend: "up" },
  ],
  inference_log: [
    {
      id: "inf-cr-1",
      at: "2026-04-11T14:22:01Z",
      input_preview: { dti: 0.38, yib: 6, industry: "5412", exposure_usd: 1_200_000 },
      output_preview: { grade: 6, pd: 0.042, override_flag: false },
      latency_ms: 38,
    },
    {
      id: "inf-cr-2",
      at: "2026-04-11T13:05:44Z",
      input_preview: { dti: 0.51, yib: 2, industry: "2389", exposure_usd: 350_000 },
      output_preview: { grade: 8, pd: 0.11, override_flag: true },
      latency_ms: 52,
    },
  ],
  line_defense_docs: {
    owner: {
      id: "doc-owner-cr",
      title: "Model owner attestation — credit risk grades",
      kind: "owner_report",
      updated_at: "2026-04-01T09:00:00Z",
    },
    mrm: {
      id: "doc-mrm-cr",
      title: "MRM challenge memo & monitoring limits",
      kind: "mrm_report",
      updated_at: "2026-03-15T11:00:00Z",
    },
    audit: {
      id: "doc-audit-cr",
      title: "Audit rotation — model risk sampling",
      kind: "audit_report",
      updated_at: "2025-10-01T16:00:00Z",
    },
  },
  documents: [
    { id: "cr-d1", title: "Conceptual soundness memo", kind: "validation", updated_at: "2025-07-01T10:00:00Z" },
    { id: "cr-d2", title: "Outcome analysis — grade migration", kind: "validation", updated_at: "2025-07-15T14:00:00Z" },
  ],
  reviews: [
    { id: "r-cr-1", review_type: "Annual model review", scheduled_for: "2026-07-31", completed_at: null, outcome: null, owner: "MRM" },
  ],
  approvals: [
    { id: "c1", stage: "Owner certification", status: "approved", requested_at: "2025-07-18T12:00:00Z", decided_at: "2025-07-19T09:00:00Z", actor_role: "Model owner", comment: "—" },
    { id: "c2", stage: "MRM approval", status: "approved", requested_at: "2025-07-19T10:00:00Z", decided_at: "2025-07-22T16:00:00Z", actor_role: "MRM", comment: "Approved with DTI monitoring add-on." },
    { id: "c3", stage: "Committee (high risk)", status: "approved", requested_at: "2025-07-22T17:00:00Z", decided_at: "2025-07-25T14:00:00Z", actor_role: "Committee", comment: "Recorded in minutes 2025-07-25." },
  ],
})

const DETAIL_BY_NAME: Record<string, ModelRiskProfileDetail> = {
  [MOCK_CECL_NAME]: CECL_DETAIL,
  [MOCK_CREDIT_NAME]: CREDIT_DETAIL,
}

function buildTrainedRegistryDetail(
  modelName: string,
  entry: TrainedRegistryEntry,
): ModelRiskProfileDetail {
  return baseDetail(modelName, {
    model_type: entry.model_type,
    risk_tier: "medium",
    governance_profile: "Experiment lab",
    lifecycle_stage: "development",
    lifecycle_stages: LIFECYCLE_DEVELOPMENT_IN_FLIGHT,
    intended_use: entry.intended_use,
    intended_use_summary: entry.intended_use_summary,
    usage_guidance:
      entry.usage_guidance ??
      "Use trained model output for the business purpose documented in the linked experiment. Match the training report feature schema and data dictionary.",
    monitoring: MONITORING_HEALTHY_DEV,
    reviews: REVIEWS_STANDARD,
    performance_kpis:
      entry.performance_kpis ??
      [{ label: "Registry stage", value: 1, unit: "development", window: "Current", trend: "flat" }],
    monitoring_pipelines: [
      {
        id: `pipe-${modelName}-drift`,
        name: "Drift / performance check (scheduled post-deployment)",
        schedule_cron: "0 6 * * 1",
        last_run_at: "2026-04-11T06:00:00Z",
        status: "ok",
      },
    ],
    approvals: [
      {
        id: `${modelName}-a1`,
        stage: "Development sign-off",
        status: "pending",
        requested_at: "2026-04-01T12:00:00Z",
        actor_role: "Model owner",
        comment: "Complete documentation and validation package before approval.",
      },
    ],
    line_defense_docs: {
      owner: {
        id: `${modelName}-owner-doc`,
        title: "Owner worksheet — development",
        kind: "owner_report",
        updated_at: "2026-04-02T10:00:00Z",
      },
      mrm: {
        id: `${modelName}-mrm-doc`,
        title: "MRM review checklist — not yet final",
        kind: "mrm_report",
        updated_at: null,
      },
      audit: {
        id: `${modelName}-audit-doc`,
        title: "Audit linkage — pending implementation",
        kind: "audit_report",
        updated_at: null,
      },
    },
    documents: [
      {
        id: `${modelName}-d1`,
        title: "Linked experiment & training report",
        kind: "development",
        updated_at: "2026-04-10T12:00:00Z",
      },
    ],
    inference_log: [
      {
        id: `${modelName}-sample-call`,
        at: "2026-04-11T15:00:00Z",
        input_preview: { note: "Shape matches training columns", batch_rows: 1 },
        output_preview: { note: "See metrics in registry row", status: "ok" },
        latency_ms: 120,
      },
    ],
  })
}

export function getMockModelRiskDetail(modelName: string): ModelRiskProfileDetail {
  const key = modelName.trim()
  const trained = TRAINED_REGISTRY[key]
  if (trained) return buildTrainedRegistryDetail(key, trained)

  const existing = DETAIL_BY_NAME[key]
  if (existing) return { ...existing, model_name: key }

  return genericDetailForUnknownModel(key)
}

function genericDetailForUnknownModel(modelName: string): ModelRiskProfileDetail {
  return baseDetail(modelName, {
    model_type: "trained",
    risk_tier: "medium",
    governance_profile: "Experiment lab",
    lifecycle_stage: "development",
    lifecycle_stages: LIFECYCLE_DEVELOPMENT_IN_FLIGHT,
    intended_use:
      "Use trained model output for the business purpose documented in the linked experiment. Complete identification and validation before production use.",
    usage_guidance: "Use only with the feature schema from the training report. Validate inputs against training ranges.",
    monitoring: MONITORING_HEALTHY_DEV,
    reviews: REVIEWS_STANDARD,
    monitoring_pipelines: [
      {
        id: "pipe-generic",
        name: "Post-training monitoring (not yet scheduled)",
        schedule_cron: "—",
        last_run_at: "2026-04-11T06:00:00Z",
        status: "warning",
      },
    ],
    performance_kpis: [{ label: "Registry status", value: 1, unit: "in development", window: "Current", trend: "flat" }],
    inference_log: [],
    line_defense_docs: {
      owner: { id: "g-o", title: "Owner report — pending", kind: "owner_report", updated_at: null },
      mrm: { id: "g-m", title: "MRM review — pending", kind: "mrm_report", updated_at: null },
      audit: { id: "g-a", title: "Audit sample — pending", kind: "audit_report", updated_at: null },
    },
  })
}

export function inventoryItemFromProfile(d: ModelRiskProfileDetail): ModelRiskInventoryItem {
  const nextReview = d.reviews.find((r) => r.scheduled_for && !r.completed_at)?.scheduled_for ?? null
  const summary = d.intended_use_summary ?? d.intended_use?.slice(0, 120) ?? undefined
  return {
    model_name: d.model_name,
    model_type: d.model_type,
    risk_tier: d.risk_tier,
    lifecycle_stage: d.lifecycle_stage,
    monitoring_status: d.monitoring.status,
    governance_profile: d.governance_profile,
    last_reviewed_at: d.reviews.find((r) => r.completed_at)?.completed_at ?? null,
    intended_use_summary: summary,
    next_review_due: nextReview,
  }
}

function inventoryFromDetail(d: ModelRiskProfileDetail): ModelRiskInventoryItem {
  return inventoryItemFromProfile(d)
}

export function getMockModelRiskInventory(): ModelRiskInventoryItem[] {
  return [inventoryFromDetail(CECL_DETAIL), inventoryFromDetail(CREDIT_DETAIL)]
}

/** Merge API inventory with synthetic rows for trained models missing from the demo list. */
export function mergeMockInventoryWithTrainedModelNames(trainedNames: string[]): ModelRiskInventoryItem[] {
  const base = getMockModelRiskInventory()
  const map = new Map(base.map((i) => [i.model_name, i]))
  for (const name of trainedNames) {
    if (!map.has(name)) {
      map.set(name, inventoryItemFromProfile(getMockModelRiskDetail(name)))
    }
  }
  return Array.from(map.values())
}

export function getMockModelRiskDashboard(portal: ModelRiskPortalRole): ModelRiskDashboard {
  const items = getMockModelRiskInventory()
  const headlines: Record<ModelRiskPortalRole, string> = {
    owner: "Your owned models: attestations, remediation, and monitoring responses.",
    validation: "Challenge function: inventory coverage, validation backlog, and drift exceptions.",
    board: "Oversight: concentration of high-risk models and upcoming reviews.",
  }
  return {
    portal,
    headline: headlines[portal],
    counts: {
      models: items.length,
      pending_approvals: 0,
      monitoring_watch: items.filter((i) => i.monitoring_status === "watch").length,
      breaches: items.filter((i) => i.monitoring_status === "breach").length,
    },
    items: items.map((it) => ({
      model_name: it.model_name,
      snippet: it.intended_use_summary ?? it.lifecycle_stage,
      risk_tier: it.risk_tier,
      monitoring_status: it.monitoring_status,
    })),
  }
}
