import React, { useEffect, useMemo, useRef, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Separator } from "@/components/ui/separator";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import {
  Play,
  Paperclip,
  Sparkles,
  FileText,
  Beaker,
  ShieldCheck,
  Database,
  Settings2,
  LineChart,
  Link as LinkIcon,
  CheckCircle2,
  Loader2,
} from "lucide-react";

/**
 * POC UI ONLY.
 * - Chat-first layout
 * - Reactive side panel driven by a simple state machine
 * - Simulated runs (no backend)
 *
 * Wire-up later:
 * - Replace `simulateRun()` with real orchestration calls
 * - Stream tool logs into `traceEvents`
 * - Persist `artifacts` + `experiments`
 */

const PHASES = ["data", "plan", "training", "results", "audit"] as const;

type Phase = (typeof PHASES)[number];

type Msg = {
  id: string;
  role: "user" | "agent";
  text: string;
  ts: number;
  links?: { kind: "artifact" | "dataset" | "experiment"; id: string; label: string }[];
};

type Artifact = {
  id: string;
  kind: "model" | "eval" | "config" | "dataset";
  name: string;
  createdAt: number;
  meta?: Record<string, string>;
};

type Experiment = {
  id: string;
  name: string;
  createdAt: number;
  status: "queued" | "running" | "succeeded" | "failed";
  metrics?: { label: string; value: string }[];
  notes?: string;
};

type TraceEvent = {
  id: string;
  ts: number;
  level: "info" | "warn" | "error";
  title: string;
  detail?: string;
};

function uid(prefix = "id") {
  return `${prefix}_${Math.random().toString(16).slice(2)}_${Date.now().toString(16)}`;
}

function fmtTime(ts: number) {
  const d = new Date(ts);
  return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function phaseLabel(p: Phase) {
  switch (p) {
    case "data":
      return "Data";
    case "plan":
      return "Plan";
    case "training":
      return "Training";
    case "results":
      return "Results";
    case "audit":
      return "Audit";
  }
}

function phaseIcon(p: Phase) {
  switch (p) {
    case "data":
      return <Database className="h-4 w-4" />;
    case "plan":
      return <Settings2 className="h-4 w-4" />;
    case "training":
      return <Beaker className="h-4 w-4" />;
    case "results":
      return <LineChart className="h-4 w-4" />;
    case "audit":
      return <ShieldCheck className="h-4 w-4" />;
  }
}

export default function PocMlWorkspace() {
  const [phase, setPhase] = useState<Phase>("data");
  const [activeTab, setActiveTab] = useState<"panel" | "artifacts" | "experiments" | "trace">("panel");

  // Autonomy control (POC toggle)
  const [autoMode, setAutoMode] = useState(false);

  // Chat
  const [messages, setMessages] = useState<Msg[]>(() => [
    {
      id: uid("m"),
      role: "agent",
      ts: Date.now(),
      text:
        "Welcome. Tell me what model you want to build. Example: ‘Train a churn model optimizing recall. Use my CSV.’",
    },
  ]);
  const [draft, setDraft] = useState("");

  // Data (POC)
  const [datasetName, setDatasetName] = useState("churn.csv");
  const [targetCol, setTargetCol] = useState("churn");
  const [previewRows] = useState([
    { customer_id: "A102", tenure_months: 4, plan: "basic", monthly_spend: 32.5, churn: 1 },
    { customer_id: "B441", tenure_months: 18, plan: "pro", monthly_spend: 88.0, churn: 0 },
    { customer_id: "C009", tenure_months: 2, plan: "basic", monthly_spend: 25.1, churn: 1 },
    { customer_id: "D777", tenure_months: 31, plan: "pro", monthly_spend: 92.3, churn: 0 },
  ]);

  // Plan (POC)
  const [metric, setMetric] = useState("recall");
  const [split, setSplit] = useState("80/20");
  const [baselineModel, setBaselineModel] = useState("Logistic Regression");
  const [candidateModel, setCandidateModel] = useState("XGBoost");
  const [hyperparams, setHyperparams] = useState("n_estimators=300\nmax_depth=6\nlearning_rate=0.05\nsubsample=0.9");
  const [comments, setComments] = useState("Prioritize catching churners; false negatives are more costly.");

  // Execution (POC)
  const [isRunning, setIsRunning] = useState(false);
  const [progress, setProgress] = useState(0);
  const [statusLine, setStatusLine] = useState<string>("");

  // Artifacts / Experiments / Trace
  const [artifacts, setArtifacts] = useState<Artifact[]>(() => [
    { id: uid("a"), kind: "dataset", name: "dataset@v1 (churn.csv)", createdAt: Date.now() - 60_000, meta: { rows: "10,000", cols: "24" } },
  ]);
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [traceEvents, setTraceEvents] = useState<TraceEvent[]>([]);

  const chatEndRef = useRef<HTMLDivElement | null>(null);
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages.length]);

  const currentExperiment = useMemo(() => {
    const running = [...experiments].reverse().find((e) => e.status === "running" || e.status === "queued");
    return running ?? null;
  }, [experiments]);

  function pushAgent(text: string, links?: Msg["links"]) {
    setMessages((m) => [...m, { id: uid("m"), role: "agent", ts: Date.now(), text, links }]);
  }

  function pushUser(text: string) {
    setMessages((m) => [...m, { id: uid("m"), role: "user", ts: Date.now(), text }]);
  }

  function addTrace(level: TraceEvent["level"], title: string, detail?: string) {
    setTraceEvents((t) => [...t, { id: uid("t"), ts: Date.now(), level, title, detail }]);
  }

  function linkToChat(kind: "artifact" | "dataset" | "experiment", id: string, label: string) {
    const icon = kind === "artifact" ? "📎" : kind === "dataset" ? "🗂️" : "🧪";
    pushAgent(`Linked ${kind}: ${label}`, [{ kind, id, label }]);
  }

  function onSend() {
    const text = draft.trim();
    if (!text) return;
    pushUser(text);
    setDraft("");

    // Simple intent handling for the POC
    if (/train|fit|model/i.test(text)) {
      setPhase("plan");
      setActiveTab("panel");
      pushAgent(
        "Got it. I’ll propose a plan in the side panel. Review and click ‘Run experiment’. You can edit metric, split, models, and hyperparameters.",
      );
      if (autoMode) {
        // auto-run after a short beat
        window.setTimeout(() => {
          if (!isRunning) simulateRun();
        }, 600);
      }
    } else if (/data|schema|columns|preview/i.test(text)) {
      setPhase("data");
      setActiveTab("panel");
      pushAgent("Showing a dataset preview and basic stats in the side panel. You can rename the dataset and confirm the target column.");
    } else if (/audit|trace|lineage/i.test(text)) {
      setPhase("audit");
      setActiveTab("panel");
      pushAgent("Opening the audit + lineage view. You can click any trace item to link it back to the conversation.");
    } else {
      pushAgent("Noted. For this POC, try asking me to train a model (e.g., ‘train churn model optimizing recall’).");
    }
  }

  async function simulateRun() {
    if (isRunning) return;

    setIsRunning(true);
    setProgress(0);
    setStatusLine("Queued…");

    // Create experiment
    const expId = uid("exp");
    const expName = `Churn experiment (${baselineModel} → ${candidateModel})`;
    setExperiments((e) => [
      ...e,
      { id: expId, name: expName, createdAt: Date.now(), status: "queued", notes: `metric=${metric}, split=${split}` },
    ]);

    addTrace("info", "Experiment queued", `name=${expName}`);

    setPhase("training");
    setActiveTab("panel");

    pushAgent(
      "I’m starting the run now. Watch the side panel for progress. I’ll produce artifacts (model, eval report, config) and a full trace.",
    );

    const steps: { p: number; line: string; trace: [TraceEvent["level"], string, string?] }[] = [
      { p: 10, line: "Validating dataset + schema…", trace: ["info", "Validated dataset", `dataset=${datasetName}, target=${targetCol}`] },
      { p: 25, line: "Building eval + split…", trace: ["info", "Prepared evaluation", `metric=${metric}, split=${split}`] },
      { p: 40, line: `Training baseline: ${baselineModel}…`, trace: ["info", "Trained baseline", baselineModel] },
      { p: 60, line: `Training candidate: ${candidateModel}…`, trace: ["info", "Trained candidate", candidateModel] },
      { p: 78, line: "Evaluating + comparing…", trace: ["info", "Computed metrics", "baseline vs candidate"] },
      { p: 90, line: "Packaging artifacts…", trace: ["info", "Created artifacts", "model + eval + config"] },
      { p: 100, line: "Finalizing trace + lineage…", trace: ["info", "Finalized audit trail", "lineage + hashes"] },
    ];

    // flip to running
    setExperiments((exps) => exps.map((x) => (x.id === expId ? { ...x, status: "running" } : x)));

    for (const s of steps) {
      // eslint-disable-next-line no-await-in-loop
      await new Promise((r) => window.setTimeout(r, 650));
      setProgress(s.p);
      setStatusLine(s.line);
      addTrace(...s.trace);
    }

    // Create artifacts
    const modelArtifact: Artifact = {
      id: uid("a"),
      kind: "model",
      name: `churn_model_${Date.now().toString(16)}.pkl`,
      createdAt: Date.now(),
      meta: { model: candidateModel, metric, split },
    };
    const evalArtifact: Artifact = {
      id: uid("a"),
      kind: "eval",
      name: `eval_${Date.now().toString(16)}.json`,
      createdAt: Date.now(),
      meta: { recall: "0.86", precision: "0.41", auc: "0.78" },
    };
    const cfgArtifact: Artifact = {
      id: uid("a"),
      kind: "config",
      name: `config_${Date.now().toString(16)}.yaml`,
      createdAt: Date.now(),
      meta: { baseline: baselineModel, candidate: candidateModel },
    };

    setArtifacts((a) => [
      ...a,
      modelArtifact,
      evalArtifact,
      cfgArtifact,
    ]);

    setExperiments((exps) =>
      exps.map((x) =>
        x.id === expId
          ? {
              ...x,
              status: "succeeded",
              metrics: [
                { label: "Recall", value: "0.86 (+0.12)" },
                { label: "Precision", value: "0.41 (-0.02)" },
                { label: "AUC", value: "0.78 (+0.06)" },
              ],
            }
          : x,
      ),
    );

    addTrace("info", "Experiment succeeded", `exp_id=${expId}`);

    setPhase("results");
    setActiveTab("panel");

    pushAgent(
      "Run complete. Candidate improved recall by +0.12. I’ve attached artifacts and a full trace. You can click an artifact to link it into the chat.",
      [
        { kind: "artifact", id: modelArtifact.id, label: modelArtifact.name },
        { kind: "artifact", id: evalArtifact.id, label: evalArtifact.name },
        { kind: "artifact", id: cfgArtifact.id, label: cfgArtifact.name },
        { kind: "experiment", id: expId, label: expName },
      ],
    );

    setIsRunning(false);
    setStatusLine("");
  }

  const phasePill = (
    <Badge variant="secondary" className="gap-2">
      {phaseIcon(phase)}
      <span>{phaseLabel(phase)}</span>
    </Badge>
  );

  return (
    <div className="min-h-screen w-full bg-background p-4">
      <div className="mx-auto grid max-w-6xl grid-cols-1 gap-4 lg:grid-cols-12">
        {/* Header */}
        <div className="lg:col-span-12">
          <Card className="rounded-2xl">
            <CardContent className="flex flex-col gap-3 p-4 md:flex-row md:items-center md:justify-between">
              <div className="flex items-center gap-3">
                <div className="flex h-10 w-10 items-center justify-center rounded-2xl bg-muted">
                  <Sparkles className="h-5 w-5" />
                </div>
                <div>
                  <div className="text-lg font-semibold leading-tight">POC: AI-Native ML Workspace</div>
                  <div className="text-sm text-muted-foreground">Chat-first • Reactive side panel • Artifacts + trace</div>
                </div>
              </div>
              <div className="flex flex-wrap items-center gap-3">
                {phasePill}
                <div className="flex items-center gap-2 rounded-xl border px-3 py-2">
                  <Switch id="auto" checked={autoMode} onCheckedChange={setAutoMode} />
                  <Label htmlFor="auto" className="text-sm">Auto mode</Label>
                </div>
                <Button
                  className="rounded-xl"
                  variant={isRunning ? "secondary" : "default"}
                  onClick={() => (isRunning ? null : simulateRun())}
                  disabled={isRunning}
                >
                  {isRunning ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <Play className="mr-2 h-4 w-4" />}
                  {isRunning ? "Running" : "Run experiment"}
                </Button>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Chat */}
        <div className="lg:col-span-7">
          <Card className="h-[78vh] rounded-2xl">
            <CardHeader className="pb-3">
              <CardTitle className="flex items-center justify-between">
                <span className="flex items-center gap-2"><Sparkles className="h-4 w-4" /> Chat</span>
                <span className="text-xs font-normal text-muted-foreground">Prompt what you want to do and iterate</span>
              </CardTitle>
            </CardHeader>
            <CardContent className="flex h-[calc(78vh-76px)] flex-col gap-3 p-4 pt-0">
              <ScrollArea className="flex-1 rounded-xl border p-3">
                <div className="space-y-3">
                  {messages.map((m) => (
                    <div key={m.id} className={`flex ${m.role === "user" ? "justify-end" : "justify-start"}`}>
                      <div
                        className={`max-w-[92%] rounded-2xl px-3 py-2 text-sm shadow-sm ${
                          m.role === "user" ? "bg-primary text-primary-foreground" : "bg-muted"
                        }`}
                      >
                        <div className="flex items-center justify-between gap-3">
                          <div className="font-medium">{m.role === "user" ? "You" : "Agent"}</div>
                          <div className={`text-xs ${m.role === "user" ? "text-primary-foreground/80" : "text-muted-foreground"}`}>
                            {fmtTime(m.ts)}
                          </div>
                        </div>
                        <div className="mt-1 whitespace-pre-wrap leading-relaxed">{m.text}</div>

                        {!!m.links?.length && (
                          <div className="mt-2 space-y-1">
                            {m.links.map((l) => (
                              <button
                                key={`${m.id}-${l.id}`}
                                className={`flex w-full items-center gap-2 rounded-xl px-2 py-1 text-left text-xs ${
                                  m.role === "user" ? "bg-primary-foreground/10" : "bg-background"
                                } hover:opacity-90`}
                                onClick={() => {
                                  setActiveTab(l.kind === "experiment" ? "experiments" : l.kind === "dataset" ? "panel" : "artifacts");
                                  if (l.kind === "experiment") setPhase("results");
                                }}
                              >
                                <LinkIcon className="h-3.5 w-3.5" />
                                <span className="truncate">{l.label}</span>
                                <Badge variant="outline" className="ml-auto text-[10px]">
                                  {l.kind}
                                </Badge>
                              </button>
                            ))}
                          </div>
                        )}
                      </div>
                    </div>
                  ))}
                  <div ref={chatEndRef} />
                </div>
              </ScrollArea>

              <div className="flex items-end gap-2">
                <Textarea
                  value={draft}
                  onChange={(e) => setDraft(e.target.value)}
                  placeholder="Try: ‘Train a churn model optimizing recall.’ or ‘Show me the dataset preview.’"
                  className="min-h-[44px] resize-none rounded-2xl"
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
                      e.preventDefault();
                      onSend();
                    }
                  }}
                />
                <Button className="rounded-2xl" onClick={onSend}>
                  Send
                </Button>
              </div>
              <div className="text-xs text-muted-foreground">Tip: Press Ctrl/Cmd + Enter to send.</div>
            </CardContent>
          </Card>
        </div>

        {/* Side Panel */}
        <div className="lg:col-span-5">
          <Card className="h-[78vh] rounded-2xl">
            <CardHeader className="pb-3">
              <CardTitle className="flex items-center justify-between">
                <span className="flex items-center gap-2"><Paperclip className="h-4 w-4" /> Side Panel</span>
                <span className="text-xs font-normal text-muted-foreground">Reactive to your workflow stage</span>
              </CardTitle>
            </CardHeader>
            <CardContent className="h-[calc(78vh-76px)] p-4 pt-0">
              <Tabs value={activeTab} onValueChange={(v) => setActiveTab(v as any)} className="h-full">
                <TabsList className="grid w-full grid-cols-4 rounded-2xl">
                  <TabsTrigger value="panel" className="rounded-2xl">Panel</TabsTrigger>
                  <TabsTrigger value="artifacts" className="rounded-2xl">Artifacts</TabsTrigger>
                  <TabsTrigger value="experiments" className="rounded-2xl">Experiments</TabsTrigger>
                  <TabsTrigger value="trace" className="rounded-2xl">Trace</TabsTrigger>
                </TabsList>

                <TabsContent value="panel" className="mt-3 h-[calc(100%-52px)]">
                  <ScrollArea className="h-full rounded-2xl border p-3">
                    <div className="space-y-4">
                      <div className="flex items-center justify-between">
                        <div className="text-sm font-semibold">{phaseLabel(phase)} panel</div>
                        <Badge variant="outline" className="gap-2">
                          {phaseIcon(phase)}
                          <span className="text-[10px]">{phase}</span>
                        </Badge>
                      </div>

                      {/* Phase-specific */}
                      {phase === "data" && (
                        <>
                          <Card className="rounded-2xl">
                            <CardHeader className="pb-2">
                              <CardTitle className="text-sm flex items-center gap-2"><Database className="h-4 w-4" /> Dataset</CardTitle>
                            </CardHeader>
                            <CardContent className="space-y-3">
                              <div className="grid grid-cols-2 gap-2">
                                <div>
                                  <div className="text-xs text-muted-foreground">Name</div>
                                  <Input value={datasetName} onChange={(e) => setDatasetName(e.target.value)} className="rounded-xl" />
                                </div>
                                <div>
                                  <div className="text-xs text-muted-foreground">Target</div>
                                  <Input value={targetCol} onChange={(e) => setTargetCol(e.target.value)} className="rounded-xl" />
                                </div>
                              </div>
                              <div className="text-xs text-muted-foreground">Preview</div>
                              <div className="rounded-xl border bg-background p-2 text-xs">
                                <pre className="overflow-x-auto">{JSON.stringify(previewRows, null, 2)}</pre>
                              </div>
                              <div className="flex gap-2">
                                <Button
                                  variant="secondary"
                                  className="rounded-xl"
                                  onClick={() => {
                                    const ds = artifacts.find((a) => a.kind === "dataset");
                                    if (ds) linkToChat("dataset", ds.id, ds.name);
                                  }}
                                >
                                  <LinkIcon className="mr-2 h-4 w-4" /> Link dataset to chat
                                </Button>
                                <Button
                                  className="rounded-xl"
                                  onClick={() => {
                                    setPhase("plan");
                                    pushAgent("Dataset confirmed. Next I’ll propose the training plan in the side panel.");
                                  }}
                                >
                                  Continue to plan
                                </Button>
                              </div>
                            </CardContent>
                          </Card>
                        </>
                      )}

                      {phase === "plan" && (
                        <>
                          <Card className="rounded-2xl">
                            <CardHeader className="pb-2">
                              <CardTitle className="text-sm flex items-center gap-2"><Settings2 className="h-4 w-4" /> Proposed plan (editable)</CardTitle>
                            </CardHeader>
                            <CardContent className="space-y-3">
                              <div className="grid grid-cols-2 gap-2">
                                <div>
                                  <div className="text-xs text-muted-foreground">Metric</div>
                                  <Input value={metric} onChange={(e) => setMetric(e.target.value)} className="rounded-xl" />
                                </div>
                                <div>
                                  <div className="text-xs text-muted-foreground">Split</div>
                                  <Input value={split} onChange={(e) => setSplit(e.target.value)} className="rounded-xl" />
                                </div>
                                <div>
                                  <div className="text-xs text-muted-foreground">Baseline model</div>
                                  <Input value={baselineModel} onChange={(e) => setBaselineModel(e.target.value)} className="rounded-xl" />
                                </div>
                                <div>
                                  <div className="text-xs text-muted-foreground">Candidate model</div>
                                  <Input value={candidateModel} onChange={(e) => setCandidateModel(e.target.value)} className="rounded-xl" />
                                </div>
                              </div>
                              <div>
                                <div className="text-xs text-muted-foreground">Hyperparameters</div>
                                <Textarea value={hyperparams} onChange={(e) => setHyperparams(e.target.value)} className="rounded-2xl" />
                              </div>
                              <div>
                                <div className="text-xs text-muted-foreground">Comments</div>
                                <Textarea value={comments} onChange={(e) => setComments(e.target.value)} className="rounded-2xl" />
                              </div>

                              <div className="rounded-2xl border bg-muted/40 p-3 text-xs">
                                <div className="font-semibold">What will happen</div>
                                <ul className="mt-2 list-disc space-y-1 pl-5">
                                  <li>Validate dataset + target</li>
                                  <li>Build evaluation for <b>{metric}</b> using split <b>{split}</b></li>
                                  <li>Train baseline: <b>{baselineModel}</b></li>
                                  <li>Train candidate: <b>{candidateModel}</b></li>
                                  <li>Compare metrics and package artifacts</li>
                                  <li>Emit full audit + lineage trace</li>
                                </ul>
                              </div>

                              <div className="flex flex-wrap gap-2">
                                <Button className="rounded-xl" onClick={simulateRun} disabled={isRunning}>
                                  <Play className="mr-2 h-4 w-4" /> Run experiment
                                </Button>
                                <Button
                                  variant="secondary"
                                  className="rounded-xl"
                                  onClick={() => {
                                    pushAgent(
                                      `Plan updated: metric=${metric}, split=${split}, baseline=${baselineModel}, candidate=${candidateModel}.`,
                                    );
                                  }}
                                >
                                  Save notes to chat
                                </Button>
                              </div>
                            </CardContent>
                          </Card>
                        </>
                      )}

                      {phase === "training" && (
                        <>
                          <Card className="rounded-2xl">
                            <CardHeader className="pb-2">
                              <CardTitle className="text-sm flex items-center gap-2"><Beaker className="h-4 w-4" /> Training progress</CardTitle>
                            </CardHeader>
                            <CardContent className="space-y-3">
                              <div className="flex items-center justify-between text-xs">
                                <div className="text-muted-foreground">{statusLine || "Running…"}</div>
                                <div className="font-medium">{progress}%</div>
                              </div>
                              <Progress value={progress} />

                              <Separator />

                              <div className="text-xs">
                                <div className="font-semibold">Current run</div>
                                <div className="mt-1 text-muted-foreground">
                                  {currentExperiment ? currentExperiment.name : "—"}
                                </div>
                              </div>

                              <div className="rounded-2xl border bg-background p-3 text-xs">
                                <div className="font-semibold">Live notes</div>
                                <div className="mt-1 text-muted-foreground">We’re keeping full trace, params, and dataset lineage.</div>
                              </div>
                            </CardContent>
                          </Card>
                        </>
                      )}

                      {phase === "results" && (
                        <>
                          <Card className="rounded-2xl">
                            <CardHeader className="pb-2">
                              <CardTitle className="text-sm flex items-center gap-2"><LineChart className="h-4 w-4" /> Results</CardTitle>
                            </CardHeader>
                            <CardContent className="space-y-3">
                              <div className="rounded-2xl border bg-muted/40 p-3">
                                <div className="text-xs text-muted-foreground">Summary</div>
                                <div className="mt-1 text-sm font-semibold">Recall improved (+0.12) using {candidateModel}</div>
                                <div className="mt-1 text-xs text-muted-foreground">Precision decreased slightly; review tradeoffs and adjust threshold if needed.</div>
                              </div>

                              <div className="grid grid-cols-3 gap-2">
                                <MetricCard title="Recall" value="0.86" delta="+0.12" />
                                <MetricCard title="Precision" value="0.41" delta="-0.02" />
                                <MetricCard title="AUC" value="0.78" delta="+0.06" />
                              </div>

                              <div className="flex flex-wrap gap-2">
                                <Button className="rounded-xl" onClick={() => { setPhase("audit"); setActiveTab("panel"); }}>
                                  <ShieldCheck className="mr-2 h-4 w-4" /> View audit trail
                                </Button>
                                <Button
                                  variant="secondary"
                                  className="rounded-xl"
                                  onClick={() => {
                                    setPhase("plan");
                                    pushAgent("Let’s iterate. Update metric/params in the plan panel and run again.");
                                  }}
                                >
                                  Iterate again
                                </Button>
                              </div>

                              <div className="rounded-2xl border bg-background p-3 text-xs">
                                <div className="font-semibold">Artifacts created</div>
                                <div className="mt-2 space-y-1">
                                  {artifacts
                                    .filter((a) => a.kind !== "dataset")
                                    .slice(-3)
                                    .map((a) => (
                                      <button
                                        key={a.id}
                                        className="flex w-full items-center gap-2 rounded-xl px-2 py-1 hover:bg-muted"
                                        onClick={() => linkToChat("artifact", a.id, a.name)}
                                      >
                                        <FileText className="h-4 w-4" />
                                        <span className="truncate">{a.name}</span>
                                        <Badge variant="outline" className="ml-auto text-[10px]">{a.kind}</Badge>
                                      </button>
                                    ))}
                                </div>
                              </div>

                              <div className="rounded-2xl border bg-muted/40 p-3 text-xs">
                                <div className="font-semibold">(Optional in future) Auto-deploy</div>
                                <div className="mt-1 text-muted-foreground">POC placeholder. In v2: deploy to API endpoint, batch scoring, or agent tool-call.</div>
                              </div>
                            </CardContent>
                          </Card>
                        </>
                      )}

                      {phase === "audit" && (
                        <>
                          <Card className="rounded-2xl">
                            <CardHeader className="pb-2">
                              <CardTitle className="text-sm flex items-center gap-2"><ShieldCheck className="h-4 w-4" /> Audit & lineage</CardTitle>
                            </CardHeader>
                            <CardContent className="space-y-3">
                              <div className="rounded-2xl border bg-background p-3 text-xs">
                                <div className="font-semibold">Lineage</div>
                                <div className="mt-2 grid grid-cols-1 gap-2">
                                  <LineageNode label={`Dataset: ${datasetName}`} sub="hash=ds_9af…" onClick={() => {
                                    const ds = artifacts.find((a) => a.kind === "dataset");
                                    if (ds) linkToChat("dataset", ds.id, ds.name);
                                  }} />
                                  <LineageNode label={`Config: metric=${metric}, split=${split}`} sub="hash=cfg_21c…" onClick={() => {
                                    const cfg = [...artifacts].reverse().find((a) => a.kind === "config");
                                    if (cfg) linkToChat("artifact", cfg.id, cfg.name);
                                  }} />
                                  <LineageNode label={`Model: ${candidateModel}`} sub="hash=mdl_55b…" onClick={() => {
                                    const mdl = [...artifacts].reverse().find((a) => a.kind === "model");
                                    if (mdl) linkToChat("artifact", mdl.id, mdl.name);
                                  }} />
                                  <LineageNode label="Eval report" sub="hash=eval_88d…" onClick={() => {
                                    const ev = [...artifacts].reverse().find((a) => a.kind === "eval");
                                    if (ev) linkToChat("artifact", ev.id, ev.name);
                                  }} />
                                </div>
                              </div>

                              <div className="rounded-2xl border bg-muted/40 p-3 text-xs">
                                <div className="font-semibold">Compliance-ready trace</div>
                                <div className="mt-1 text-muted-foreground">Every step is logged with parameters and decisions. Click any step in the Trace tab to link it to the chat.</div>
                              </div>

                              <div className="flex flex-wrap gap-2">
                                <Button className="rounded-xl" onClick={() => { setActiveTab("trace"); }}>
                                  <FileText className="mr-2 h-4 w-4" /> Open trace
                                </Button>
                                <Button variant="secondary" className="rounded-xl" onClick={() => { setPhase("results"); setActiveTab("panel"); }}>
                                  Back to results
                                </Button>
                              </div>

                              <div className="rounded-2xl border bg-background p-3 text-xs">
                                <div className="font-semibold">(Future) Turn model into a tool</div>
                                <div className="mt-1 text-muted-foreground">POC placeholder. In v2: register model as a tool callable by agents (predict, score, batch).</div>
                              </div>
                            </CardContent>
                          </Card>
                        </>
                      )}
                    </div>
                  </ScrollArea>
                </TabsContent>

                <TabsContent value="artifacts" className="mt-3 h-[calc(100%-52px)]">
                  <ScrollArea className="h-full rounded-2xl border p-3">
                    <div className="flex items-center justify-between">
                      <div className="text-sm font-semibold">Artifacts</div>
                      <Badge variant="secondary" className="text-[10px]">{artifacts.length}</Badge>
                    </div>
                    <Separator className="my-3" />
                    <div className="space-y-2">
                      {([...artifacts].reverse()).map((a) => (
                        <button
                          key={a.id}
                          className="w-full rounded-2xl border bg-background p-3 text-left hover:bg-muted"
                          onClick={() => linkToChat(a.kind === "dataset" ? "dataset" : "artifact", a.id, a.name)}
                        >
                          <div className="flex items-start gap-2">
                            <FileText className="mt-0.5 h-4 w-4" />
                            <div className="min-w-0 flex-1">
                              <div className="flex items-center gap-2">
                                <div className="truncate text-sm font-semibold">{a.name}</div>
                                <Badge variant="outline" className="text-[10px]">{a.kind}</Badge>
                              </div>
                              <div className="mt-1 text-xs text-muted-foreground">Created {new Date(a.createdAt).toLocaleString()}</div>
                              {!!a.meta && (
                                <div className="mt-2 flex flex-wrap gap-1">
                                  {Object.entries(a.meta).map(([k, v]) => (
                                    <Badge key={`${a.id}-${k}`} variant="secondary" className="text-[10px]">
                                      {k}: {v}
                                    </Badge>
                                  ))}
                                </div>
                              )}
                            </div>
                            <div className="flex items-center gap-1 text-xs text-muted-foreground">
                              <LinkIcon className="h-3.5 w-3.5" /> Link
                            </div>
                          </div>
                        </button>
                      ))}
                    </div>
                  </ScrollArea>
                </TabsContent>

                <TabsContent value="experiments" className="mt-3 h-[calc(100%-52px)]">
                  <ScrollArea className="h-full rounded-2xl border p-3">
                    <div className="flex items-center justify-between">
                      <div className="text-sm font-semibold">Experiments</div>
                      <Badge variant="secondary" className="text-[10px]">{experiments.length}</Badge>
                    </div>
                    <Separator className="my-3" />
                    {experiments.length === 0 ? (
                      <div className="text-sm text-muted-foreground">No experiments yet. Click “Run experiment”.</div>
                    ) : (
                      <div className="space-y-2">
                        {[...experiments].reverse().map((e) => (
                          <button
                            key={e.id}
                            className="w-full rounded-2xl border bg-background p-3 text-left hover:bg-muted"
                            onClick={() => {
                              linkToChat("experiment", e.id, e.name);
                              setPhase(e.status === "succeeded" ? "results" : phase);
                            }}
                          >
                            <div className="flex items-start gap-2">
                              <Beaker className="mt-0.5 h-4 w-4" />
                              <div className="min-w-0 flex-1">
                                <div className="flex items-center gap-2">
                                  <div className="truncate text-sm font-semibold">{e.name}</div>
                                  <StatusBadge status={e.status} />
                                </div>
                                <div className="mt-1 text-xs text-muted-foreground">Created {new Date(e.createdAt).toLocaleString()}</div>
                                {!!e.metrics?.length && (
                                  <div className="mt-2 grid grid-cols-3 gap-2">
                                    {e.metrics.slice(0, 3).map((m) => (
                                      <div key={`${e.id}-${m.label}`} className="rounded-xl border bg-muted/40 px-2 py-1">
                                        <div className="text-[10px] text-muted-foreground">{m.label}</div>
                                        <div className="text-xs font-semibold">{m.value}</div>
                                      </div>
                                    ))}
                                  </div>
                                )}
                                {!!e.notes && <div className="mt-2 text-xs text-muted-foreground">{e.notes}</div>}
                              </div>
                              <div className="flex items-center gap-1 text-xs text-muted-foreground">
                                <LinkIcon className="h-3.5 w-3.5" /> Link
                              </div>
                            </div>
                          </button>
                        ))}
                      </div>
                    )}
                  </ScrollArea>
                </TabsContent>

                <TabsContent value="trace" className="mt-3 h-[calc(100%-52px)]">
                  <ScrollArea className="h-full rounded-2xl border p-3">
                    <div className="flex items-center justify-between">
                      <div className="text-sm font-semibold">Trace</div>
                      <Badge variant="secondary" className="text-[10px]">{traceEvents.length}</Badge>
                    </div>
                    <Separator className="my-3" />
                    {traceEvents.length === 0 ? (
                      <div className="text-sm text-muted-foreground">No trace yet. Run an experiment.</div>
                    ) : (
                      <div className="space-y-2">
                        {[...traceEvents].reverse().map((t) => (
                          <button
                            key={t.id}
                            className="w-full rounded-2xl border bg-background p-3 text-left hover:bg-muted"
                            onClick={() => linkToChat("artifact", t.id, `${t.title}`)}
                            title="Click to link this trace step into chat"
                          >
                            <div className="flex items-start gap-2">
                              <TraceDot level={t.level} />
                              <div className="min-w-0 flex-1">
                                <div className="flex items-center justify-between gap-2">
                                  <div className="truncate text-sm font-semibold">{t.title}</div>
                                  <div className="text-xs text-muted-foreground">{fmtTime(t.ts)}</div>
                                </div>
                                {!!t.detail && <div className="mt-1 text-xs text-muted-foreground">{t.detail}</div>}
                              </div>
                              <div className="flex items-center gap-1 text-xs text-muted-foreground">
                                <LinkIcon className="h-3.5 w-3.5" /> Link
                              </div>
                            </div>
                          </button>
                        ))}
                      </div>
                    )}
                  </ScrollArea>
                </TabsContent>
              </Tabs>
            </CardContent>
          </Card>
        </div>

        {/* Footer */}
        <div className="lg:col-span-12">
          <Card className="rounded-2xl">
            <CardContent className="flex flex-col gap-2 p-4 text-sm md:flex-row md:items-center md:justify-between">
              <div className="text-muted-foreground">
                POC notes: This UI simulates runs. Next step: wire to orchestration + tool execution + artifact storage.
              </div>
              <div className="flex flex-wrap items-center gap-2">
                <Badge variant="outline" className="gap-2"><CheckCircle2 className="h-3.5 w-3.5" /> Artifacts-first</Badge>
                <Badge variant="outline" className="gap-2"><ShieldCheck className="h-3.5 w-3.5" /> Audit-ready</Badge>
                <Badge variant="outline" className="gap-2"><Sparkles className="h-3.5 w-3.5" /> Cursor-like iteration</Badge>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}

function MetricCard({ title, value, delta }: { title: string; value: string; delta: string }) {
  const isPos = delta.startsWith("+");
  return (
    <div className="rounded-2xl border bg-background p-3">
      <div className="text-xs text-muted-foreground">{title}</div>
      <div className="mt-1 flex items-baseline gap-2">
        <div className="text-lg font-semibold">{value}</div>
        <Badge variant={isPos ? "secondary" : "outline"} className="text-[10px]">{delta}</Badge>
      </div>
    </div>
  );
}

function StatusBadge({ status }: { status: Experiment["status"] }) {
  if (status === "succeeded") return <Badge className="text-[10px]" variant="secondary">succeeded</Badge>;
  if (status === "running") return <Badge className="text-[10px]" variant="default">running</Badge>;
  if (status === "queued") return <Badge className="text-[10px]" variant="outline">queued</Badge>;
  return <Badge className="text-[10px]" variant="destructive">failed</Badge>;
}

function TraceDot({ level }: { level: TraceEvent["level"] }) {
  const base = "mt-1 h-3 w-3 rounded-full border";
  if (level === "error") return <div className={`${base} bg-destructive`} />;
  if (level === "warn") return <div className={`${base} bg-muted-foreground`} />;
  return <div className={`${base} bg-primary`} />;
}

function LineageNode({ label, sub, onClick }: { label: string; sub: string; onClick?: () => void }) {
  return (
    <button
      className="w-full rounded-2xl border bg-background p-3 text-left hover:bg-muted"
      onClick={onClick}
      type="button"
      title="Click to link into chat"
    >
      <div className="flex items-start gap-2">
        <div className="mt-1 flex h-7 w-7 items-center justify-center rounded-xl bg-muted">
          <LinkIcon className="h-4 w-4" />
        </div>
        <div className="min-w-0 flex-1">
          <div className="truncate text-sm font-semibold">{label}</div>
          <div className="mt-1 text-xs text-muted-foreground">{sub}</div>
        </div>
        <Badge variant="outline" className="text-[10px]">link</Badge>
      </div>
    </button>
  );
}
