import {
  ArrowLeft,
  FileText,
  RefreshCw,
  Radio,
} from "lucide-react"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { LifecycleTimeline } from "@/components/model-risk/LifecycleTimeline"
import { MonitoringPanel } from "@/components/model-risk/MonitoringPanel"
import type { ModelRiskTier } from "@/lib/api"
import { useModelRiskApprovalMutation, useModelRiskDetail } from "@/lib/queries"
import { cn } from "@/lib/utils"
function tierBadgeClass(tier: ModelRiskTier): string {
  switch (tier) {
    case "low":
      return "border-emerald-500/30 bg-emerald-500/10 text-emerald-800 dark:text-emerald-300"
    case "medium":
      return "border-amber-500/35 bg-amber-500/10 text-amber-900 dark:text-amber-200"
    case "high":
      return "border-orange-500/40 bg-orange-500/10 text-orange-900 dark:text-orange-200"
    case "critical":
      return "border-destructive/45 bg-destructive/10 text-destructive"
    default:
      return ""
  }
}

interface ModelRiskProfilePageProps {
  modelName: string
  onBack: () => void
}

export function ModelRiskProfilePage({ modelName, onBack }: ModelRiskProfilePageProps) {
  const detailQuery = useModelRiskDetail(modelName)
  const approvalMut = useModelRiskApprovalMutation()

  const d = detailQuery.data

  if (detailQuery.isPending) {
    return (
      <div className="flex-1 overflow-auto">
        <div className="max-w-3xl mx-auto px-6 sm:px-8 py-10 space-y-6 animate-pulse">
          <div className="h-9 w-40 rounded-md bg-muted" />
          <div className="h-10 w-2/3 max-w-md rounded-md bg-muted" />
          <div className="h-48 rounded-xl bg-muted" />
        </div>
      </div>
    )
  }

  if (detailQuery.isError || !d) {
    return (
      <div className="flex-1 overflow-auto">
        <div className="max-w-3xl mx-auto px-6 sm:px-8 py-10">
          <Button type="button" variant="ghost" size="sm" className="gap-1.5 mb-6" onClick={onBack}>
            <ArrowLeft className="h-4 w-4" />
            Back to registry
          </Button>
          <Card className="border-destructive/30">
            <CardHeader>
              <CardTitle className="text-base">Could not load model risk profile</CardTitle>
              <CardDescription>
                {detailQuery.error?.message ?? "Unknown error"}
              </CardDescription>
            </CardHeader>
            <CardContent>
              <Button type="button" variant="outline" size="sm" onClick={() => void detailQuery.refetch()}>
                <RefreshCw className="h-3.5 w-3.5 mr-1.5" />
                Retry
              </Button>
            </CardContent>
          </Card>
        </div>
      </div>
    )
  }

  return (
    <div className="flex-1 overflow-auto">
      <div className="max-w-3xl mx-auto px-6 sm:px-8 py-8 sm:py-10">
        <Button type="button" variant="ghost" size="sm" className="gap-1.5 -ml-2 mb-6" onClick={onBack}>
          <ArrowLeft className="h-4 w-4" />
          Back to registry
        </Button>

        <header className="space-y-2 border-b border-border/60 pb-6">
          <h1 className="font-headline text-2xl sm:text-3xl font-semibold text-foreground tracking-tight break-all">
            {d.model_name}
          </h1>
          <Badge variant="outline" className={cn("capitalize w-fit", tierBadgeClass(d.risk_tier))}>
            {d.risk_tier} risk
          </Badge>
        </header>

        <Tabs defaultValue="overview" className="mt-8">
          <TabsList className="flex h-auto min-h-9 w-full flex-wrap justify-start gap-0.5 bg-muted/40 p-1">
            <TabsTrigger value="overview" className="text-sm">
              Overview
            </TabsTrigger>
            <TabsTrigger value="lifecycle" className="text-sm">
              Lifecycle
            </TabsTrigger>
            <TabsTrigger value="approvals" className="text-sm">
              Approvals
            </TabsTrigger>
            <TabsTrigger value="monitoring" className="gap-1.5 text-sm">
              <Radio className="h-3.5 w-3.5 opacity-70" />
              Monitoring
            </TabsTrigger>
            <TabsTrigger value="records" className="text-sm">
              Documents & reviews
            </TabsTrigger>
          </TabsList>

          <TabsContent value="overview" className="mt-6 space-y-6">
            {(d.intended_use || d.usage_guidance) && (
              <Card>
                <CardHeader className="pb-3">
                  <CardTitle className="text-base">Purpose & boundaries</CardTitle>
                </CardHeader>
                <CardContent className="space-y-5 text-sm leading-relaxed">
                  {d.intended_use && (
                    <div>
                      <p className="text-overline text-muted-foreground mb-1.5">Intended use</p>
                      <p className="text-muted-foreground">{d.intended_use}</p>
                    </div>
                  )}
                  {d.usage_guidance && (
                    <div>
                      <p className="text-overline text-muted-foreground mb-1.5">How to use</p>
                      <p className="text-muted-foreground">{d.usage_guidance}</p>
                    </div>
                  )}
                </CardContent>
              </Card>
            )}

            {d.performance_kpis && d.performance_kpis.length > 0 && (
              <div className="flex flex-wrap gap-2">
                {d.performance_kpis.map((k) => (
                  <div
                    key={k.label}
                    className="rounded-lg border border-border/70 bg-muted/20 px-3 py-2 min-w-[8rem]"
                  >
                    <p className="text-overline text-muted-foreground truncate">{k.label}</p>
                    <p className="text-lg font-medium tabular-nums text-foreground">
                      {k.value.toLocaleString(undefined, { maximumFractionDigits: 2 })}
                      {k.unit ? <span className="text-sm font-normal text-muted-foreground ml-0.5">{k.unit}</span> : null}
                    </p>
                    {k.window && <p className="text-caption text-muted-foreground mt-0.5">{k.window}</p>}
                  </div>
                ))}
              </div>
            )}

            {d.line_defense_docs && (
              <Card>
                <CardHeader className="pb-3">
                  <CardTitle className="text-base">Line of defense reports</CardTitle>
                </CardHeader>
                <CardContent className="grid gap-3 sm:grid-cols-3">
                  {(
                    [
                      ["Owner", d.line_defense_docs.owner],
                      ["MRM", d.line_defense_docs.mrm],
                      ["Audit", d.line_defense_docs.audit],
                    ] as const
                  ).map(([label, doc]) => (
                    <div key={doc.id} className="rounded-lg border border-border/60 bg-muted/10 px-3 py-2.5">
                      <p className="text-overline text-muted-foreground mb-1">{label}</p>
                      <p className="text-sm font-medium text-foreground leading-snug">{doc.title}</p>
                      {doc.updated_at ? (
                        <p className="text-caption text-muted-foreground mt-1 tabular-nums">
                          {new Date(doc.updated_at).toLocaleDateString()}
                        </p>
                      ) : null}
                    </div>
                  ))}
                </CardContent>
              </Card>
            )}
          </TabsContent>

          <TabsContent value="lifecycle" className="mt-6">
            <Card>
              <CardContent className="pt-6">
                <LifecycleTimeline stages={d.lifecycle_stages} />
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="approvals" className="mt-6 space-y-3">
            {d.approvals.map((a) => (
              <Card key={a.id}>
                <CardHeader className="pb-2">
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <CardTitle className="text-base">{a.stage}</CardTitle>
                    <Badge
                      variant={a.status === "pending" ? "default" : a.status === "approved" ? "secondary" : "destructive"}
                      className="capitalize"
                    >
                      {a.status}
                    </Badge>
                  </div>
                  <CardDescription className="text-xs">
                    {new Date(a.requested_at).toLocaleDateString()}
                    {a.decided_at ? ` → ${new Date(a.decided_at).toLocaleDateString()}` : ""}
                    {a.actor_role ? ` · ${a.actor_role}` : ""}
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-3">
                  {a.comment && <p className="text-sm text-muted-foreground">{a.comment}</p>}
                  {a.status === "pending" && (
                    <div className="flex flex-wrap gap-2">
                      <Button
                        type="button"
                        size="sm"
                        disabled={approvalMut.isPending}
                        onClick={() =>
                          approvalMut.mutate({
                            modelName: d.model_name,
                            body: { approval_id: a.id, decision: "approved" },
                          })
                        }
                      >
                        Approve
                      </Button>
                      <Button
                        type="button"
                        size="sm"
                        variant="outline"
                        disabled={approvalMut.isPending}
                        onClick={() =>
                          approvalMut.mutate({
                            modelName: d.model_name,
                            body: { approval_id: a.id, decision: "rejected" },
                          })
                        }
                      >
                        Reject
                      </Button>
                    </div>
                  )}
                </CardContent>
              </Card>
            ))}
          </TabsContent>

          <TabsContent value="monitoring" className="mt-6 space-y-6">
            <MonitoringPanel monitoring={d.monitoring} />

            {d.monitoring_pipelines && d.monitoring_pipelines.length > 0 && (
              <div>
                <h3 className="text-sm font-medium text-foreground mb-3">Scheduled jobs</h3>
                <div className="space-y-2">
                  {d.monitoring_pipelines.map((p) => (
                    <div
                      key={p.id}
                      className="flex flex-wrap items-center justify-between gap-2 rounded-lg border border-border/50 bg-muted/15 px-3 py-2 text-sm"
                    >
                      <span className="font-medium text-foreground">{p.name}</span>
                      <div className="flex items-center gap-2 text-xs text-muted-foreground">
                        <Badge
                          variant="outline"
                          className={cn(
                            "capitalize",
                            p.status === "ok" && "border-emerald-500/40 text-emerald-800 dark:text-emerald-300",
                            p.status === "warning" && "border-amber-500/40 text-amber-900 dark:text-amber-200",
                            p.status === "failed" && "border-destructive/40 text-destructive",
                          )}
                        >
                          {p.status}
                        </Badge>
                        <span className="tabular-nums hidden sm:inline">{new Date(p.last_run_at).toLocaleString()}</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {d.inference_log && d.inference_log.length > 0 && (
              <div>
                <h3 className="text-sm font-medium text-foreground mb-3">Recent calls</h3>
                <div className="space-y-3">
                  {d.inference_log.map((call) => (
                    <div
                      key={call.id}
                      className="rounded-lg border border-border/50 bg-muted/10 p-3 text-xs font-mono space-y-2"
                    >
                      <div className="flex flex-wrap justify-between gap-2 text-muted-foreground text-[11px] font-sans">
                        <span>{new Date(call.at).toLocaleString()}</span>
                        <span>{call.latency_ms} ms</span>
                      </div>
                      <div className="grid gap-2 sm:grid-cols-2 text-[11px] leading-relaxed">
                        <pre className="overflow-x-auto rounded bg-background/80 p-2 border border-border/40">
                          {JSON.stringify(call.input_preview)}
                        </pre>
                        <pre className="overflow-x-auto rounded bg-background/80 p-2 border border-border/40">
                          {JSON.stringify(call.output_preview)}
                        </pre>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </TabsContent>

          <TabsContent value="records" className="mt-6 space-y-8">
            <section>
              <h3 className="text-sm font-medium text-foreground mb-3">Documents</h3>
              <div className="space-y-2">
                {d.documents.map((doc) => (
                  <div
                    key={doc.id}
                    className="flex items-start gap-3 rounded-lg border border-border/50 bg-muted/10 px-3 py-2.5"
                  >
                    <FileText className="h-4 w-4 mt-0.5 text-muted-foreground shrink-0" />
                    <div className="min-w-0 flex-1">
                      <p className="text-sm font-medium text-foreground">{doc.title}</p>
                      <p className="text-caption text-muted-foreground capitalize">{doc.kind}</p>
                    </div>
                    {doc.updated_at && (
                      <span className="text-caption text-muted-foreground tabular-nums shrink-0">
                        {new Date(doc.updated_at).toLocaleDateString()}
                      </span>
                    )}
                  </div>
                ))}
              </div>
            </section>
            <section>
              <h3 className="text-sm font-medium text-foreground mb-3">Scheduled reviews</h3>
              <div className="space-y-2">
                {d.reviews.map((r) => (
                  <div
                    key={r.id}
                    className="rounded-lg border border-border/50 bg-muted/10 px-3 py-2.5 text-sm"
                  >
                    <p className="font-medium text-foreground">{r.review_type}</p>
                    <p className="text-caption text-muted-foreground mt-0.5">
                      {r.scheduled_for && `Due ${r.scheduled_for}`}
                      {r.owner && ` · ${r.owner}`}
                      {r.completed_at && ` · Done ${new Date(r.completed_at).toLocaleDateString()}`}
                    </p>
                    {r.outcome && <p className="text-muted-foreground mt-2 text-sm">{r.outcome}</p>}
                  </div>
                ))}
              </div>
            </section>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  )
}
