import { useMemo } from "react"
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts"
import { cn } from "@/lib/utils"
import { Copy } from "lucide-react"
import type { AnalysisInsight } from "@/types/agent"

const COLORS = ["#6366f1", "#22c55e", "#f97316", "#ec4899", "#06b6d4", "#a855f7"]

function ChartSpecChart({ spec }: { spec: Record<string, unknown> }) {
  const typ = String(spec.type || "bar")
  const data = (spec.data as Array<Record<string, unknown>>) || []
  const title = String(spec.title || "Chart")
  const xKey = String(spec.xKey || spec.x || "x")
  const yKey = String(spec.yKey || spec.y || "y")

  if (!data.length) {
    return <p className="text-xs text-muted-foreground">No chart data.</p>
  }

  if (typ === "histogram") {
    const xk = String(spec.xKey || "x")
    const yk = String(spec.yKey || "count")
    return (
      <ResponsiveContainer width="100%" height={220}>
        <BarChart data={data} margin={{ top: 8, right: 8, left: 0, bottom: 40 }}>
          <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
          <XAxis dataKey={xk} tick={{ fontSize: 10 }} angle={-25} textAnchor="end" height={60} />
          <YAxis tick={{ fontSize: 10 }} />
          <Tooltip />
          <Bar dataKey={yk} fill={COLORS[0]} radius={[4, 4, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    )
  }

  if (typ === "scatter") {
    const cols = Object.keys(data[0] || {})
    const xCol = cols[0]
    const yCol = cols[1]
    return (
      <ResponsiveContainer width="100%" height={240}>
        <ScatterChart margin={{ top: 8, right: 8, bottom: 8, left: 8 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis type="number" dataKey={xCol} name={xCol} tick={{ fontSize: 10 }} />
          <YAxis type="number" dataKey={yCol} name={yCol} tick={{ fontSize: 10 }} />
          <Tooltip cursor={{ strokeDasharray: "3 3" }} />
          <Scatter name={title} data={data} fill={COLORS[0]} />
        </ScatterChart>
      </ResponsiveContainer>
    )
  }

  if (typ === "line") {
    const rows = data.map((row) => ({ ...row }))
    const lineKey = Object.keys(rows[0] || {}).find((k) => k !== "x") || "value"
    return (
      <ResponsiveContainer width="100%" height={220}>
        <LineChart data={rows} margin={{ top: 8, right: 8, left: 0, bottom: 8 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="x" tick={{ fontSize: 10 }} />
          <YAxis tick={{ fontSize: 10 }} />
          <Tooltip />
          <Line type="monotone" dataKey={lineKey} stroke={COLORS[0]} strokeWidth={2} dot={false} />
        </LineChart>
      </ResponsiveContainer>
    )
  }

  if (typ === "grouped_bar" && Array.isArray(spec.bars)) {
    const bars = spec.bars as string[]
    return (
      <ResponsiveContainer width="100%" height={260}>
        <BarChart data={data} margin={{ top: 8, right: 8, left: 0, bottom: 24 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey={xKey} tick={{ fontSize: 10 }} />
          <YAxis tick={{ fontSize: 10 }} />
          <Tooltip />
          <Legend />
          {bars.map((b, i) => (
            <Bar key={b} dataKey={b} stackId="a" fill={COLORS[i % COLORS.length]} radius={[2, 2, 0, 0]} />
          ))}
        </BarChart>
      </ResponsiveContainer>
    )
  }

  // bar (default)
  return (
    <ResponsiveContainer width="100%" height={240}>
      <BarChart data={data} margin={{ top: 8, right: 8, left: 0, bottom: 48 }}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey={xKey} tick={{ fontSize: 10 }} angle={-20} textAnchor="end" height={70} />
        <YAxis tick={{ fontSize: 10 }} />
        <Tooltip />
        <Bar dataKey={yKey} fill={COLORS[0]} radius={[4, 4, 0, 0]} />
      </BarChart>
    </ResponsiveContainer>
  )
}

/** Compact bar chart for group_summary_tool payloads (mean by first split dimension). */
function GroupSummaryChart({ payload }: { payload: Record<string, unknown> }) {
  const groups = payload.groups as Array<Record<string, unknown>> | undefined
  const gbCols = payload.group_by as string[] | undefined
  const metrics = payload.metrics as string[] | undefined
  if (!groups?.length || !gbCols?.length) return null
  const gb = gbCols[0]
  const metric = metrics?.[0]
  if (!metric) return null
  const meanKey = `${metric}_mean`
  const rows = groups
    .map((g) => ({
      label: String(g[gb] ?? ""),
      value: typeof g[meanKey] === "number" ? (g[meanKey] as number) : Number(g[meanKey]),
    }))
    .filter((r) => Number.isFinite(r.value))
  if (!rows.length) return null
  return (
    <ResponsiveContainer width="100%" height={220}>
      <BarChart data={rows} margin={{ top: 8, right: 8, left: 8, bottom: 48 }}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="label" tick={{ fontSize: 10 }} angle={-15} textAnchor="end" height={56} />
        <YAxis tick={{ fontSize: 10 }} tickFormatter={(v) => (v >= 1000 ? `${(v / 1000).toFixed(1)}k` : String(v))} />
        <Tooltip formatter={(v: number) => [v.toLocaleString(undefined, { maximumFractionDigits: 0 }), `mean ${metric}`]} />
        <Bar dataKey="value" fill={COLORS[0]} radius={[4, 4, 0, 0]} name={`mean ${metric}`} />
      </BarChart>
    </ResponsiveContainer>
  )
}

function EdaCompact({ payload }: { payload: Record<string, unknown> }) {
  const shape = payload.shape as { rows?: number; columns?: number } | undefined
  const assoc = payload.target_associations as Array<{ column: string; metric: string; value: number }> | undefined
  const target = payload.target_analysis as { column?: string; task?: string } | undefined
  return (
    <div className="space-y-2 text-xs">
      {shape && (
        <p className="text-muted-foreground">
          {shape.rows?.toLocaleString()} rows × {shape.columns} columns
          {target?.column ? (
            <>
              {" "}
              · target: <span className="font-medium text-foreground">{target.column}</span> ({target.task})
            </>
          ) : null}
        </p>
      )}
      {assoc && assoc.length > 0 ? (
        <div className="rounded-md border border-border/60 overflow-hidden">
          <table className="w-full text-left">
            <thead>
              <tr className="border-b border-border/60 bg-muted/40">
                <th className="px-2 py-1 font-medium">Feature</th>
                <th className="px-2 py-1 font-medium">{assoc[0]?.metric ?? "Metric"}</th>
              </tr>
            </thead>
            <tbody>
              {assoc.slice(0, 8).map((a) => (
                <tr key={a.column} className="border-b border-border/40 last:border-0">
                  <td className="px-2 py-1 font-mono">{a.column}</td>
                  <td className="px-2 py-1">{typeof a.value === "number" ? a.value.toFixed(3) : String(a.value)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <p className="text-muted-foreground">Profile ready — see chat summary for takeaways.</p>
      )}
    </div>
  )
}

function InferentialGroupTable({ payload }: { payload: Record<string, unknown> }) {
  const p = payload.p_value as number | undefined
  const d = payload.cohens_d as number | undefined
  return (
    <div className="space-y-1 text-xs font-mono">
      <p>
        <span className="text-muted-foreground">test:</span> {String(payload.test ?? "—")}{" "}
        <span className="text-muted-foreground">p:</span>{" "}
        {p != null ? p.toExponential(3) : "—"}{" "}
        {d != null ? (
          <>
            <span className="text-muted-foreground">Cohen&apos;s d:</span> {d.toFixed(3)}
          </>
        ) : null}
      </p>
    </div>
  )
}

function InferentialCategoricalTable({ payload }: { payload: Record<string, unknown> }) {
  const chi2 = payload.chi2 as number | undefined
  const p = payload.p_value as number | undefined
  const v = payload.cramers_v as number | undefined
  return (
    <div className="rounded-md border border-border/60 overflow-hidden text-xs">
      <table className="w-full text-left">
        <tbody>
          <tr className="border-b border-border/40">
            <td className="px-2 py-1 text-muted-foreground">χ²</td>
            <td className="px-2 py-1">{chi2 != null ? chi2.toFixed(4) : "—"}</td>
          </tr>
          <tr className="border-b border-border/40">
            <td className="px-2 py-1 text-muted-foreground">p-value</td>
            <td className="px-2 py-1">{p != null ? p.toExponential(3) : "—"}</td>
          </tr>
          <tr>
            <td className="px-2 py-1 text-muted-foreground">Cramér&apos;s V</td>
            <td className="px-2 py-1">{v != null ? v.toFixed(4) : "—"}</td>
          </tr>
        </tbody>
      </table>
    </div>
  )
}

function RegressionCoefTable({ payload }: { payload: Record<string, unknown> }) {
  const coefs = (payload.coefficients as Array<Record<string, unknown>>) || []
  const rows = coefs.slice(0, 12)
  if (!rows.length) return <p className="text-xs text-muted-foreground">No coefficients.</p>
  return (
    <div className="rounded-md border border-border/60 overflow-hidden text-xs max-h-48 overflow-y-auto">
      <table className="w-full text-left">
        <thead>
          <tr className="border-b border-border/60 bg-muted/40">
            <th className="px-2 py-1">Term</th>
            <th className="px-2 py-1">p</th>
            <th className="px-2 py-1">coef</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={String(row.name)} className="border-b border-border/40 last:border-0">
              <td className="px-2 py-0.5 font-mono max-w-[120px] truncate">{String(row.name)}</td>
              <td className="px-2 py-0.5">
                {typeof row.p_value === "number" ? row.p_value.toExponential(2) : "—"}
              </td>
              <td className="px-2 py-0.5">{typeof row.coef === "number" ? row.coef.toFixed(4) : "—"}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function CorrelationBars({ payload }: { payload: Record<string, unknown> }) {
  const pairs = (payload.top_pairs as Array<{ col1: string; col2: string; correlation: number }>) || []
  const rows = pairs.slice(0, 12).map((p) => ({
    name: `${p.col1}↔${p.col2}`.slice(0, 28),
    r: Math.abs(p.correlation ?? 0),
    raw: p.correlation ?? 0,
  }))
  if (!rows.length) return null
  return (
    <ResponsiveContainer width="100%" height={Math.min(400, 40 + rows.length * 28)}>
      <BarChart layout="vertical" data={rows} margin={{ left: 8, right: 16, top: 8, bottom: 8 }}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis type="number" domain={[0, 1]} tick={{ fontSize: 10 }} />
        <YAxis type="category" dataKey="name" width={120} tick={{ fontSize: 9 }} />
        <Tooltip formatter={(value) => [`${Number(value).toFixed(3)}`, "|r|"]} />
        <Bar dataKey="r" radius={[0, 4, 4, 0]}>
          {rows.map((_, i) => (
            <Cell key={i} fill={COLORS[i % COLORS.length]} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  )
}

export function AnalysisCard({ insight }: { insight: AnalysisInsight }) {
  const spec = insight.chartSpec || (insight.payload as Record<string, unknown>)?.spec as Record<string, unknown> | undefined

  const title = useMemo(() => {
    if (insight.summary) return insight.summary
    return `${insight.tool} · ${insight.kind}`
  }, [insight])

  const copyPayload = () => {
    void navigator.clipboard.writeText(JSON.stringify(insight.payload ?? {}, null, 2))
  }

  const hasVisual =
    (insight.kind === "correlation" && insight.payload) ||
    (spec && Object.keys(spec).length > 0) ||
    insight.kind === "trend" ||
    insight.kind === "concentration" ||
    insight.kind === "group_summary" ||
    insight.kind === "eda" ||
    insight.kind === "chart" ||
    insight.kind === "inferential_group" ||
    insight.kind === "inferential_categorical" ||
    insight.kind === "inferential_regression"

  return (
    <div
      className={cn(
        "rounded-xl border border-border/80 bg-card/80 shadow-sm overflow-hidden",
        "backdrop-blur-sm",
      )}
    >
      <div className="flex items-center justify-between gap-2 px-3 py-2 border-b border-border/60 bg-muted/30">
        <span className="text-xs font-semibold text-foreground truncate">{title}</span>
        <button
          type="button"
          className="p-1 rounded-md hover:bg-muted text-muted-foreground shrink-0"
          title="Copy raw analysis JSON"
          onClick={copyPayload}
        >
          <Copy className="h-3.5 w-3.5" />
        </button>
      </div>
      <div className="p-3 space-y-2">
        {insight.kind === "group_summary" && insight.payload ? (
          <GroupSummaryChart payload={insight.payload as Record<string, unknown>} />
        ) : null}
        {insight.kind === "eda" && insight.payload ? (
          <EdaCompact payload={insight.payload as Record<string, unknown>} />
        ) : null}
        {insight.kind === "correlation" && insight.payload ? (
          <CorrelationBars payload={insight.payload as Record<string, unknown>} />
        ) : null}
        {insight.kind === "inferential_group" && insight.payload ? (
          <InferentialGroupTable payload={insight.payload as Record<string, unknown>} />
        ) : null}
        {insight.kind === "inferential_categorical" && insight.payload ? (
          <InferentialCategoricalTable payload={insight.payload as Record<string, unknown>} />
        ) : null}
        {insight.kind === "inferential_regression" && insight.payload ? (
          <RegressionCoefTable payload={insight.payload as Record<string, unknown>} />
        ) : null}
        {spec && Object.keys(spec).length > 0 ? <ChartSpecChart spec={spec} /> : null}
        {insight.kind === "trend" && Array.isArray((insight.payload as Record<string, unknown> & { periods?: unknown }).periods) ? (
          <ResponsiveContainer width="100%" height={220}>
            <LineChart
              data={(insight.payload as { periods: Array<{ period: string; value: number }> }).periods.map((p) => ({
                t: p.period,
                v: p.value,
              }))}
              margin={{ top: 8, right: 8, left: 0, bottom: 8 }}
            >
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="t" tick={{ fontSize: 9 }} />
              <YAxis tick={{ fontSize: 10 }} />
              <Tooltip />
              <Line type="monotone" dataKey="v" stroke={COLORS[2]} strokeWidth={2} dot />
            </LineChart>
          </ResponsiveContainer>
        ) : null}
        {insight.kind === "concentration" &&
        Array.isArray((insight.payload as { lorenz_curve?: unknown }).lorenz_curve) ? (
          <ResponsiveContainer width="100%" height={200}>
            <LineChart
              data={(insight.payload as { lorenz_curve: Array<{ pct_of_entities: number; pct_of_value: number }> }).lorenz_curve.map((pt) => ({
                x: pt.pct_of_entities,
                y: pt.pct_of_value,
              }))}
              margin={{ top: 8, right: 8, left: 0, bottom: 8 }}
            >
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis tick={{ fontSize: 10 }} />
              <YAxis tick={{ fontSize: 10 }} />
              <Tooltip />
              <Line type="monotone" dataKey="y" stroke={COLORS[3]} strokeWidth={2} dot={false} name="% value" />
            </LineChart>
          </ResponsiveContainer>
        ) : null}
        {!hasVisual && insight.payload != null && (
          <p className="text-xs text-muted-foreground">Analysis data attached — use copy button for export.</p>
        )}
      </div>
    </div>
  )
}
