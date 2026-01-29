import React from "react"

/**
 * Section wrapper with a title
 */
export function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div>
      <h3 className="text-sm font-medium mb-4">{title}</h3>
      {children}
    </div>
  )
}

/**
 * Metric display box for hero metrics
 */
export function MetricBox({
  label,
  value,
  highlight,
}: {
  label: string
  value: string
  highlight?: boolean
}) {
  return (
    <div className={`rounded-xl p-4 ${highlight ? "bg-foreground/5" : "bg-muted/30"}`}>
      <div className="text-xs text-muted-foreground mb-1">{label}</div>
      <div className="text-xl font-semibold">{value}</div>
    </div>
  )
}

/**
 * Metric row with label and value
 */
export function MetricRow({
  label,
  value,
  highlight,
}: {
  label: string
  value: string
  highlight?: boolean
}) {
  return (
    <div className="flex justify-between items-center">
      <span className="text-muted-foreground">{label}</span>
      <span className={`text-xl font-semibold ${highlight ? "text-green-600" : ""}`}>{value}</span>
    </div>
  )
}

/**
 * Info row for key-value display
 */
export function InfoRow({ label, value }: { label: string; value?: string | null }) {
  return (
    <div className="flex justify-between items-start gap-4 py-2 border-b border-border/50 last:border-0">
      <span className="text-muted-foreground shrink-0">{label}</span>
      <span className="text-right">{value || "N/A"}</span>
    </div>
  )
}

/**
 * Info box for displaying labeled values
 */
export function InfoBox({
  label,
  value,
  mono,
  highlight,
  small,
}: {
  label: string
  value?: string | null
  mono?: boolean
  highlight?: boolean
  small?: boolean
}) {
  return (
    <div className={`bg-muted/50 rounded-lg ${small ? "px-2 py-1.5" : "px-3 py-2"}`}>
      <div className={`text-muted-foreground ${small ? "text-xs" : "text-sm"} mb-0.5`}>{label}</div>
      <div
        className={`${mono ? "font-mono text-xs" : small ? "text-sm" : ""} ${highlight ? "text-green-600 font-medium" : ""} break-all`}
      >
        {value || "N/A"}
      </div>
    </div>
  )
}

/**
 * Pipeline row for data flow visualization
 */
export function PipelineRow({
  label,
  value,
  last,
}: {
  label: string
  value?: string | null
  last?: boolean
}) {
  return (
    <div
      className={`flex justify-between items-center px-4 py-3 ${!last ? "border-b border-border/30" : ""}`}
    >
      <span className="text-muted-foreground">{label}</span>
      <code className="text-xs font-mono truncate max-w-[60%]">{value || "N/A"}</code>
    </div>
  )
}
