import React from "react"
import { cn } from "@/lib/utils"

/**
 * Section wrapper with a title
 */
export function Section({
  title,
  description,
  children,
}: {
  title: string
  description?: React.ReactNode
  children: React.ReactNode
}) {
  return (
    <div>
      <h3 className="text-xs font-medium text-muted-foreground tracking-tight mb-2.5">{title}</h3>
      {description != null ? (
        <div className="text-xs text-muted-foreground mb-3 leading-relaxed">{description}</div>
      ) : null}
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
  className,
}: {
  label: string
  value: string
  highlight?: boolean
  className?: string
}) {
  return (
    <div
      className={cn(
        "rounded-lg p-3 min-w-0 max-w-full flex flex-col",
        highlight ? "bg-foreground/5" : "bg-muted/30",
        className
      )}
    >
      <div className="text-caption text-muted-foreground mb-0.5 shrink-0">{label}</div>
      <div className="text-sm font-semibold leading-snug break-words [overflow-wrap:anywhere] tabular-nums">
        {value}
      </div>
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
    <div className="flex justify-between items-center gap-3 min-w-0 py-0.5">
      <span className="text-sm text-muted-foreground shrink-0">{label}</span>
      <span
        className={`text-base font-semibold tabular-nums text-right min-w-0 break-words [overflow-wrap:anywhere] ${highlight ? "text-success" : ""}`}
      >
        {value}
      </span>
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
        className={`${mono ? "font-mono text-xs" : small ? "text-sm" : ""} ${highlight ? "text-success font-medium" : ""} break-all`}
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
