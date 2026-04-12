import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import type { ModelRiskMonitoring, ModelRiskMonitoringStatus } from "@/lib/api"
import { cn } from "@/lib/utils"

function statusBadgeClass(status: ModelRiskMonitoringStatus): string {
  switch (status) {
    case "healthy":
      return "border-emerald-500/35 bg-emerald-500/10 text-emerald-700 dark:text-emerald-400"
    case "watch":
      return "border-amber-500/40 bg-amber-500/10 text-amber-800 dark:text-amber-300"
    case "breach":
      return "border-destructive/40 bg-destructive/10 text-destructive"
    default:
      return "border-border bg-muted text-muted-foreground"
  }
}

interface MonitoringPanelProps {
  monitoring: ModelRiskMonitoring
  className?: string
}

export function MonitoringPanel({ monitoring, className }: MonitoringPanelProps) {
  return (
    <div className={cn("space-y-4", className)}>
      <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
        <Badge variant="outline" className={cn("capitalize", statusBadgeClass(monitoring.status))}>
          {monitoring.status}
        </Badge>
        <span className="text-caption text-muted-foreground tabular-nums">
          {new Date(monitoring.as_of).toLocaleDateString()}
        </span>
      </div>
      {monitoring.narrative && (
        <p className="text-sm text-muted-foreground leading-relaxed">{monitoring.narrative}</p>
      )}

      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-base">Drift & performance</CardTitle>
        </CardHeader>
        <CardContent className="space-y-6 pt-0">
          <div className="overflow-x-auto">
            <p className="text-overline text-muted-foreground mb-2">PSI / CSI</p>
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-border text-left text-overline text-muted-foreground">
                  <th className="py-1.5 pr-4 font-medium">Feature</th>
                  <th className="py-1.5 pr-4 font-medium tabular-nums">PSI</th>
                  <th className="py-1.5 font-medium tabular-nums">CSI</th>
                </tr>
              </thead>
              <tbody>
                {monitoring.psi_csi.map((row) => (
                  <tr key={row.feature} className="border-b border-border/60 last:border-0">
                    <td className="py-1.5 pr-4 font-mono text-xs text-foreground/90">{row.feature}</td>
                    <td className="py-1.5 pr-4 tabular-nums text-muted-foreground">{row.psi.toFixed(4)}</td>
                    <td className="py-1.5 tabular-nums text-muted-foreground">
                      {row.csi != null ? row.csi.toFixed(4) : "—"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="overflow-x-auto">
            <p className="text-overline text-muted-foreground mb-2">Backtest</p>
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-border text-left text-overline text-muted-foreground">
                  <th className="py-1.5 pr-3 font-medium">Period</th>
                  <th className="py-1.5 pr-3 font-medium">Metric</th>
                  <th className="py-1.5 pr-3 font-medium tabular-nums">Value</th>
                  <th className="py-1.5 pr-3 font-medium tabular-nums">Bench</th>
                  <th className="py-1.5 font-medium">Pass</th>
                </tr>
              </thead>
              <tbody>
                {monitoring.backtest.map((row, idx) => (
                  <tr key={`${row.period}-${row.metric}-${idx}`} className="border-b border-border/60 last:border-0">
                    <td className="py-1.5 pr-3 text-muted-foreground">{row.period}</td>
                    <td className="py-1.5 pr-3">{row.metric}</td>
                    <td className="py-1.5 pr-3 tabular-nums">{row.value.toFixed(4)}</td>
                    <td className="py-1.5 pr-3 tabular-nums text-muted-foreground">
                      {row.benchmark != null ? row.benchmark.toFixed(4) : "—"}
                    </td>
                    <td className="py-1.5">
                      <Badge variant={row.pass ? "secondary" : "destructive"} className="text-overline">
                        {row.pass ? "Pass" : "Fail"}
                      </Badge>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
