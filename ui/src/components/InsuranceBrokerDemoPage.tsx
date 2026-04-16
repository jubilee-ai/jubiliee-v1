import { useMemo, useState } from "react"
import {
  ChevronRight,
  PhoneOff,
  Sparkles,
  TrendingDown,
  UserRound,
} from "lucide-react"
import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"
import { Separator } from "@/components/ui/separator"
import { cn } from "@/lib/utils"
import {
  BROKER_PERSONAS,
  DEMO_SCENARIO_TITLE,
  propensityLabel,
  tierLabel,
  type BrokerPersona,
  type ClaimRiskTier,
  type IntakeField,
} from "@/lib/insuranceBrokerDemoData"

const TOP_N = 5

function tierStyles(tier: ClaimRiskTier): { badge: string; bar: string } {
  switch (tier) {
    case "favorable":
      return {
        badge: "border-emerald-500/35 bg-emerald-500/10 text-emerald-900 dark:text-emerald-300",
        bar: "bg-emerald-500/85",
      }
    case "standard":
      return {
        badge: "border-amber-500/40 bg-amber-500/10 text-amber-950 dark:text-amber-200",
        bar: "bg-amber-500/85",
      }
    case "elevated":
      return {
        badge: "border-orange-500/45 bg-orange-500/10 text-orange-950 dark:text-orange-200",
        bar: "bg-orange-500/85",
      }
  }
}

function initialsFromLabel(name: string): string {
  const parts = name.trim().split(/\s+/).filter(Boolean)
  if (parts.length === 0) return "?"
  if (parts.length === 1) return parts[0]!.slice(0, 2).toUpperCase()
  return (parts[0]![0]! + parts[parts.length - 1]![0]!).toUpperCase()
}

function intakeValue(fields: IntakeField[], label: string): string | undefined {
  return fields.find((f) => f.label === label)?.value
}

function profileSnapshot(persona: BrokerPersona): { label: string; value: string }[] {
  const keys = ["Territory", "Dwelling", "Loss history"] as const
  const out: { label: string; value: string }[] = []
  for (const label of keys) {
    const value = intakeValue(persona.intakeFields, label)
    if (value) out.push({ label, value })
  }
  return out
}

export function InsuranceBrokerDemoPage() {
  const [selectedId] = useState(BROKER_PERSONAS[0]!.id)

  const selected = useMemo(
    () => BROKER_PERSONAS.find((p) => p.id === selectedId) ?? BROKER_PERSONAS[0]!,
    [selectedId],
  )

  const topPartners = selected.rankedPartners.slice(0, TOP_N)
  const ts = tierStyles(selected.tier)
  const prospectLine = intakeValue(selected.intakeFields, "Prospect") ?? selected.label
  const snapshot = profileSnapshot(selected)

  return (
    <div className="min-h-full bg-background">
      <div className="pointer-events-none fixed inset-0 -z-10 bg-[radial-gradient(ellipse_80%_45%_at_50%_-15%,rgba(18,86,210,0.1),transparent),radial-gradient(ellipse_50%_35%_at_100%_20%,rgba(52,211,153,0.05),transparent)]" />

      <div className="mx-auto max-w-[1200px] px-4 py-8 sm:px-6 lg:px-8">
        <header className="mb-8 border-b border-border/50 pb-8">
          <h1 className="font-headline text-3xl font-semibold tracking-tight text-foreground sm:text-[2rem]">
            {DEMO_SCENARIO_TITLE}
          </h1>
          {/* <p className="mt-2 max-w-xl text-sm leading-relaxed text-muted-foreground">
            Choose someone, review their profile and intake, then see risk and carrier placement.
          </p> */}
        </header>

        {/* 1 — Compact applicant picker */}
        <section
          className="mb-8 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between sm:gap-6"
          aria-label="Select applicant"
        >
          <div className="flex min-w-0 items-center gap-2 text-sm font-medium text-muted-foreground">
            <span className="shrink-0 text-overline text-[10px] font-bold uppercase tracking-[0.2em] text-foreground/70">
              Applicant
            </span>
            <ChevronRight className="h-4 w-4 shrink-0 opacity-40" aria-hidden />
            <span className="truncate text-foreground">{selected.label}</span>
          </div>
        </section>

        <div
          key={selected.id}
          className="grid gap-8 lg:grid-cols-[minmax(0,340px)_minmax(0,1fr)] lg:items-start lg:gap-10"
        >
          {/* Left: person — profile + intake */}
          <div className="flex min-w-0 flex-col gap-6">
            <Card className="overflow-hidden border-border/70 bg-card shadow-sm">
              <div className="border-b border-border/50 bg-muted/20 px-4 py-3">
                <div className="flex items-center gap-2 text-muted-foreground">
                  <UserRound className="h-4 w-4" aria-hidden />
                  <span className="text-overline text-[10px] font-bold uppercase tracking-[0.2em]">
                    Profile
                  </span>
                </div>
              </div>
              <CardContent className="space-y-5 p-5 pt-5">
                <div className="flex gap-4">
                  <div
                    className="flex h-16 w-16 shrink-0 items-center justify-center rounded-2xl border border-border/50 bg-gradient-to-br from-primary/20 to-primary/5 font-headline text-lg font-semibold text-primary"
                    aria-hidden
                  >
                    {initialsFromLabel(selected.label)}
                  </div>
                  <div className="min-w-0 flex-1">
                    <h2 className="font-headline text-xl font-semibold leading-tight tracking-tight text-foreground">
                      {selected.label}
                    </h2>
                    <p className="mt-1 text-sm leading-snug text-muted-foreground">{selected.tagline}</p>
                  </div>
                </div>
                <p className="text-[15px] leading-relaxed text-foreground/95">{prospectLine}</p>
                {snapshot.length > 0 ? (
                  <>
                    <Separator />
                    <dl className="grid gap-4 sm:grid-cols-1">
                      {snapshot.map((row) => (
                        <div key={row.label}>
                          <dt className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
                            {row.label}
                          </dt>
                          <dd className="mt-1 text-sm leading-snug text-foreground">{row.value}</dd>
                        </div>
                      ))}
                    </dl>
                  </>
                ) : null}
              </CardContent>
            </Card>

            <Card className="border-border/70 bg-card shadow-sm">
              <CardHeader className="pb-2 pt-4">
                <div className="flex items-center gap-2 text-muted-foreground">
                  <PhoneOff className="h-4 w-4" aria-hidden />
                  <span className="text-overline text-[10px] font-bold uppercase tracking-[0.2em]">
                    Call intake
                  </span>
                </div>
                <CardTitle className="font-headline text-base">Operator fields</CardTitle>
                <CardDescription className="text-xs">
                  CRM snapshot used for this placement run.
                </CardDescription>
              </CardHeader>
              <CardContent className="max-h-[min(48vh,440px)] space-y-2 overflow-y-auto pb-4 pr-1 pt-0">
                {selected.intakeFields.map((row) => (
                  <div
                    key={row.label}
                    className="rounded-lg border border-border/50 bg-muted/20 px-3 py-2 text-left"
                  >
                    <p className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
                      {row.label}
                    </p>
                    <p className="mt-1 text-sm leading-snug text-foreground">{row.value}</p>
                  </div>
                ))}
              </CardContent>
            </Card>
          </div>

          {/* Right: placement — risk then carriers */}
          <div className="flex min-w-0 flex-col gap-6">
            <div className="flex items-center gap-2 text-muted-foreground">
              <span className="text-overline text-[10px] font-bold uppercase tracking-[0.2em]">
                Placement
              </span>
            </div>

            <Card className="border-border/70 bg-card shadow-sm">
              <CardHeader className="pb-3">
                <div className="flex items-center gap-2">
                  <TrendingDown className="h-4 w-4 text-emerald-600 dark:text-emerald-400" aria-hidden />
                  <div>
                    <p className="text-overline text-[10px] font-bold uppercase tracking-[0.18em] text-muted-foreground">
                      Risk
                    </p>
                    <CardTitle className="font-headline mt-0.5 text-lg">Claim propensity</CardTitle>
                  </div>
                </div>
                <CardDescription className="text-xs">
                  Versus similar risks in your book (illustrative).
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-5">
                <div className="flex items-end gap-2">
                  <span className="font-headline text-5xl font-semibold tabular-nums tracking-tight text-foreground">
                    {propensityLabel(selected.propensity)}
                  </span>
                  <span className="pb-1.5 text-sm text-muted-foreground">vs. peers</span>
                </div>
                <div className="space-y-2">
                  <div className="h-2.5 w-full overflow-hidden rounded-full bg-muted">
                    <div
                      className={cn("h-full rounded-full transition-all", ts.bar)}
                      style={{ width: `${Math.min(100, selected.propensity * 100 * 2.2)}%` }}
                    />
                  </div>
                  <p className="text-xs leading-relaxed text-muted-foreground">
                    Lower usually means fewer expected claims
                  </p>
                </div>
                <Badge variant="outline" className={cn("font-normal", ts.badge)}>
                  {tierLabel(selected.tier)}
                </Badge>
              </CardContent>
            </Card>

            <Card className="border-border/70 bg-card shadow-sm">
              <CardHeader className="pb-3">
                <div className="flex items-center gap-2">
                  <Sparkles className="h-4 w-4 text-primary" aria-hidden />
                  <div>
                    <p className="text-overline text-[10px] font-bold uppercase tracking-[0.18em] text-muted-foreground">
                      Carriers
                    </p>
                    <CardTitle className="font-headline mt-0.5 text-lg">Partner match</CardTitle>
                  </div>
                </div>
                <CardDescription className="text-xs">
                  Top {TOP_N} of {selected.rankedPartners.length} appointed carriers for this stack.
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-3">
                {topPartners.map((p) => (
                  <div
                    key={`${selected.id}-${p.rank}-${p.name}`}
                    className="rounded-xl border border-border/50 bg-muted/15 px-3 py-3 transition-colors hover:bg-muted/30"
                  >
                    <div className="flex items-start justify-between gap-2">
                      <div className="min-w-0">
                        <div className="flex items-center gap-2">
                          <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-md bg-primary/10 font-mono text-[11px] font-bold text-primary">
                            {p.rank}
                          </span>
                          <span className="truncate font-medium text-foreground">{p.name}</span>
                        </div>
                        <p className="mt-1 pl-8 text-[11px] text-muted-foreground">{p.focus}</p>
                        <p className="mt-1.5 pl-8 text-xs leading-snug text-foreground/90">{p.rationale}</p>
                      </div>
                      <span className="shrink-0 font-mono text-sm font-semibold tabular-nums text-primary">
                        {p.matchScore}
                      </span>
                    </div>
                    <Progress value={p.matchScore} className="mt-3 h-1.5 bg-muted/80" />
                  </div>
                ))}
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </div>
  )
}
