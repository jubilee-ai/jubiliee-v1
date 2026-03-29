import { clsx, type ClassValue } from "clsx"
import { twMerge } from "tailwind-merge"

/**
 * Merge Tailwind CSS classes with clsx.
 * Standard shadcn/ui utility for className composition.
 */
export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

/**
 * Format a number with optional decimal places.
 */
export function formatNumber(value: number | null | undefined, decimals = 4): string {
  if (value == null || Number.isNaN(value)) return "N/A"
  return Number(value).toFixed(decimals)
}

/**
 * Format a value as a percentage (0–1 scale to 0–100%).
 */
export function formatPercent(value: number | null | undefined, decimals = 2): string {
  if (value == null || Number.isNaN(value)) return "N/A"
  return `${(Number(value) * 100).toFixed(decimals)}%`
}

/**
 * Generate a unique id with optional prefix.
 */
export function uid(prefix = "id"): string {
  return `${prefix}-${Math.random().toString(36).slice(2, 11)}`
}

/** Short sidebar title from the user’s first chat line (auto-created experiments). */
export function suggestExperimentTitleFromUserMessage(text: string): string | undefined {
  const t = text.replace(/\s+/g, " ").trim()
  if (!t) return undefined
  const max = 52
  return t.length > max ? `${t.slice(0, max - 1).trimEnd()}…` : t
}

/** Title when the session starts by linking datasets before any message. */
export function suggestExperimentTitleFromLinkedDatasets(ids: string[]): string | undefined {
  if (!ids.length) return undefined
  const first = ids[0]?.split("/").pop() || "Dataset"
  const label = ids.length === 1 ? first : `${first} +${ids.length - 1} more`
  return label.length > 52 ? `${label.slice(0, 51)}…` : label
}
