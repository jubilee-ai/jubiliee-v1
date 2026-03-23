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
