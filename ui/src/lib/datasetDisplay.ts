import type { Dataset as ApiDataset } from "@/lib/api"

const UUID_RE =
  /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i

export function isUuidLike(s: string): boolean {
  return UUID_RE.test(s.trim())
}

function buildDatasetNameLookup(datasets: ApiDataset[]): Map<string, string> {
  const m = new Map<string, string>()
  for (const d of datasets) {
    const label = d.name?.trim()
    if (!label) continue
    if (d.id) m.set(String(d.id), label)
    if (d.file) m.set(d.file, label)
    if (d.storage_key) m.set(d.storage_key, label)
    m.set(label, label)
  }
  return m
}

function shortIdHint(s: string): string {
  const t = s.trim()
  if (isUuidLike(t)) return t.replace(/-/g, "").slice(0, 8)
  return t.slice(0, 12)
}

/**
 * Turn stored plan labels/refs into names suitable for the UI (plan card, task view, report).
 * Uses the dataset catalog when refs match id, file, or storage_key.
 */
export function resolveDatasetDisplayNames(
  labels: string[],
  refs: string[] | undefined | null,
  datasets: ApiDataset[],
): string[] {
  const lookup = buildDatasetNameLookup(datasets)
  const n = Math.max(labels.length, refs?.length ?? 0)
  if (n === 0) return []

  const out: string[] = []
  const uuidPlaceholders: { outIndex: number; raw: string }[] = []

  for (let i = 0; i < n; i++) {
    const label = labels[i] ?? ""
    const ref = refs?.[i] ?? ""
    const keys = [ref, label].filter((k) => Boolean(k && String(k).trim())) as string[]
    let name: string | undefined
    for (const k of keys) {
      const hit = lookup.get(k)
      if (hit) {
        name = hit
        break
      }
    }
    if (name) {
      out.push(name)
      continue
    }
    const raw = (label || ref).trim()
    if (!raw) {
      out.push("Dataset")
      continue
    }
    if (isUuidLike(raw)) {
      uuidPlaceholders.push({ outIndex: out.length, raw })
      out.push("")
    } else {
      out.push(raw.split("/").pop() ?? raw)
    }
  }

  const uLen = uuidPlaceholders.length
  for (const slot of uuidPlaceholders) {
    out[slot.outIndex] =
      uLen === 1
        ? "Registered dataset"
        : `Registered dataset (${shortIdHint(slot.raw)})`
  }

  return out
}
