/**
 * Static demo data for the insurance broker pipeline scenario (investor / UX demo).
 * Replace scoring with your own agent or API — UI only consumes ranked results.
 */

export type ClaimRiskTier = "favorable" | "standard" | "elevated"

export interface IntakeField {
  label: string
  value: string
}

export interface PartnerMatch {
  rank: number
  name: string
  focus: string
  matchScore: number
  rationale: string
}

export interface BrokerPersona {
  id: string
  label: string
  tagline: string
  intakeFields: IntakeField[]
  propensity: number
  tier: ClaimRiskTier
  rankedPartners: PartnerMatch[]
}

export const DEMO_SCENARIO_TITLE = "Post-call placement"

export function propensityLabel(p: number): string {
  return `${(p * 100).toFixed(1)}%`
}

export function tierLabel(tier: ClaimRiskTier): string {
  switch (tier) {
    case "favorable":
      return "Favorable"
    case "standard":
      return "Standard"
    case "elevated":
      return "Elevated · review before bind"
    default:
      return tier
  }
}

const CARRIER_POOL: Omit<PartnerMatch, "rank" | "matchScore">[] = [
  { name: "NorthPeak Mutual", focus: "HO + auto bundles, Mountain West", rationale: "Strong CO appetite · multi-line discount" },
  { name: "Harborline Specialty", focus: "High-value dwelling, coastal & inland", rationale: "Dwelling limit in sweet spot" },
  { name: "Crestwood National", focus: "Personal lines, motorcycle-friendly", rationale: "Moto endorsement in guidelines" },
  { name: "SummitSure Partners", focus: "First-time buyers, digital bind", rationale: "Fast-track underwriting path" },
  { name: "BlueCanopy General", focus: "Umbrella follow, excess liability", rationale: "Umbrella appetite matches stack" },
  { name: "Redwood Assurance", focus: "Preferred risks, EFT discount", rationale: "Pay plan + credit tier aligned" },
  { name: "Granite Gate Insurance", focus: "Regional multi-line", rationale: "Competitive on bundled HO" },
  { name: "Silverline Underwriters", focus: "Personal umbrella focus", rationale: "Umbrella pricing competitive" },
  { name: "PrairieStar Carrier", focus: "Plains & Rockies admitted", rationale: "Territory match" },
  { name: "Evergreen Risk Co.", focus: "HO-3 standard dwelling", rationale: "Dwelling band fit" },
  { name: "Atlas Frontier P&C", focus: "Multi-vehicle households", rationale: "Fleet discount structure" },
  { name: "Lighthouse Mutual", focus: "Coastal & mountain states", rationale: "Regional guidelines" },
  { name: "Ironwood National", focus: "Standard personal lines", rationale: "General placement" },
  { name: "Clearwater Specialty", focus: "High-net-worth adjacent", rationale: "Slight overlap on limits" },
  { name: "Beacon Hill Insurance", focus: "Northeast & Midwest", rationale: "Secondary fit" },
  { name: "Riverstone General", focus: "Auto-heavy books", rationale: "Auto competitive; HO ok" },
  { name: "Mesa Ridge Mutual", focus: "Desert SW", rationale: "Adjacent region" },
  { name: "Pinewood Assurance", focus: "Wood-frame dwelling focus", rationale: "Construction class ok" },
  { name: "Trailhead P&C", focus: "Outdoor recreation risks", rationale: "Moto ok; HO secondary" },
  { name: "Copperfield Underwriters", focus: "Mid-market HO", rationale: "Standard tier" },
  { name: "Sagebrush National", focus: "Rural & exurban", rationale: "Metro not primary" },
  { name: "Tundra Mutual", focus: "Northern climates", rationale: "Region stretch" },
  { name: "Osprey Specialty", focus: "Coastal wind", rationale: "Not primary for CO" },
  { name: "Quarrystone General", focus: "Commercial-heavy", rationale: "Personal lines thinner" },
  { name: "Willowbrook Carrier", focus: "Senior homeowners", rationale: "Age band mismatch" },
  { name: "Cobblestone Assurance", focus: "Urban condos", rationale: "Product: condo focus" },
  { name: "Highland Ridge Insurance", focus: "Farm & ranch", rationale: "Not target class" },
  { name: "Mariner's Gate P&C", focus: "Watercraft", rationale: "Niche marine" },
  { name: "Sunvault Mutual", focus: "Solar home programs", rationale: "Program not triggered" },
  { name: "Keystone National", focus: "Mid-Atlantic", rationale: "Territory stretch" },
  { name: "Foxglove Specialty", focus: "Landlord DP3", rationale: "Owner-occ not primary" },
  { name: "Birchwood General", focus: "Credit-sensitive", rationale: "Pricing overlap only" },
  { name: "Thundercap Underwriters", focus: "Catastrophe exposed", rationale: "Wildfire tier friction" },
  { name: "Goldfield Assurance", focus: "Luxury autos", rationale: "Auto-led; HO weaker" },
  { name: "Starling Mutual", focus: "Renters-heavy", rationale: "HO not core" },
  { name: "Coldstream National", focus: "Commercial package", rationale: "Personal lines limited" },
  { name: "Driftwood P&C", focus: "Vacation homes", rationale: "Primary res not focus" },
  { name: "Nighthawk Specialty", focus: "High-crime urban", rationale: "Risk zone mismatch" },
  { name: "Daybreak Insurance Group", focus: "Budget minimum limits", rationale: "Below target dwelling" },
]

function identityRank(): PartnerMatch[] {
  return CARRIER_POOL.map((row, i) => ({
    ...row,
    rank: i + 1,
    matchScore: Math.max(59, 97 - i*5),
  }))
}

function mulberry32(seed: number): () => number {
  let a = seed >>> 0
  return () => {
    a = (a + 0x6d2b79f5) | 0
    let t = Math.imul(a ^ (a >>> 15), a | 1)
    t = (t + Math.imul(t ^ (t >>> 7), t | 61)) | 0
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

function rankPartnersForSeed(seed: number): PartnerMatch[] {
  const rand = mulberry32(seed)
  const arr = CARRIER_POOL.map((p) => ({ ...p }))
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1))
    const tmp = arr[i]!
    arr[i] = arr[j]!
    arr[j] = tmp
  }
  return arr.map((row, i) => ({
    ...row,
    rank: i + 1,
    matchScore: Math.max(59, 97 - i),
  }))
}

const PERSONA_SEEDS = [42, 913, 2044, 5517, 8801] as const

const PERSONA_BASE = [
  {
    id: "jordan",
    label: "Jordan Ellis",
    tagline: "First-time buyer · Denver",
    propensity: 0.186,
    tier: "favorable" as const,
    intakeFields: [
      { label: "Prospect", value: "Jordan Ellis · first-time buyer" },
      { label: "Territory", value: "Denver metro · CO admitted carriers only" },
      { label: "Dwelling", value: "$420k replacement · HO-3" },
      { label: "Liability", value: "$500k · umbrella quoted" },
      { label: "Vehicles", value: "2 autos · 1 motorcycle endorsement" },
      { label: "Loss history", value: "0 homeowner claims · 1 not-at-fault auto (2022)" },
      { label: "Credit / pay plan", value: "Good · monthly EFT" },
    ],
  },
  {
    id: "maria",
    label: "Maria Chen",
    tagline: "Coastal condo · wind pool",
    propensity: 0.241,
    tier: "standard" as const,
    intakeFields: [
      { label: "Prospect", value: "Maria Chen · relocating from NYC" },
      { label: "Territory", value: "Miami-Dade · FL admitted + wind" },
      { label: "Dwelling", value: "HO-6 condo · $85k interior / assoc. master" },
      { label: "Liability", value: "$300k · no umbrella yet" },
      { label: "Vehicles", value: "1 auto · leased" },
      { label: "Loss history", value: "1 water loss (2019) · subrogation recovered" },
      { label: "Credit / pay plan", value: "Excellent · annual pay" },
    ],
  },
  {
    id: "robert",
    label: "Robert Walsh",
    tagline: "Rural · farm & equipment",
    propensity: 0.312,
    tier: "standard" as const,
    intakeFields: [
      { label: "Prospect", value: "Robert Walsh · owner-occupant + small acreage" },
      { label: "Territory", value: "Central TX · county mutual eligible" },
      { label: "Dwelling", value: "$280k dwelling · outbuilding scheduled" },
      { label: "Liability", value: "$1M · farm liability endorsement" },
      { label: "Vehicles", value: "Truck + UTV · equipment rider" },
      { label: "Loss history", value: "2 wind/hail (roof replaced 2021)" },
      { label: "Credit / pay plan", value: "Good · semi-annual" },
    ],
  },
  {
    id: "ava",
    label: "Ava Park",
    tagline: "Urban row home · parking",
    propensity: 0.198,
    tier: "favorable" as const,
    intakeFields: [
      { label: "Prospect", value: "Ava Park · renewal shopping" },
      { label: "Territory", value: "Philadelphia PA · city limits" },
      { label: "Dwelling", value: "$510k HO-3 · masonry" },
      { label: "Liability", value: "$500k · umbrella $1M quoted" },
      { label: "Vehicles", value: "1 EV + 1 hybrid" },
      { label: "Loss history", value: "0 claims · 5-year clean" },
      { label: "Credit / pay plan", value: "Very good · autopay" },
    ],
  },
  {
    id: "devon",
    label: "Devon Brooks",
    tagline: "Young household · multi-claims",
    propensity: 0.367,
    tier: "elevated" as const,
    intakeFields: [
      { label: "Prospect", value: "Devon Brooks · budget-conscious family" },
      { label: "Territory", value: "Phoenix AZ · suburban tract" },
      { label: "Dwelling", value: "$355k · HO-3 · pool fenced" },
      { label: "Liability", value: "$300k · higher limits declined" },
      { label: "Vehicles", value: "3 autos · 1 youthful operator" },
      { label: "Loss history", value: "2 at-fault auto · 1 theft (recovered)" },
      { label: "Credit / pay plan", value: "Fair · monthly card" },
    ],
  },
] as const

export const BROKER_PERSONAS: BrokerPersona[] = PERSONA_SEEDS.map((seed, i) => {
  const base = PERSONA_BASE[i]!
  return {
    id: base.id,
    label: base.label,
    tagline: base.tagline,
    intakeFields: [...base.intakeFields],
    propensity: base.propensity,
    tier: base.tier,
    rankedPartners: i === 0 ? identityRank() : rankPartnersForSeed(seed),
  }
})

/** @deprecated Use selected persona from BROKER_PERSONAS */
export const INTAKE_FIELDS = BROKER_PERSONAS[0]!.intakeFields
export const DEMO_PROPENSITY = BROKER_PERSONAS[0]!.propensity
export const DEMO_TIER = BROKER_PERSONAS[0]!.tier
export const RANKED_PARTNERS = BROKER_PERSONAS[0]!.rankedPartners
