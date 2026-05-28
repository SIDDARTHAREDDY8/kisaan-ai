export interface AnalyzeResponse {
  session_id: string
  intent: string
  crop: string
  condition: string
  confidence: number
  severity: string
  response: string
  top5: { label: string; score: number }[]
  retrieved_docs: RetrievedDoc[]
  agent_trace: string[]
  latency_ms: number
}

export interface RetrievedDoc {
  disease_name: string
  crop: string
  symptoms: string
  cause: string
  treatment: string
  prevention: string
  severity: string
  similarity: number
}

export interface MarketPrice {
  commodity: string
  market: string
  state: string
  district: string
  min_price: number | null
  max_price: number | null
  modal_price: number | null
  arrival_date: string
  unit: string
}
