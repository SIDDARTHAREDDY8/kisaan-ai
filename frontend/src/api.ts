import type { AnalyzeResponse, MarketPrice } from './types'

const BASE = '/api'

export async function analyzeImage(
  image: File | null,
  query: string,
  commodity: string,
  location: string,
): Promise<AnalyzeResponse> {
  const form = new FormData()
  if (image) form.append('image', image)
  form.append('query', query)
  form.append('commodity', commodity)
  form.append('location', location)

  const res = await fetch(`${BASE}/analyze`, { method: 'POST', body: form })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail ?? 'Request failed')
  }
  return res.json()
}

export async function fetchMarket(commodity: string, state: string): Promise<MarketPrice[]> {
  const params = new URLSearchParams({ commodity, state })
  const res = await fetch(`${BASE}/market?${params}`)
  if (!res.ok) throw new Error('Market fetch failed')
  const data = await res.json()
  return data.prices
}
