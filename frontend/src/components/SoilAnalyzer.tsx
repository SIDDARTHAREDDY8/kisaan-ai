import { useState } from 'react'
import { Layers, Loader2 } from 'lucide-react'
import AgentTrace from './AgentTrace'

interface SoilResponse {
  health_class: string
  health_score: number
  deficiencies: string[]
  excesses: string[]
  suitable_crops: string[]
  amendments: string[]
  response: string
  agent_trace: string[]
}

const HEALTH_COLOR: Record<string, string> = {
  Excellent: 'bg-green-500', Good: 'bg-lime-500',
  Fair: 'bg-yellow-500', Poor: 'bg-orange-500', Critical: 'bg-red-500',
}

export default function SoilAnalyzer() {
  const [params, setParams] = useState({
    nitrogen: 280, phosphorus: 15, potassium: 150,
    ph: 7.0, organic_carbon: 0.6, moisture: 20,
  })
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<SoilResponse | null>(null)
  const [error, setError] = useState('')

  function set(key: string, val: string) {
    setParams(p => ({ ...p, [key]: parseFloat(val) || 0 }))
  }

  async function analyze() {
    setLoading(true); setError(''); setResult(null)
    try {
      const res = await fetch('/api/soil/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(params),
      })
      if (!res.ok) throw new Error(await res.text())
      setResult(await res.json())
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Analysis failed')
    } finally {
      setLoading(false)
    }
  }

  const fields = [
    { key: 'nitrogen', label: 'Nitrogen (kg/ha)', placeholder: '280', help: 'Low<280 | Med 280-560 | High>560' },
    { key: 'phosphorus', label: 'Phosphorus (kg/ha)', placeholder: '15', help: 'Low<10 | Med 10-25 | High>25' },
    { key: 'potassium', label: 'Potassium (kg/ha)', placeholder: '150', help: 'Low<108 | Med 108-280 | High>280' },
    { key: 'ph', label: 'pH', placeholder: '7.0', help: 'Optimal: 6.5–7.5' },
    { key: 'organic_carbon', label: 'Organic Carbon (%)', placeholder: '0.6', help: 'Low<0.5 | Med 0.5-0.75 | High>0.75' },
    { key: 'moisture', label: 'Soil Moisture (%)', placeholder: '20', help: 'Optimal: 15–35%' },
  ]

  return (
    <div className="space-y-5">
      <div className="bg-white border border-gray-200 rounded-2xl shadow-sm p-5 space-y-4">
        <div className="flex items-center gap-2 mb-1">
          <Layers size={16} className="text-green-700" />
          <span className="font-semibold text-gray-800 text-sm">Enter Soil Test Values</span>
          <span className="ml-auto text-xs text-gray-400">From your Soil Health Card or lab report</span>
        </div>
        <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
          {fields.map(f => (
            <div key={f.key}>
              <label className="block text-xs font-medium text-gray-600 mb-1">{f.label}</label>
              <input
                type="number" step="any"
                value={params[f.key as keyof typeof params]}
                onChange={e => set(f.key, e.target.value)}
                placeholder={f.placeholder}
                className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-green-500 focus:outline-none"
              />
              <p className="text-xs text-gray-400 mt-0.5">{f.help}</p>
            </div>
          ))}
        </div>
        {error && <p className="text-sm text-red-600">{error}</p>}
        <button
          onClick={analyze} disabled={loading}
          className="w-full bg-green-700 text-white rounded-xl py-3 text-sm font-semibold hover:bg-green-800 transition disabled:opacity-60 flex items-center justify-center gap-2"
        >
          {loading ? <><Loader2 size={16} className="animate-spin" />Analyzing soil...</> : 'Analyze Soil Health'}
        </button>
      </div>

      {result && (
        <div className="bg-white border border-gray-200 rounded-2xl shadow-sm overflow-hidden">
          <div className="bg-green-700 text-white px-5 py-3 flex items-center gap-3">
            <span className="font-semibold text-sm">Soil Health Report</span>
            <span className={`text-xs font-bold px-2 py-0.5 rounded-full ${HEALTH_COLOR[result.health_class] ?? 'bg-gray-500'}`}>
              {result.health_class}
            </span>
            <span className="ml-auto text-green-200 text-xs">{result.health_score}/100</span>
          </div>
          <div className="p-5 space-y-4">
            {/* Score bar */}
            <div className="w-full bg-gray-100 rounded-full h-3">
              <div
                className={`h-3 rounded-full transition-all ${HEALTH_COLOR[result.health_class] ?? 'bg-green-500'}`}
                style={{ width: `${result.health_score}%` }}
              />
            </div>

            <div className="grid grid-cols-2 gap-3 text-sm">
              {result.deficiencies.length > 0 && (
                <div className="bg-red-50 rounded-lg p-3">
                  <p className="font-semibold text-red-700 text-xs mb-1">⚠ Deficiencies</p>
                  <ul className="text-red-600 text-xs space-y-0.5">{result.deficiencies.map(d => <li key={d}>• {d}</li>)}</ul>
                </div>
              )}
              {result.excesses.length > 0 && (
                <div className="bg-orange-50 rounded-lg p-3">
                  <p className="font-semibold text-orange-700 text-xs mb-1">↑ Excesses</p>
                  <ul className="text-orange-600 text-xs space-y-0.5">{result.excesses.map(e => <li key={e}>• {e}</li>)}</ul>
                </div>
              )}
              {result.suitable_crops.length > 0 && (
                <div className="bg-green-50 rounded-lg p-3">
                  <p className="font-semibold text-green-700 text-xs mb-1">✓ Suitable Crops</p>
                  <p className="text-green-600 text-xs">{result.suitable_crops.join(', ')}</p>
                </div>
              )}
            </div>

            {result.amendments.length > 0 && (
              <div>
                <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-2">Recommended Amendments</p>
                <ul className="space-y-1">
                  {result.amendments.map((a, i) => (
                    <li key={i} className="text-xs text-gray-700 bg-yellow-50 border border-yellow-100 rounded-lg px-3 py-2">• {a}</li>
                  ))}
                </ul>
              </div>
            )}

            <div className="prose prose-sm max-w-none text-gray-700 text-sm">
              <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-1">AI Advisory</p>
              <p className="whitespace-pre-wrap">{result.response}</p>
            </div>

            <AgentTrace trace={result.agent_trace} />
          </div>
        </div>
      )}
    </div>
  )
}
