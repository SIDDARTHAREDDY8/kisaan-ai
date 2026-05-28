import { useState } from 'react'
import { FileText, Loader2, Search } from 'lucide-react'
import AgentTrace from './AgentTrace'

interface SchemeResponse {
  response: string
  retrieved_docs: SchemeDoc[]
  agent_trace: string[]
}
interface SchemeDoc {
  scheme_name: string
  category: string
  benefit: string
  eligibility: string
  how_to_apply: string
  contact: string
  similarity: number
}

const EXAMPLE_QUERIES = [
  'How do I get crop insurance?',
  'I am a small farmer with 1 acre, what schemes am I eligible for?',
  'How to apply for Kisan Credit Card?',
  'What is PM-KISAN and how to register?',
  'Subsidy for drip irrigation',
]

export default function SchemeNavigator() {
  const [query, setQuery] = useState('')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<SchemeResponse | null>(null)
  const [error, setError] = useState('')

  async function search(q: string = query) {
    if (!q.trim()) return
    setLoading(true); setError(''); setResult(null)
    try {
      const res = await fetch(`/api/schemes?q=${encodeURIComponent(q)}`)
      if (!res.ok) throw new Error(await res.text())
      setResult(await res.json())
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Search failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="space-y-5">
      <div className="bg-white border border-gray-200 rounded-2xl shadow-sm p-5 space-y-4">
        <div className="flex items-center gap-2">
          <FileText size={16} className="text-green-700" />
          <span className="font-semibold text-gray-800 text-sm">Government Scheme Navigator</span>
          <span className="ml-auto text-xs text-gray-400">PM-KISAN · PMFBY · KCC · eNAM · PMKSY</span>
        </div>

        <div className="flex gap-2">
          <input
            value={query}
            onChange={e => setQuery(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && search()}
            placeholder="Ask about any government farming scheme..."
            className="flex-1 border border-gray-300 rounded-xl px-4 py-2.5 text-sm focus:ring-2 focus:ring-green-500 focus:outline-none"
          />
          <button
            onClick={() => search()} disabled={loading || !query.trim()}
            className="bg-green-700 text-white rounded-xl px-4 py-2.5 text-sm font-medium hover:bg-green-800 transition disabled:opacity-50 flex items-center gap-1"
          >
            {loading ? <Loader2 size={15} className="animate-spin" /> : <Search size={15} />}
          </button>
        </div>

        <div className="flex flex-wrap gap-2">
          {EXAMPLE_QUERIES.map(q => (
            <button
              key={q} onClick={() => { setQuery(q); search(q) }}
              className="text-xs bg-green-50 text-green-700 border border-green-200 rounded-full px-3 py-1 hover:bg-green-100 transition"
            >
              {q}
            </button>
          ))}
        </div>

        {error && <p className="text-sm text-red-600">{error}</p>}
      </div>

      {result && (
        <div className="bg-white border border-gray-200 rounded-2xl shadow-sm overflow-hidden">
          <div className="bg-green-700 text-white px-5 py-3">
            <span className="font-semibold text-sm">Scheme Recommendations</span>
            <span className="ml-2 text-green-200 text-xs">{result.retrieved_docs.length} schemes retrieved</span>
          </div>
          <div className="p-5 space-y-4">
            <div className="prose prose-sm max-w-none text-gray-700 whitespace-pre-wrap">{result.response}</div>

            {result.retrieved_docs.length > 0 && (
              <details>
                <summary className="text-xs font-semibold text-gray-500 uppercase tracking-wide cursor-pointer hover:text-green-700">
                  Retrieved Scheme Sources ({result.retrieved_docs.length})
                </summary>
                <div className="mt-2 space-y-2">
                  {result.retrieved_docs.map((doc, i) => (
                    <div key={i} className="bg-green-50 border border-green-100 rounded-lg p-3 text-xs">
                      <p className="font-semibold text-green-800">{doc.scheme_name}
                        <span className="ml-2 text-gray-400 font-normal">({doc.category}) · {(doc.similarity * 100).toFixed(0)}% match</span>
                      </p>
                      <p className="mt-1 text-gray-600"><strong>Contact:</strong> {doc.contact}</p>
                    </div>
                  ))}
                </div>
              </details>
            )}

            <AgentTrace trace={result.agent_trace} />
          </div>
        </div>
      )}
    </div>
  )
}
