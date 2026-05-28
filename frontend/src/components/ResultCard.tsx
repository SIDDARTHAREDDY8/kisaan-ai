import ReactMarkdown from 'react-markdown'
import type { AnalyzeResponse } from '../types'
import AgentTrace from './AgentTrace'
import SeverityBadge from './SeverityBadge'

export default function ResultCard({ result }: { result: AnalyzeResponse }) {
  const isDiseaseQuery = result.intent === 'disease'

  return (
    <div className="bg-white border border-gray-200 rounded-2xl shadow-sm overflow-hidden">
      {/* Header bar */}
      <div className="bg-green-700 text-white px-5 py-3 flex items-center justify-between gap-3 flex-wrap">
        <div className="flex items-center gap-2">
          <span className="font-semibold text-sm">
            {isDiseaseQuery ? `${result.crop} — ${result.condition}` : result.intent === 'market' ? 'Market Intelligence' : 'Advisory'}
          </span>
          {isDiseaseQuery && <SeverityBadge severity={result.severity} />}
        </div>
        <div className="flex gap-3 text-xs text-green-200">
          {isDiseaseQuery && (
            <span>Confidence: <strong className="text-white">{(result.confidence * 100).toFixed(1)}%</strong></span>
          )}
          <span>Latency: <strong className="text-white">{result.latency_ms.toFixed(0)}ms</strong></span>
        </div>
      </div>

      <div className="p-5 space-y-5">
        {/* AI Response */}
        <div className="prose prose-sm max-w-none text-gray-700">
          <ReactMarkdown>{result.response}</ReactMarkdown>
        </div>

        {/* Top-5 classification results */}
        {result.top5.length > 0 && (
          <div>
            <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-2">Classifier Top-5</p>
            <div className="space-y-1.5">
              {result.top5.map((item, i) => (
                <div key={i} className="flex items-center gap-2">
                  <div className="text-xs text-gray-500 w-4">{i + 1}</div>
                  <div className="flex-1 bg-gray-100 rounded-full h-5 overflow-hidden">
                    <div
                      className={`h-full rounded-full text-xs text-white flex items-center pl-2 ${i === 0 ? 'bg-green-600' : 'bg-gray-400'}`}
                      style={{ width: `${Math.max(item.score * 100, 8)}%` }}
                    >
                      {(item.score * 100).toFixed(1)}%
                    </div>
                  </div>
                  <div className="text-xs text-gray-600 w-48 truncate">{item.label.replace(/___/g, ' › ').replace(/_/g, ' ')}</div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Retrieved docs */}
        {result.retrieved_docs.length > 0 && (
          <details className="group">
            <summary className="text-xs font-semibold text-gray-500 uppercase tracking-wide cursor-pointer hover:text-green-700 transition">
              RAG Sources ({result.retrieved_docs.length} docs retrieved)
            </summary>
            <div className="mt-2 space-y-2">
              {result.retrieved_docs.map((doc, i) => (
                <div key={i} className="bg-green-50 border border-green-100 rounded-lg p-3 text-xs text-gray-700">
                  <p className="font-semibold text-green-800">{doc.disease_name} — {doc.crop}
                    <span className="font-normal text-gray-400 ml-2">similarity: {(doc.similarity * 100).toFixed(1)}%</span>
                  </p>
                  <p className="mt-1"><span className="font-medium">Treatment:</span> {doc.treatment}</p>
                </div>
              ))}
            </div>
          </details>
        )}

        {/* Agent trace */}
        <AgentTrace trace={result.agent_trace} />
      </div>
    </div>
  )
}
