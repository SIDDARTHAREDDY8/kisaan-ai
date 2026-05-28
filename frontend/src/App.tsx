import { useState } from 'react'
import './index.css'
import { analyzeImage } from './api'
import type { AnalyzeResponse } from './types'
import Header from './components/Header'
import ImageUploader from './components/ImageUploader'
import ResultCard from './components/ResultCard'
import MarketPanel from './components/MarketPanel'
import VoiceRecorder from './components/VoiceRecorder'
import SoilAnalyzer from './components/SoilAnalyzer'
import SchemeNavigator from './components/SchemeNavigator'
import { Leaf, TrendingUp, Mic, Layers, FileText, Loader2 } from 'lucide-react'

type Tab = 'disease' | 'market' | 'voice' | 'soil' | 'schemes'

const TABS = [
  { key: 'disease' as Tab, label: 'Crop Disease', Icon: Leaf },
  { key: 'market' as Tab, label: 'Market Prices', Icon: TrendingUp },
  { key: 'voice' as Tab, label: 'Voice Query', Icon: Mic },
  { key: 'soil' as Tab, label: 'Soil Health', Icon: Layers },
  { key: 'schemes' as Tab, label: 'Govt Schemes', Icon: FileText },
]

export default function App() {
  const [tab, setTab] = useState<Tab>('disease')
  const [image, setImage] = useState<File | null>(null)
  const [query, setQuery] = useState('')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<AnalyzeResponse | null>(null)
  const [error, setError] = useState('')

  async function handleSubmit() {
    if (!image && !query.trim()) {
      setError('Upload a crop photo or type a question.')
      return
    }
    setLoading(true); setError(''); setResult(null)
    try {
      setResult(await analyzeImage(image, query, '', ''))
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Something went wrong.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div style={{ minHeight: '100vh', backgroundColor: '#f9fafb', display: 'flex', flexDirection: 'column' }}>
      <Header />

      {/* Tab bar */}
      <div className="bg-white border-b border-gray-200 px-4 flex gap-0 overflow-x-auto">
        {TABS.map(({ key, label, Icon }) => (
          <button
            key={key}
            onClick={() => setTab(key)}
            className={`flex items-center gap-1.5 px-4 py-3 text-xs font-medium border-b-2 transition whitespace-nowrap ${
              tab === key
                ? 'border-green-600 text-green-700'
                : 'border-transparent text-gray-500 hover:text-gray-700'
            }`}
          >
            <Icon size={14} />
            {label}
          </button>
        ))}
      </div>

      <main style={{ flex: 1, maxWidth: '768px', margin: '0 auto', width: '100%', padding: '24px 16px' }}>
        <div className="space-y-5">
          {tab === 'disease' && (
            <>
              <div className="bg-white border border-gray-200 rounded-2xl shadow-sm p-5 space-y-4">
                <ImageUploader onFileChange={setImage} />
                <textarea
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  placeholder="Describe what you see, or ask any crop question…"
                  rows={3}
                  className="w-full border border-gray-300 rounded-xl px-4 py-3 text-sm resize-none focus:ring-2 focus:ring-green-500 focus:outline-none"
                />
                {error && <p className="text-sm text-red-600">{error}</p>}
                <button
                  onClick={handleSubmit} disabled={loading}
                  className="w-full bg-green-700 text-white rounded-xl py-3 text-sm font-semibold hover:bg-green-800 transition disabled:opacity-60 flex items-center justify-center gap-2"
                >
                  {loading ? <><Loader2 size={16} className="animate-spin" />Analyzing with 4 AI agents…</> : 'Analyze Crop'}
                </button>
              </div>
              {result && <ResultCard result={result} />}
            </>
          )}

          {tab === 'market' && <MarketPanel />}
          {tab === 'voice' && <VoiceRecorder />}
          {tab === 'soil' && <SoilAnalyzer />}
          {tab === 'schemes' && <SchemeNavigator />}
        </div>
      </main>

      <footer style={{ textAlign: 'center', fontSize: '12px', color: '#9ca3af', padding: '16px', borderTop: '1px solid #e5e7eb' }}>
        Kisaan AI v2 · HuggingFace (Whisper · NLLB · MMS-TTS · DETR · NLI · BERT-NER) + LangGraph + Claude + pgvector
      </footer>
    </div>
  )
}
