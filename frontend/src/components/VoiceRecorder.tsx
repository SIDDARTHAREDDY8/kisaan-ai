import { useState, useRef } from 'react'
import { Mic, MicOff, Loader2, Volume2 } from 'lucide-react'
import AgentTrace from './AgentTrace'

interface VoiceResponse {
  session_id: string
  transcript: string
  detected_language: string
  intent: string
  response: string
  response_translated: string
  agent_trace: string[]
  latency_ms: number
  has_audio: boolean
}

const LANG_LABELS: Record<string, string> = {
  en: 'English', hi: 'Hindi', te: 'Telugu', ta: 'Tamil',
  mr: 'Marathi', kn: 'Kannada', bn: 'Bengali', sw: 'Swahili',
}

export default function VoiceRecorder() {
  const [recording, setRecording] = useState(false)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<VoiceResponse | null>(null)
  const [error, setError] = useState('')
  const mediaRecorder = useRef<MediaRecorder | null>(null)
  const chunks = useRef<Blob[]>([])

  async function startRecording() {
    setError('')
    setResult(null)
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
    const mr = new MediaRecorder(stream)
    chunks.current = []
    mr.ondataavailable = (e) => chunks.current.push(e.data)
    mr.onstop = handleStop
    mr.start()
    mediaRecorder.current = mr
    setRecording(true)
  }

  function stopRecording() {
    mediaRecorder.current?.stop()
    mediaRecorder.current?.stream.getTracks().forEach(t => t.stop())
    setRecording(false)
    setLoading(true)
  }

  async function handleStop() {
    const blob = new Blob(chunks.current, { type: 'audio/webm' })
    const form = new FormData()
    form.append('audio', blob, 'recording.webm')

    try {
      const res = await fetch('/api/voice/analyze', { method: 'POST', body: form })
      if (!res.ok) throw new Error(await res.text())
      setResult(await res.json())
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Voice analysis failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="space-y-5">
      <div className="bg-white border border-gray-200 rounded-2xl shadow-sm p-6 text-center space-y-4">
        <p className="text-sm text-gray-600">
          Speak your farming question in any language — Hindi, Telugu, Tamil, English, Swahili, and more.
        </p>

        <button
          onClick={recording ? stopRecording : startRecording}
          disabled={loading}
          className={`w-24 h-24 rounded-full mx-auto flex items-center justify-center text-white text-3xl shadow-lg transition ${
            recording ? 'bg-red-500 hover:bg-red-600 animate-pulse' : 'bg-green-700 hover:bg-green-800'
          } disabled:opacity-50`}
        >
          {loading ? <Loader2 size={36} className="animate-spin" /> : recording ? <MicOff size={36} /> : <Mic size={36} />}
        </button>

        <p className="text-sm font-medium text-gray-600">
          {loading ? 'Processing your voice...' : recording ? 'Recording — tap to stop' : 'Tap to start recording'}
        </p>

        {error && <p className="text-sm text-red-600">{error}</p>}
      </div>

      {result && (
        <div className="bg-white border border-gray-200 rounded-2xl shadow-sm overflow-hidden">
          <div className="bg-green-700 text-white px-5 py-3 flex items-center justify-between flex-wrap gap-2">
            <div className="flex items-center gap-2">
              <Volume2 size={15} />
              <span className="font-semibold text-sm">Voice Response</span>
            </div>
            <div className="flex gap-3 text-xs text-green-200">
              <span>Language: <strong className="text-white">{LANG_LABELS[result.detected_language] ?? result.detected_language}</strong></span>
              <span>Intent: <strong className="text-white">{result.intent}</strong></span>
              <span>{result.latency_ms.toFixed(0)}ms</span>
            </div>
          </div>

          <div className="p-5 space-y-4">
            {result.transcript && (
              <div className="bg-gray-50 rounded-lg p-3 text-sm">
                <p className="text-xs font-semibold text-gray-500 mb-1">Transcript</p>
                <p className="text-gray-700 italic">"{result.transcript}"</p>
              </div>
            )}

            <div className="prose prose-sm max-w-none text-gray-700">
              <p className="text-xs font-semibold text-gray-500 mb-1">AI Response
                {result.detected_language !== 'en' && ` (translated to ${LANG_LABELS[result.detected_language] ?? result.detected_language})`}
              </p>
              <p className="whitespace-pre-wrap">{result.response_translated || result.response}</p>
            </div>

            <AgentTrace trace={result.agent_trace} />
          </div>
        </div>
      )}
    </div>
  )
}
