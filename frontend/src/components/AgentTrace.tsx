import { useState } from 'react'
import { ChevronDown, ChevronUp, Terminal } from 'lucide-react'

export default function AgentTrace({ trace }: { trace: string[] }) {
  const [open, setOpen] = useState(false)
  if (!trace.length) return null

  return (
    <div className="mt-4 border border-gray-200 rounded-lg overflow-hidden">
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-center gap-2 px-4 py-2.5 bg-gray-50 text-sm font-medium text-gray-600 hover:bg-gray-100 transition"
      >
        <Terminal size={14} />
        Agent Reasoning Chain ({trace.length} steps)
        {open ? <ChevronUp size={14} className="ml-auto" /> : <ChevronDown size={14} className="ml-auto" />}
      </button>
      {open && (
        <ol className="divide-y divide-gray-100">
          {trace.map((step, i) => (
            <li key={i} className="px-4 py-2 text-xs font-mono text-gray-600 bg-white flex gap-2">
              <span className="text-gray-400 select-none">{i + 1}.</span>
              {step}
            </li>
          ))}
        </ol>
      )}
    </div>
  )
}
