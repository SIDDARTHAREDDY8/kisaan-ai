const palette: Record<string, string> = {
  low: 'bg-green-100 text-green-800',
  medium: 'bg-yellow-100 text-yellow-800',
  high: 'bg-orange-100 text-orange-800',
  critical: 'bg-red-100 text-red-800',
}

export default function SeverityBadge({ severity }: { severity: string }) {
  const cls = palette[severity] ?? 'bg-gray-100 text-gray-700'
  return (
    <span className={`text-xs font-semibold px-2.5 py-0.5 rounded-full uppercase tracking-wide ${cls}`}>
      {severity}
    </span>
  )
}
