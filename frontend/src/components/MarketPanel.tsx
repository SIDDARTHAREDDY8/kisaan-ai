import { useState } from 'react'
import { TrendingUp, Search } from 'lucide-react'
import { fetchMarket } from '../api'
import type { MarketPrice } from '../types'

export default function MarketPanel() {
  const [commodity, setCommodity] = useState('Tomato')
  const [state, setState] = useState('')
  const [prices, setPrices] = useState<MarketPrice[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  async function search() {
    setLoading(true)
    setError('')
    try {
      const data = await fetchMarket(commodity, state)
      setPrices(data)
      if (!data.length) setError('No price data found. Try a different commodity or state.')
    } catch {
      setError('Failed to fetch market data.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="bg-white border border-gray-200 rounded-2xl shadow-sm overflow-hidden">
      <div className="bg-green-700 text-white px-5 py-3 flex items-center gap-2">
        <TrendingUp size={16} />
        <span className="font-semibold text-sm">Mandi Price Intelligence</span>
        <span className="ml-auto text-xs text-green-200">via India Agmarknet API</span>
      </div>
      <div className="p-5 space-y-4">
        <div className="flex gap-2">
          <input
            value={commodity}
            onChange={(e) => setCommodity(e.target.value)}
            placeholder="Commodity (e.g. Tomato)"
            className="flex-1 border border-gray-300 rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-green-500 focus:outline-none"
          />
          <input
            value={state}
            onChange={(e) => setState(e.target.value)}
            placeholder="State (optional)"
            className="w-36 border border-gray-300 rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-green-500 focus:outline-none"
          />
          <button
            onClick={search}
            disabled={loading}
            className="bg-green-700 text-white rounded-lg px-4 py-2 text-sm font-medium hover:bg-green-800 transition disabled:opacity-50 flex items-center gap-1"
          >
            <Search size={14} />
            {loading ? 'Fetching…' : 'Search'}
          </button>
        </div>

        {error && <p className="text-sm text-red-600">{error}</p>}

        {prices.length > 0 && (
          <div className="overflow-x-auto">
            <table className="w-full text-xs text-left">
              <thead>
                <tr className="bg-green-50 text-green-800">
                  <th className="px-3 py-2 font-semibold">Market</th>
                  <th className="px-3 py-2 font-semibold">State</th>
                  <th className="px-3 py-2 font-semibold text-right">Min ₹</th>
                  <th className="px-3 py-2 font-semibold text-right">Modal ₹</th>
                  <th className="px-3 py-2 font-semibold text-right">Max ₹</th>
                  <th className="px-3 py-2 font-semibold">Date</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-100">
                {prices.map((p, i) => (
                  <tr key={i} className="hover:bg-gray-50">
                    <td className="px-3 py-2 font-medium text-gray-700">{p.market}</td>
                    <td className="px-3 py-2 text-gray-500">{p.state}</td>
                    <td className="px-3 py-2 text-right">{p.min_price ?? '—'}</td>
                    <td className="px-3 py-2 text-right font-semibold text-green-700">{p.modal_price ?? '—'}</td>
                    <td className="px-3 py-2 text-right">{p.max_price ?? '—'}</td>
                    <td className="px-3 py-2 text-gray-400">{p.arrival_date}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <p className="text-xs text-gray-400 mt-2">Per quintal (100 kg). Source: Agmarknet.</p>
          </div>
        )}
      </div>
    </div>
  )
}
