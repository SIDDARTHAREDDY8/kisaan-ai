export default function Header() {
  return (
    <header className="bg-green-700 text-white px-6 py-4 flex items-center gap-3 shadow-md">
      <span className="text-3xl">🌿</span>
      <div>
        <h1 className="text-xl font-bold tracking-tight leading-none">Kisaan AI</h1>
        <p className="text-green-200 text-xs mt-0.5">Autonomous Farm Intelligence</p>
      </div>
      <div className="ml-auto flex gap-2 text-xs text-green-200">
        <span className="bg-green-800 rounded-full px-3 py-1">Crop Disease</span>
        <span className="bg-green-800 rounded-full px-3 py-1">Market Prices</span>
        <span className="bg-green-800 rounded-full px-3 py-1">RAG Advisory</span>
      </div>
    </header>
  )
}
