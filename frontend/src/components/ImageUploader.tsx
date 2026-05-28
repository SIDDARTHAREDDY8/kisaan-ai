import { useRef, useState } from 'react'
import { Upload, X } from 'lucide-react'

interface Props {
  onFileChange: (file: File | null) => void
}

export default function ImageUploader({ onFileChange }: Props) {
  const [preview, setPreview] = useState<string | null>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  function handleFile(file: File | null) {
    if (!file) return
    setPreview(URL.createObjectURL(file))
    onFileChange(file)
  }

  function handleDrop(e: React.DragEvent) {
    e.preventDefault()
    handleFile(e.dataTransfer.files[0] ?? null)
  }

  function clear() {
    setPreview(null)
    onFileChange(null)
    if (inputRef.current) inputRef.current.value = ''
  }

  return (
    <div className="w-full">
      {preview ? (
        <div className="relative rounded-xl overflow-hidden border-2 border-green-400">
          <img src={preview} alt="crop" className="w-full max-h-72 object-contain bg-gray-50" />
          <button
            onClick={clear}
            className="absolute top-2 right-2 bg-white rounded-full p-1 shadow hover:bg-red-50 transition"
          >
            <X size={16} className="text-gray-500" />
          </button>
        </div>
      ) : (
        <div
          onDrop={handleDrop}
          onDragOver={(e) => e.preventDefault()}
          onClick={() => inputRef.current?.click()}
          className="border-2 border-dashed border-green-400 rounded-xl p-10 text-center cursor-pointer
                     hover:bg-green-50 transition flex flex-col items-center gap-3"
        >
          <Upload size={32} className="text-green-500" />
          <p className="text-sm text-gray-600">
            Drag & drop a crop photo, or <span className="text-green-700 font-medium">click to upload</span>
          </p>
          <p className="text-xs text-gray-400">JPG, PNG up to 10 MB</p>
        </div>
      )}
      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        className="hidden"
        onChange={(e) => handleFile(e.target.files?.[0] ?? null)}
      />
    </div>
  )
}
