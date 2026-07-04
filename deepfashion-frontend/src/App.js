import { useState, useRef, useCallback, useEffect } from 'react';
import './App.css';

const API_BASE = 'http://localhost:8000';

function App() {
  const [preview, setPreview] = useState(null);
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [dragging, setDragging] = useState(false);
  const [topK, setTopK] = useState(5);
  const [selectedFile, setSelectedFile] = useState(null);
  const [searchQuery, setSearchQuery] = useState('');

  const fileInputRef = useRef(null);

  const performTextSearch = async (query) => {
    if (!query.trim()) return;

    setLoading(true);
    setResults([]);
    setError(null);

    try {
      const response = await fetch(
        `${API_BASE}/search-text?query=${encodeURIComponent(query)}&k=${topK}`
      );

      if (!response.ok) throw new Error('Search failed');

      const data = await response.json();
      setResults(data);
    } catch {
      setError('Search failed.');
    } finally {
      setLoading(false);
    }
  };

  const handleTextSearch = (e) => {
    if (e.key === 'Enter') {
      performTextSearch(searchQuery);
    }
  };

  const handleFile = async (file) => {
    if (!file || !file.type.startsWith('image/')) {
      setError('Please upload an image file.');
      return;
    }

    setSelectedFile(file);

    setError(null);
    setResults([]);

    if (!preview) {
      setPreview(URL.createObjectURL(file));
    }

    setLoading(true);

    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch(`${API_BASE}/search?k=${topK}`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) throw new Error('Search failed');

      const data = await response.json();
      setResults(data);
    } catch {
      setError('Search failed. Make sure the backend is running.');
    } finally {
      setLoading(false);
    }
  };

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    setDragging(false);
    handleFile(e.dataTransfer.files[0]);
  }, []);

  const handleDragOver = (e) => {
    e.preventDefault();
    setDragging(true);
  };

  const handleDragLeave = () => setDragging(false);

  const getImageUrl = (path) =>
    `${API_BASE}/image?image_path=${encodeURIComponent(path)}`;

  useEffect(() => {
    if (selectedFile) {
      handleFile(selectedFile);
    } else if (searchQuery.trim()) {
      performTextSearch(searchQuery);
    }
  }, [topK]);

  return (
    <div className="app">
      <header className="header">
        <h1 className="logo">FASHION<span>LENS</span></h1>
        <p className="tagline">Find similar fashion, instantly</p>
      </header>

      <div className="search-bar">
        <svg className="search-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <circle cx="11" cy="11" r="8" />
          <path d="m21 21-4.35-4.35" />
        </svg>

        <input
          type="text"
          placeholder="Search fashion... (e.g. blue oversized hoodie)"
          className="search-input"
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          onKeyDown={handleTextSearch}
        />

        <button className="icon-btn" title="Voice search (coming soon)" disabled>
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" width="18" height="18">
            <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z" />
            <path d="M19 10v2a7 7 0 0 1-14 0v-2M12 19v4M8 23h8" />
          </svg>
        </button>

        <select
          className="k-select"
          value={topK}
          onChange={(e) => setTopK(Number(e.target.value))}
        >
          <option value={5}>Top 5</option>
          <option value={10}>Top 10</option>
          <option value={20}>Top 20</option>
        </select>

        <button
          className="icon-btn camera-btn"
          onClick={() => fileInputRef.current.click()}
          title="Search by image"
        >
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" width="18" height="18">
            <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z" />
            <circle cx="12" cy="13" r="4" />
          </svg>
          <span>Search by image</span>
        </button>
      </div>

      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        style={{ display: 'none' }}
        onChange={(e) => handleFile(e.target.files[0])}
      />

      {!preview && (
        <div
          className={`drop-zone ${dragging ? 'dragging' : ''}`}
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onClick={() => fileInputRef.current.click()}
        >
          <div className="drop-icon">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" width="48" height="48">
              <rect x="3" y="3" width="18" height="18" rx="4" />
              <circle cx="8.5" cy="8.5" r="1.5" />
              <path d="m21 15-5-5L5 21" />
            </svg>
          </div>
          <p>Drop a photo here or click to upload</p>
          <span>Find visually similar fashion items instantly</span>
        </div>
      )}

      {preview && (
        <div className="query-section">
          <p className="section-label">Your item</p>
          <div className="query-image-container">
            <img src={preview} alt="Query" className="query-image" />
            <button
              className="search-again-btn"
              onClick={() => {
                setPreview(null);
                setResults([]);
                setError(null);
                setSelectedFile(null);
                setSearchQuery('');
              }}
            >
              ← Search again
            </button>
          </div>
        </div>
      )}

      {loading && (
        <div className="loading">
          <div className="spinner" />
          <p>Finding similar items...</p>
        </div>
      )}

      {results.length > 0 && (
        <div className="results-section">
          <p className="section-label">Similar items</p>
          <div className="results-grid">
            {results.map((item, i) => (
              <div key={i} className="result-card">
                <img
                  src={getImageUrl(item.image_path)}
                  alt={item.item_id}
                  className="result-image"
                />
                <div className="result-info">
                  <span className="score">{(item.score * 100).toFixed(0)}% match</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {error && <div className="error">{error}</div>}
    </div>
  );
}

export default App;