/** Modern Movie Search with live results and filters */

import { useState, useEffect, useCallback } from 'react';
import { movieApi } from '../api';
import MovieCard from './MovieCard';

function MovieSearch({ onSelect, onRate, onLike, onWatchlist, userInteractions = {} }) {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [showResults, setShowResults] = useState(false);
  const [genres, setGenres] = useState([]);
  const [selectedGenre, setSelectedGenre] = useState('');

  useEffect(() => { movieApi.getGenres().then((data) => setGenres(data.genres || [])).catch(() => {}); }, []);

  // Live search with debounce
  useEffect(() => {
    if (query.trim().length < 2 && !selectedGenre) {
      setResults([]);
      setShowResults(false);
      return;
    }

    const timer = setTimeout(() => {
      searchMovies(query, selectedGenre);
    }, 300);

    return () => clearTimeout(timer);
  }, [query, selectedGenre]);

  const searchMovies = async (searchQuery, genre = selectedGenre) => {
    setLoading(true);
    try {
      const movies = await movieApi.search({ query: searchQuery.trim() || undefined, genres: genre ? [genre] : [], limit: 40 });
      setResults(movies);
      setShowResults(true);
    } catch (err) {
      console.error('Search failed:', err);
      setResults([]);
    } finally {
      setLoading(false);
    }
  };

  const handleClear = () => {
    setQuery('');
    setResults([]);
    setShowResults(false);
  };

  const handleMovieClick = (movie) => {
    onSelect?.(movie);
  };

  return (
    <div style={styles.container}>
      <div style={styles.searchBox}>
        <div style={styles.searchInputWrapper}>
          <span style={styles.searchIcon}>🔍</span>
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search movies by title..."
            style={styles.searchInput}
            onFocus={() => query.trim() && setShowResults(true)}
          />
          {query && (
            <button onClick={handleClear} style={styles.clearBtn}>
              ✕
            </button>
          )}
          {loading && <div style={styles.spinner} />}
        </div>
      </div>

      <div style={styles.filterRow}>
        <span style={styles.filterLabel}>Browse by genre</span>
        <select value={selectedGenre} onChange={(e) => setSelectedGenre(e.target.value)} style={styles.genreSelect}>
          <option value="">All genres</option>
          {genres.map((genre) => <option value={genre} key={genre}>{genre}</option>)}
        </select>
      </div>

      {showResults && results.length > 0 && (
        <div style={styles.resultsContainer} className="animate-fadeIn">
          <div style={styles.resultsHeader}>
            <h3 style={styles.resultsTitle}>
              {results.length} titles
            </h3>
            <button onClick={() => setShowResults(false)} style={styles.closeBtn}>
              Close
            </button>
          </div>
          <div style={styles.resultsGrid}>
            {results.map((movie) => (
              <div key={movie.movie_id} onClick={() => handleMovieClick(movie)}>
                <MovieCard
                  movie={movie}
                  userInteractions={userInteractions}
                  onRate={onRate}
                  onLike={onLike}
                  onWatchlist={onWatchlist}
                  showActions={true}
                />
              </div>
            ))}
          </div>
        </div>
      )}

      {showResults && results.length === 0 && !loading && query.trim() && (
        <div style={styles.noResults} className="animate-fadeIn">
          <p style={styles.noResultsText}>
            No movies found for "{query}"
          </p>
          <p style={styles.noResultsHint}>
            Try different keywords or check the spelling
          </p>
        </div>
      )}
    </div>
  );
}

const styles = {
  container: {
    marginBottom: 'var(--space-8)',
  },
  searchBox: {
    marginBottom: 'var(--space-6)',
  },
  filterRow: { display: 'flex', alignItems: 'center', gap: 12, margin: '-12px 0 20px', color: 'var(--color-text-muted)', fontSize: 13 },
  filterLabel: { fontWeight: 600 },
  genreSelect: { background: 'var(--color-surface)', color: 'var(--color-text)', border: '1px solid var(--color-border)', borderRadius: 10, padding: '9px 12px' },
  searchInputWrapper: {
    position: 'relative',
    display: 'flex',
    alignItems: 'center',
    background: 'var(--color-surface)',
    border: '2px solid var(--color-border)',
    borderRadius: 'var(--radius-xl)',
    padding: 'var(--space-2) var(--space-4)',
    transition: 'all var(--transition-fast)',
  },
  searchIcon: {
    fontSize: '20px',
    marginRight: 'var(--space-3)',
    color: 'var(--color-text-muted)',
  },
  searchInput: {
    flex: 1,
    background: 'none',
    border: 'none',
    outline: 'none',
    fontSize: 'var(--font-size-lg)',
    color: 'var(--color-text)',
    padding: 'var(--space-2) 0',
  },
  clearBtn: {
    width: '32px',
    height: '32px',
    borderRadius: 'var(--radius-full)',
    background: 'var(--color-surface-elevated)',
    color: 'var(--color-text-muted)',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    marginLeft: 'var(--space-2)',
    cursor: 'pointer',
    fontSize: '16px',
    transition: 'all var(--transition-fast)',
  },
  spinner: {
    width: '20px',
    height: '20px',
    border: '2px solid var(--color-border)',
    borderTopColor: 'var(--color-primary)',
    borderRadius: '50%',
    animation: 'spin 1s linear infinite',
    marginLeft: 'var(--space-2)',
  },
  resultsContainer: {
    background: 'var(--color-surface)',
    borderRadius: 'var(--radius-xl)',
    padding: 'var(--space-6)',
    boxShadow: 'var(--shadow-lg)',
  },
  resultsHeader: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 'var(--space-6)',
    paddingBottom: 'var(--space-4)',
    borderBottom: '1px solid var(--color-border)',
  },
  resultsTitle: {
    fontSize: 'var(--font-size-xl)',
    fontWeight: 'var(--font-weight-bold)',
    color: 'var(--color-text)',
  },
  closeBtn: {
    padding: 'var(--space-2) var(--space-4)',
    background: 'var(--color-surface-elevated)',
    color: 'var(--color-text)',
    borderRadius: 'var(--radius-md)',
    fontSize: 'var(--font-size-sm)',
    fontWeight: 'var(--font-weight-medium)',
    cursor: 'pointer',
    transition: 'all var(--transition-fast)',
  },
  resultsGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fill, minmax(180px, 1fr))',
    gap: 'var(--space-6)',
  },
  noResults: {
    textAlign: 'center',
    padding: 'var(--space-12)',
    background: 'var(--color-surface)',
    borderRadius: 'var(--radius-xl)',
  },
  noResultsText: {
    fontSize: 'var(--font-size-lg)',
    color: 'var(--color-text)',
    marginBottom: 'var(--space-2)',
  },
  noResultsHint: {
    fontSize: 'var(--font-size-sm)',
    color: 'var(--color-text-muted)',
  },
};

export default MovieSearch;
