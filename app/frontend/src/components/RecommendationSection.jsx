/** Modern Recommendations Section with grid layout */

import { useState, useEffect } from 'react';
import { recommendationApi } from '../api';
import MovieCard from './MovieCard';

function RecommendationSection({ userId, onInteract, userInteractions = {}, interactionVersion = 0, intent = null }) {
  const [recommendations, setRecommendations] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [refreshing, setRefreshing] = useState(false);

  useEffect(() => {
    if (userId) loadRecommendations(recommendations.length > 0);
  }, [userId, interactionVersion]);

  async function loadRecommendations(quiet = false) {
    if (quiet) setRefreshing(true);
    else setLoading(true);
    setError(null);
    try {
      const data = await recommendationApi.get(userId, { k: 20, method: 'auto', mode: 'personalized', diversity: 0.85 });
      console.log('Recommendations loaded:', data);
      setRecommendations(data.recommendations);
    } catch (err) {
      console.error('Failed to load recommendations:', err);
      setError(err.message || 'Failed to load recommendations');
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }

  const handleRate = (movieId, rating, movie) => {
    onInteract?.(movieId, 'rate', rating, movie);
  };

  const handleLike = (movieId, movie) => {
    onInteract?.(movieId, 'like', undefined, movie);
  };

  const handleWatchlist = (movieId, movie) => {
    onInteract?.(movieId, 'watchlist', undefined, movie);
  };

  const visibleRecommendations = recommendations
    .filter((rec) => {
      const state = userInteractions[rec.movie.movie_id];
      return !state || !(state.rating != null || state.liked || state.watchlisted || state.dismissed);
    })
    .map((rec, index) => {
      const matchedGenres = intent?.movie?.genres?.filter((genre) => rec.movie.genres?.includes(genre)) || [];
      return { rec, index, matchedGenres };
    })
    .sort((a, b) => {
      const isPositive = intent && (intent.type === 'like' || intent.type === 'watchlist' || (intent.type === 'rate' && intent.rating >= 3.5));
      if (!isPositive) return a.index - b.index;
      return b.matchedGenres.length - a.matchedGenres.length || a.index - b.index;
    });

  if (loading) {
    return (
      <div style={styles.loadingContainer}>
        <div style={styles.spinner} />
        <p style={styles.loadingText}>Loading your personalized recommendations...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div style={styles.errorContainer}>
        <span style={styles.errorIcon}>⚠️</span>
        <p style={styles.errorText}>Failed to load recommendations</p>
        <p style={styles.errorDetail}>{error}</p>
        <button onClick={loadRecommendations} style={styles.retryBtn}>
          Try Again
        </button>
      </div>
    );
  }

  return (
    <section id="recommendations" style={styles.container}>
      <div style={styles.header}>
        <div>
          <h2 style={styles.title}>Recommended for You</h2>
          <p style={styles.subtitle}>
            {intent?.type === 'dislike'
              ? `Removed ${intent.movie.title} from your picks. We’ll learn from that.`
              : intent
                ? `Because you ${intent.type === 'rate' ? `rated ${intent.rating}/5` : intent.type === 'watchlist' ? 'saved' : 'liked'} ${intent.movie.title}, here are related picks.`
                : 'Based on your preferences and viewing history'}
          </p>
        </div>
        <div style={styles.autoBadge}>{refreshing ? 'Updating your picks…' : '✦ Picked for your taste'}</div>
      </div>

      {visibleRecommendations.length === 0 ? (
        <div style={styles.emptyState}>
          <span style={styles.emptyIcon}>🎬</span>
          <h3 style={styles.emptyTitle}>No recommendations yet</h3>
          <p style={styles.emptyText}>
            Start rating movies to get personalized recommendations!
          </p>
        </div>
      ) : (
        <div style={styles.grid}>
          {visibleRecommendations.map(({ rec, matchedGenres }) => (
            <div key={rec.movie.movie_id} className="animate-fadeIn">
              <MovieCard
                movie={{
                  ...rec.movie,
                  score: rec.score,
                  explanation: intent && matchedGenres.length ? `Shares ${matchedGenres.slice(0, 3).join(', ')} with ${intent.movie.title}` : rec.explanation,
                  evidence: intent && matchedGenres.length ? [...(rec.evidence || []), `Similar genres to ${intent.movie.title}: ${matchedGenres.slice(0, 3).join(', ')}`] : rec.evidence,
                }}
                onRate={handleRate}
                onLike={handleLike}
                onWatchlist={handleWatchlist}
                userInteractions={userInteractions}
                showActions={true}
              />
              <button type="button" style={styles.dismissButton} onClick={() => onInteract?.(rec.movie.movie_id, 'dislike', undefined, rec.movie)}>Not for me</button>
            </div>
          ))}
        </div>
      )}
    </section>
  );
}

const styles = {
  container: {
    marginBottom: 'var(--space-8)',
  },
  header: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    marginBottom: 'var(--space-6)',
    gap: 'var(--space-4)',
    flexWrap: 'wrap',
  },
  title: {
    fontSize: 'var(--font-size-3xl)',
    fontWeight: 'var(--font-weight-bold)',
    color: 'var(--color-text)',
    marginBottom: 'var(--space-1)',
  },
  subtitle: {
    fontSize: 'var(--font-size-base)',
    color: 'var(--color-text-muted)',
  },
  methodSelector: {
    display: 'flex',
    alignItems: 'center',
    gap: 'var(--space-2)',
    flexWrap: 'wrap',
  },
  autoBadge: { color: 'var(--color-accent)', fontSize: 'var(--font-size-sm)', fontWeight: 600, padding: '10px 14px', borderRadius: '999px', background: 'rgba(0, 212, 170, 0.1)', border: '1px solid rgba(0, 212, 170, 0.25)' },
  dismissButton: { display: 'block', margin: '8px auto 0', padding: '7px 12px', color: 'var(--color-text-muted)', background: 'transparent', border: '1px solid var(--color-border)', borderRadius: '999px', fontSize: 12, cursor: 'pointer' },
  diversityControl: {
    display: 'flex',
    alignItems: 'center',
    gap: 'var(--space-2)',
    color: 'var(--color-text-muted)',
    fontSize: 'var(--font-size-sm)',
  },
  methodLabel: {
    fontSize: 'var(--font-size-sm)',
    color: 'var(--color-text-muted)',
  },
  methodSelect: {
    padding: 'var(--space-2) var(--space-4)',
    background: 'var(--color-surface)',
    border: '2px solid var(--color-border)',
    borderRadius: 'var(--radius-lg)',
    color: 'var(--color-text)',
    fontSize: 'var(--font-size-sm)',
    cursor: 'pointer',
    minWidth: '200px',
  },
  grid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))',
    gap: 'var(--space-6)',
  },
  loadingContainer: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    minHeight: '400px',
    gap: 'var(--space-4)',
  },
  spinner: {
    width: '48px',
    height: '48px',
    border: '4px solid var(--color-border)',
    borderTopColor: 'var(--color-primary)',
    borderRadius: '50%',
    animation: 'spin 1s linear infinite',
  },
  loadingText: {
    fontSize: 'var(--font-size-base)',
    color: 'var(--color-text-muted)',
  },
  errorContainer: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    minHeight: '400px',
    gap: 'var(--space-4)',
    padding: 'var(--space-8)',
    background: 'var(--color-surface)',
    borderRadius: 'var(--radius-xl)',
  },
  errorIcon: {
    fontSize: '48px',
  },
  errorText: {
    fontSize: 'var(--font-size-lg)',
    color: 'var(--color-text-muted)',
  },
  errorDetail: {
    fontSize: 'var(--font-size-sm)',
    color: 'var(--color-error)',
    fontFamily: 'monospace',
    padding: 'var(--space-3)',
    background: 'var(--color-surface-elevated)',
    borderRadius: 'var(--radius-md)',
    maxWidth: '500px',
    wordBreak: 'break-word',
  },
  retryBtn: {
    padding: 'var(--space-3) var(--space-6)',
    background: 'var(--color-primary)',
    color: 'white',
    borderRadius: 'var(--radius-lg)',
    fontSize: 'var(--font-size-base)',
    fontWeight: 'var(--font-weight-medium)',
    cursor: 'pointer',
  },
  emptyState: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    minHeight: '400px',
    gap: 'var(--space-4)',
    padding: 'var(--space-8)',
    background: 'var(--color-surface)',
    borderRadius: 'var(--radius-xl)',
    textAlign: 'center',
  },
  emptyIcon: {
    fontSize: '64px',
  },
  emptyTitle: {
    fontSize: 'var(--font-size-2xl)',
    fontWeight: 'var(--font-weight-bold)',
    color: 'var(--color-text)',
  },
  emptyText: {
    fontSize: 'var(--font-size-base)',
    color: 'var(--color-text-muted)',
    maxWidth: '400px',
  },
};

export default RecommendationSection;
