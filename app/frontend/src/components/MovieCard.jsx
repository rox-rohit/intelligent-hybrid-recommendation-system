/** Ultra-modern MovieCard with animations, visual feedback, and persistence */

import { useState, useEffect } from 'react';

function MovieCard({ movie, onRate, onLike, onWatchlist, userInteractions = {}, showActions = true }) {
  const [isHovered, setIsHovered] = useState(false);
  const [showRating, setShowRating] = useState(false);
  const [localRating, setLocalRating] = useState(0);
  const [isLiked, setIsLiked] = useState(false);
  const [isWatchlisted, setIsWatchlisted] = useState(false);

  // Load existing interactions from props
  useEffect(() => {
    const interactions = userInteractions[movie.movie_id] || {};
    setLocalRating(interactions.rating || 0);
    setIsLiked(interactions.liked || false);
    setIsWatchlisted(interactions.watchlisted || false);
  }, [movie.movie_id, userInteractions]);

  const fallbackPoster = `data:image/svg+xml,${encodeURIComponent(`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 900"><defs><linearGradient id="g" x2="1" y2="1"><stop stop-color="#312e81"/><stop offset=".52" stop-color="#111827"/><stop offset="1" stop-color="#7f1d1d"/></linearGradient><radialGradient id="r"><stop stop-color="#a78bfa" stop-opacity=".55"/><stop offset="1" stop-color="#a78bfa" stop-opacity="0"/></radialGradient></defs><rect width="600" height="900" fill="url(#g)"/><circle cx="440" cy="250" r="320" fill="url(#r)"/><path d="M0 690 Q190 530 330 700 T600 640 V900 H0Z" fill="#09090b" fill-opacity=".58"/><text x="48" y="100" fill="#ddd6fe" font-family="Arial,sans-serif" font-size="20" letter-spacing="8">PERSONALAI • CINEMA PICK</text><text x="48" y="760" fill="white" font-family="Arial,sans-serif" font-weight="700" font-size="44">${movie.title.replace(/[&<>]/g, '')}</text><text x="48" y="815" fill="#c4b5fd" font-family="Arial,sans-serif" font-size="21">${(movie.genres || []).slice(0, 3).join('  •  ')}</text></svg>`)}`;
  const posterUrl = movie.poster_path ? `https://image.tmdb.org/t/p/w500${movie.poster_path}` : fallbackPoster;

  const handleRate = (stars) => {
    setLocalRating(stars);
    setShowRating(false);
    onRate?.(movie.movie_id, stars, movie);
  };

  const handleLike = () => {
    const newState = !isLiked;
    setIsLiked(newState);
    onLike?.(movie.movie_id, movie);
  };

  const handleWatchlist = () => {
    const newState = !isWatchlisted;
    setIsWatchlisted(newState);
    onWatchlist?.(movie.movie_id, movie);
  };

  return (
    <div
      style={styles.card}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => {
        setIsHovered(false);
        setShowRating(false);
      }}
      className="animate-fadeIn"
    >
      <div style={styles.posterContainer}>
        <img
          src={posterUrl}
          alt={movie.title}
          style={{
            ...styles.poster,
            transform: isHovered ? 'scale(1.05)' : 'scale(1)',
          }}
          onError={(e) => {
            e.target.onerror = null;
            e.target.src = fallbackPoster;
          }}
        />

        {/* Gradient overlay on hover */}
        {isHovered && (
          <div style={styles.overlay} className="animate-fadeIn">
            {!showRating && showActions && (
              <div style={styles.actions}>
                <button
                  onClick={() => setShowRating(true)}
                  style={styles.actionBtn}
                  title="Rate this movie"
                >
                  <span style={styles.actionIcon}>⭐</span>
                  {localRating > 0 && <span style={styles.actionLabel}>{localRating}</span>}
                </button>
                <button
                  onClick={handleLike}
                  style={{
                    ...styles.actionBtn,
                    ...(isLiked ? styles.actionBtnActive : {}),
                  }}
                  title={isLiked ? "Unlike" : "Like"}
                  className={isLiked ? "animate-scaleIn" : ""}
                >
                  <span style={styles.actionIcon}>{isLiked ? '❤️' : '🤍'}</span>
                </button>
                <button
                  onClick={handleWatchlist}
                  style={{
                    ...styles.actionBtn,
                    ...(isWatchlisted ? styles.actionBtnActive : {}),
                  }}
                  title={isWatchlisted ? "Remove from watchlist" : "Add to watchlist"}
                >
                  <span style={styles.actionIcon}>{isWatchlisted ? '✓' : '+'}</span>
                </button>
              </div>
            )}

            {showRating && (
              <div style={styles.ratingPanel} className="animate-scaleIn">
                <div style={styles.ratingTitle}>Rate this movie</div>
                <div style={styles.stars}>
                  {[1, 2, 3, 4, 5].map((star) => (
                    <button
                      key={star}
                      onClick={() => handleRate(star)}
                      onMouseEnter={(e) => {
                        e.currentTarget.style.transform = 'scale(1.3) rotate(15deg)';
                      }}
                      onMouseLeave={(e) => {
                        e.currentTarget.style.transform = 'scale(1)';
                      }}
                      style={{
                        ...styles.star,
                        color: star <= localRating ? '#fbbf24' : '#71717a',
                      }}
                    >
                      {star <= localRating ? '★' : '☆'}
                    </button>
                  ))}
                </div>
                <div style={styles.ratingHint}>
                  {localRating === 0 ? 'Click a star' : `${localRating}/5 stars`}
                </div>
              </div>
            )}
          </div>
        )}

        {/* Year badge */}
        {movie.release_year && (
          <div style={styles.yearBadge}>{movie.release_year}</div>
        )}

        {/* Status indicators */}
        <div style={styles.statusIndicators}>
          {localRating > 0 && (
            <div style={styles.statusBadge} title={`Rated ${localRating}/5`}>
              ⭐ {localRating}
            </div>
          )}
          {isLiked && (
            <div style={{...styles.statusBadge, background: 'rgba(239, 68, 68, 0.9)'}} title="Liked">
              ❤️
            </div>
          )}
          {isWatchlisted && (
            <div style={{...styles.statusBadge, background: 'rgba(0, 212, 170, 0.9)'}} title="In Watchlist">
              ✓
            </div>
          )}
        </div>
      </div>

      <div style={styles.info}>
        <h3 style={styles.title}>{movie.title}</h3>
        <div style={styles.genres}>
          {movie.genres?.slice(0, 3).map((genre, idx) => (
            <span key={genre} style={{
              ...styles.genre,
              animationDelay: `${idx * 0.05}s`,
            }} className="animate-fadeIn">
              {genre}
            </span>
          ))}
        </div>
        {movie.score && (
          <div style={styles.scoreContainer}>
            <div style={styles.scoreBar}>
              <div
                style={{
                  ...styles.scoreBarFill,
                  width: `${Math.round(movie.score * 100)}%`,
                }}
              />
            </div>
            <span style={styles.scoreText}>{Math.round(movie.score * 100)}% Match</span>
          </div>
        )}
        {movie.explanation && (
          <p style={styles.explanation}>{movie.explanation}</p>
        )}
        {movie.evidence?.length > 0 && (
          <details style={styles.evidence}>
            <summary>Why this?</summary>
            <ul style={styles.evidenceList}>
              {movie.evidence.map((item) => <li key={item}>{item}</li>)}
            </ul>
          </details>
        )}
      </div>
    </div>
  );
}

const styles = {
  card: {
    background: 'var(--color-surface)',
    borderRadius: 'var(--radius-lg)',
    overflow: 'hidden',
    transition: 'all var(--transition-normal)',
    cursor: 'pointer',
    height: '100%',
    display: 'flex',
    flexDirection: 'column',
    border: '1px solid var(--color-border)',
  },
  posterContainer: {
    position: 'relative',
    aspectRatio: '2/3',
    background: 'var(--color-surface-elevated)',
    overflow: 'hidden',
  },
  poster: {
    width: '100%',
    height: '100%',
    objectFit: 'cover',
    transition: 'transform var(--transition-slow)',
  },
  overlay: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    background: 'linear-gradient(to top, rgba(0,0,0,0.95) 0%, rgba(0,0,0,0.6) 40%, rgba(0,0,0,0.3) 100%)',
    display: 'flex',
    flexDirection: 'column',
    justifyContent: 'flex-end',
    padding: 'var(--space-4)',
  },
  actions: {
    display: 'flex',
    gap: 'var(--space-2)',
    justifyContent: 'center',
  },
  actionBtn: {
    minWidth: '56px',
    height: '56px',
    borderRadius: 'var(--radius-full)',
    background: 'rgba(20, 20, 22, 0.95)',
    backdropFilter: 'blur(20px)',
    border: '2px solid rgba(255, 255, 255, 0.15)',
    color: 'white',
    fontSize: '24px',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    flexDirection: 'column',
    gap: '2px',
    cursor: 'pointer',
    transition: 'all var(--transition-fast)',
    boxShadow: 'var(--shadow-lg)',
  },
  actionBtnActive: {
    background: 'rgba(229, 9, 20, 0.95)',
    border: '2px solid rgba(229, 9, 20, 1)',
    boxShadow: '0 0 24px var(--color-primary-glow)',
  },
  actionIcon: {
    fontSize: '24px',
  },
  actionLabel: {
    fontSize: '11px',
    fontWeight: 'var(--font-weight-bold)',
    marginTop: '-4px',
  },
  ratingPanel: {
    background: 'rgba(20, 20, 22, 0.98)',
    backdropFilter: 'blur(40px)',
    borderRadius: 'var(--radius-lg)',
    padding: 'var(--space-4)',
    border: '2px solid var(--color-border-strong)',
    boxShadow: 'var(--shadow-xl)',
  },
  ratingTitle: {
    fontSize: 'var(--font-size-sm)',
    fontWeight: 'var(--font-weight-semibold)',
    color: 'var(--color-text)',
    marginBottom: 'var(--space-3)',
    textAlign: 'center',
  },
  stars: {
    display: 'flex',
    gap: 'var(--space-2)',
    justifyContent: 'center',
    marginBottom: 'var(--space-2)',
  },
  star: {
    background: 'none',
    border: 'none',
    fontSize: '32px',
    cursor: 'pointer',
    padding: 'var(--space-1)',
    transition: 'all var(--transition-fast)',
  },
  ratingHint: {
    fontSize: 'var(--font-size-xs)',
    color: 'var(--color-text-muted)',
    textAlign: 'center',
  },
  yearBadge: {
    position: 'absolute',
    top: 'var(--space-2)',
    right: 'var(--space-2)',
    background: 'rgba(0, 0, 0, 0.9)',
    backdropFilter: 'blur(10px)',
    color: 'white',
    padding: '6px 12px',
    borderRadius: 'var(--radius-md)',
    fontSize: 'var(--font-size-xs)',
    fontWeight: 'var(--font-weight-bold)',
    border: '1px solid rgba(255, 255, 255, 0.1)',
  },
  statusIndicators: {
    position: 'absolute',
    top: 'var(--space-2)',
    left: 'var(--space-2)',
    display: 'flex',
    flexDirection: 'column',
    gap: 'var(--space-1)',
  },
  statusBadge: {
    background: 'rgba(0, 0, 0, 0.9)',
    backdropFilter: 'blur(10px)',
    color: 'white',
    padding: '4px 8px',
    borderRadius: 'var(--radius-md)',
    fontSize: 'var(--font-size-xs)',
    fontWeight: 'var(--font-weight-bold)',
    display: 'flex',
    alignItems: 'center',
    gap: '4px',
    border: '1px solid rgba(255, 255, 255, 0.2)',
  },
  info: {
    padding: 'var(--space-4)',
    flex: 1,
    display: 'flex',
    flexDirection: 'column',
    gap: 'var(--space-2)',
  },
  title: {
    fontSize: 'var(--font-size-base)',
    fontWeight: 'var(--font-weight-semibold)',
    color: 'var(--color-text)',
    lineHeight: '1.3',
    display: '-webkit-box',
    WebkitLineClamp: 2,
    WebkitBoxOrient: 'vertical',
    overflow: 'hidden',
    textOverflow: 'ellipsis',
  },
  genres: {
    display: 'flex',
    flexWrap: 'wrap',
    gap: 'var(--space-1)',
  },
  genre: {
    fontSize: 'var(--font-size-xs)',
    color: 'var(--color-text-muted)',
    background: 'var(--color-surface-elevated)',
    padding: '3px 8px',
    borderRadius: 'var(--radius-sm)',
    border: '1px solid var(--color-border)',
  },
  scoreContainer: {
    display: 'flex',
    flexDirection: 'column',
    gap: 'var(--space-1)',
  },
  scoreBar: {
    height: '4px',
    background: 'var(--color-border)',
    borderRadius: 'var(--radius-full)',
    overflow: 'hidden',
  },
  scoreBarFill: {
    height: '100%',
    background: 'var(--gradient-accent)',
    borderRadius: 'var(--radius-full)',
    transition: 'width var(--transition-slow)',
  },
  scoreText: {
    fontSize: 'var(--font-size-xs)',
    fontWeight: 'var(--font-weight-semibold)',
    color: 'var(--color-accent)',
  },
  explanation: {
    fontSize: 'var(--font-size-sm)',
    color: 'var(--color-text-subtle)',
    marginTop: 'auto',
    display: '-webkit-box',
    WebkitLineClamp: 2,
    WebkitBoxOrient: 'vertical',
    overflow: 'hidden',
    fontStyle: 'italic',
  },
  evidence: {
    marginTop: 'var(--space-2)',
    color: 'var(--color-text-muted)',
    fontSize: 'var(--font-size-xs)',
  },
  evidenceList: {
    paddingLeft: 'var(--space-4)',
    marginTop: 'var(--space-2)',
    display: 'grid',
    gap: 'var(--space-1)',
  },
};

export default MovieCard;
