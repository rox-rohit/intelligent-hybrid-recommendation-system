import { useState, useEffect, useCallback } from 'react';
import { healthApi, interactionApi, movieApi, userApi } from './api';
import UserProfile from './components/UserProfile';
import RecommendationSection from './components/RecommendationSection';
import OnboardingWizard from './components/OnboardingWizard';
import MovieSearch from './components/MovieSearch';
import './App.css';

function App() {
  const [userId, setUserId] = useState(null);
  const [health, setHealth] = useState(null);
  const [loading, setLoading] = useState(true);
  const [userInteractions, setUserInteractions] = useState({});
  const [interactionVersion, setInteractionVersion] = useState(0);
  const [recommendationIntent, setRecommendationIntent] = useState(null);
  const [userProfile, setUserProfile] = useState(null);

  useEffect(() => {
    checkHealth();
    const savedUserId = localStorage.getItem('personalai_user_id');
    if (savedUserId) {
      const id = parseInt(savedUserId, 10);
      setUserId(id);
      loadUserInteractions(id);
    }
  }, []);

  useEffect(() => {
    if (!userId) {
      setUserProfile(null);
      return;
    }
    userApi.getProfile(userId).then(setUserProfile).catch(() => setUserProfile(null));
  }, [userId]);

  async function checkHealth() {
    try {
      const data = await healthApi.check();
      setHealth(data);
    } catch (err) {
      console.warn('Health check failed:', err);
    } finally {
      setLoading(false);
    }
  }

  async function loadUserInteractions(uid) {
    try {
      const data = await userApi.getInteractions(uid);
      const interactionsMap = {};

      const rows = data.interactions || [];
      if (rows.some((item) => item.interaction_type)) {
        for (const item of [...rows].reverse()) {
          const mid = item.movie_id;
          const state = interactionsMap[mid] || { movie_id: mid, liked: false, watchlisted: false, rating: null, dismissed: false };
          if (item.interaction_type === 'like') state.liked = true;
          if (item.interaction_type === 'dislike') { state.liked = false; state.dismissed = true; }
          if (item.interaction_type === 'watchlist') state.watchlisted = true;
          if (item.interaction_type === 'remove_watchlist') state.watchlisted = false;
          if (item.rating != null) state.rating = item.rating;
          interactionsMap[mid] = state;
        }
      } else {
        rows.forEach((item) => { interactionsMap[item.movie_id] = item; });
      }

      await Promise.all(Object.values(interactionsMap).map(async (item) => {
        if (item.title || !item.movie_id) return;
        try {
          const movie = await movieApi.get(item.movie_id);
          Object.assign(item, movie);
        } catch {
          item.title = `Movie ${item.movie_id}`;
        }
      }));

      setUserInteractions(interactionsMap);
    } catch (err) {
      console.error('Failed to load interactions:', err);
    }
  }

  function handleUserChange(newUserId) {
    setUserId(newUserId);
    setUserInteractions({});
    setInteractionVersion((version) => version + 1);
    if (newUserId) {
      localStorage.setItem('personalai_user_id', newUserId.toString());
      loadUserInteractions(newUserId);
    } else {
      localStorage.removeItem('personalai_user_id');
      setUserInteractions({});
    }
  }

  const handleInteract = useCallback(async (movieId, interactionType, rating, movie) => {
    if (!userId) return;

    const previous = userInteractions[movieId] || {};
    const movieDetails = movie || previous;
    let persistedType = interactionType;
    if (interactionType === 'like') persistedType = previous.liked ? 'dislike' : 'like';
    if (interactionType === 'watchlist') persistedType = previous.watchlisted ? 'remove_watchlist' : 'watchlist';
    setUserInteractions((current) => ({
      ...current,
      [movieId]: {
        ...current[movieId],
        movie_id: movieId,
        title: movieDetails?.title || current[movieId]?.title,
        genres: movieDetails?.genres || current[movieId]?.genres || [],
        liked: interactionType === 'like' ? !previous.liked : previous.liked || false,
        watchlisted: interactionType === 'watchlist' ? !previous.watchlisted : previous.watchlisted || false,
        dismissed: interactionType === 'dislike' || (interactionType === 'like' && previous.liked),
        rating: interactionType === 'rate' ? rating : current[movieId]?.rating,
      },
    }));
    setRecommendationIntent({
      movie: { movie_id: movieId, title: movieDetails?.title || 'that title', genres: movieDetails?.genres || [] },
      type: interactionType,
      rating,
      timestamp: Date.now(),
    });
    window.setTimeout(() => document.getElementById('recommendations')?.scrollIntoView({ behavior: 'smooth', block: 'start' }), 80);

    try {
      await interactionApi.create(userId, {
        movie_id: movieId,
        interaction_type: persistedType,
        rating,
      });
      loadUserInteractions(userId);
      userApi.getProfile(userId).then(setUserProfile).catch(() => {});
      setInteractionVersion((version) => version + 1);
    } catch (err) {
      console.error('Interaction failed:', err);
      loadUserInteractions(userId);
    }
  }, [userId, userInteractions]);

  if (loading) {
    return (
      <div className="loading-screen">
        <div className="loading-content">
          <div className="loading-icon">🎬</div>
          <div className="spinner"></div>
          <h2>PersonalAI</h2>
          <p>Loading your cinema companion...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="app">
      {/* Header */}
      <header className="header glass">
        <div className="header-content">
          <div className="logo">
            <span className="logo-icon">🎬</span>
            <div className="logo-text">
              <h1>PersonalAI</h1>
              <span>Smart Recommendations</span>
            </div>
          </div>

          <div className={`status-badge ${health?.status === 'healthy' ? 'online' : 'demo'}`}>
            <span className="dot">●</span>
            {health?.status === 'healthy' ? 'Online' : 'Demo Mode'}
          </div>
        </div>
      </header>

      {/* Main Layout */}
      <div className="layout">
        {/* Sidebar */}
        <aside className="sidebar">
          <UserProfile
            userId={userId}
            onUserChange={handleUserChange}
            userInteractions={userInteractions}
          />

          {health && (
            <div className="system-status glass fade-in">
              <h3>System Status</h3>
              <div className="status-list">
                {Object.entries(health.models_loaded || {}).map(([key, loaded]) => (
                  <div key={key} className="status-item">
                    <span className={`indicator ${loaded ? 'active' : ''}`}></span>
                    <span>{formatModelName(key)}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </aside>

        {/* Main Content */}
        <main className="main-content">
          {userId ? (
            <>
              <MovieSearch
                userInteractions={userInteractions}
                onRate={(movieId, rating, movie) => handleInteract(movieId, 'rate', rating, movie)}
                onLike={(movieId, movie) => handleInteract(movieId, 'like', undefined, movie)}
                onWatchlist={(movieId, movie) => handleInteract(movieId, 'watchlist', undefined, movie)}
              />

              {userProfile && !userProfile.genre_affinity && userProfile.n_interactions === 0 && (
                <OnboardingWizard userId={userId} onComplete={() => userApi.getProfile(userId).then(setUserProfile)} />
              )}

              <RecommendationSection
                userId={userId}
                onInteract={handleInteract}
                userInteractions={userInteractions}
                interactionVersion={interactionVersion}
                intent={recommendationIntent}
              />
            </>
          ) : (
            <div className="hero fade-in">
              <div className="hero-icon">🎬</div>
              <h1 className="hero-title">
                Discover Movies<br />
                <span className="gradient">You'll Actually Love</span>
              </h1>
              <p className="hero-text">
                AI-powered recommendations that learn your unique taste.
                Rate movies, get smarter suggestions.
              </p>

              <div className="features">
                <div className="feature">
                  <span className="feature-icon">🎯</span>
                  <div>
                    <strong>Smart Matching</strong>
                    <p>AI learns your taste</p>
                  </div>
                </div>
                <div className="feature">
                  <span className="feature-icon">⚡</span>
                  <div>
                    <strong>Instant Results</strong>
                    <p>Real-time recommendations</p>
                  </div>
                </div>
                <div className="feature">
                  <span className="feature-icon">🔥</span>
                  <div>
                    <strong>Always Learning</strong>
                    <p>Better with each rating</p>
                  </div>
                </div>
                <div className="feature">
                  <span className="feature-icon">💎</span>
                  <div>
                    <strong>Truly Personal</strong>
                    <p>No generic lists</p>
                  </div>
                </div>
              </div>

              <div className="cta glass">
                <div className="cta-arrow">👈</div>
                <div>
                  <strong>Ready to start?</strong>
                  <p>Sign in or create an account in the sidebar</p>
                </div>
              </div>
            </div>
          )}
        </main>
      </div>

      {/* Footer */}
      <footer className="footer">
        <p>PersonalAI v0.2.0 • Powered by MovieLens 25M</p>
      </footer>
    </div>
  );
}

function formatModelName(key) {
  const names = {
    ranker: 'Learned Ranker',
    hybrid_tuning: 'Hybrid Model',
    processed_data: 'MovieLens Data',
  };
  return names[key] || key;
}

export default App;
