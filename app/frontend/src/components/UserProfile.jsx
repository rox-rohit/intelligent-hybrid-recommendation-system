/** User Profile — Login/Register + logged-in profile card */

import { useState, useEffect } from 'react';
import { userApi } from '../api';
import './UserProfile.css';

function UserProfile({ userId, onUserChange, userInteractions = {} }) {
  const [user, setUser] = useState(null);
  const [profile, setProfile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [mode, setMode] = useState('login'); // 'login' or 'register'
  const [username, setUsername] = useState('');
  const [email, setEmail] = useState('');
  const [error, setError] = useState('');
  const [activityTab, setActivityTab] = useState('liked');

  useEffect(() => {
    if (userId) {
      loadUser();
    }
  }, [userId]);

  async function loadUser() {
    setLoading(true);
    try {
      const [userData, profileData] = await Promise.all([
        userApi.get(userId),
        userApi.getProfile(userId).catch(() => null),
      ]);
      setUser(userData);
      setProfile(profileData);
    } catch (err) {
      console.error('Failed to load user:', err);
    } finally {
      setLoading(false);
    }
  }

  async function handleLogin(e) {
    e.preventDefault();
    if (!username.trim()) return;

    setLoading(true);
    setError('');
    try {
      const user = await userApi.login(username);
      setUser(user);
      setUsername('');
      onUserChange?.(user.id);
    } catch (err) {
      if (err.status === 404) {
        setError(`User "${username}" not found. Please register first.`);
      } else {
        setError(err.message || 'Login failed');
      }
    } finally {
      setLoading(false);
    }
  }

  async function handleRegister(e) {
    e.preventDefault();
    if (!username.trim()) return;

    setLoading(true);
    setError('');
    try {
      const newUser = await userApi.create(username, email || undefined);
      setUser(newUser);
      setUsername('');
      setEmail('');
      onUserChange?.(newUser.id);
    } catch (err) {
      if (err.status === 409) {
        setError(`Username "${username}" already exists. Try logging in instead.`);
      } else {
        setError(err.message || 'Registration failed');
      }
    } finally {
      setLoading(false);
    }
  }

  function handleLogout() {
    setUser(null);
    setProfile(null);
    onUserChange?.(null);
    localStorage.removeItem('personalai_user_id');
  }

  if (!user) {
    return (
      <div className="auth-card glass slide-up">
        <div className="auth-logo">🎬</div>
        <h2 className="auth-title">Welcome to PersonalAI</h2>
        <p className="auth-subtitle">Your personal movie recommendation assistant</p>

        <div className="auth-tabs">
          <button
            type="button"
            onClick={() => { setMode('login'); setError(''); }}
            className={`auth-tab ${mode === 'login' ? 'active' : ''}`}
          >
            Login
          </button>
          <button
            type="button"
            onClick={() => { setMode('register'); setError(''); }}
            className={`auth-tab ${mode === 'register' ? 'active' : ''}`}
          >
            Register
          </button>
        </div>

        <form onSubmit={mode === 'login' ? handleLogin : handleRegister} className="auth-form">
          <div className="auth-field">
            <label htmlFor="username">Username</label>
            <input
              id="username"
              type="text"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              placeholder="Enter your username"
              autoComplete="username"
              required
              autoFocus
            />
          </div>

          {mode === 'register' && (
            <div className="auth-field">
              <label htmlFor="email">Email (optional)</label>
              <input
                id="email"
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="your@email.com"
                autoComplete="email"
              />
            </div>
          )}

          {error && (
            <div className="auth-error">
              <span>⚠️</span>
              <span>{error}</span>
            </div>
          )}

          <button
            type="submit"
            disabled={loading || !username.trim()}
            className="auth-submit"
          >
            {loading ? (
              <>
                <span className="btn-spinner" />
                {mode === 'login' ? 'Logging in...' : 'Creating account...'}
              </>
            ) : (
              mode === 'login' ? 'Login' : 'Create Account'
            )}
          </button>
        </form>

        <div className="auth-features">
          <div className="auth-feature">
            <span>🎯</span>
            <span>Personalized Recommendations</span>
          </div>
          <div className="auth-feature">
            <span>⭐</span>
            <span>Rate & Discover Movies</span>
          </div>
          <div className="auth-feature">
            <span>🤖</span>
            <span>AI-Powered Matching</span>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="profile-card glass slide-up">
      <div className="profile-header">
        <div className="profile-avatar">
          {user.username.charAt(0).toUpperCase()}
        </div>
        <div className="profile-info">
          <h2 className="profile-username">{user.username}</h2>
          <span className="profile-id">ID: {user.id}</span>
        </div>
        <button onClick={handleLogout} className="logout-btn" title="Logout">
          🚪
        </button>
      </div>

      {profile && (
        <div className="profile-stats">
          <div className="stat">
            <div className="stat-value">{profile.n_interactions}</div>
            <div className="stat-label">Interactions</div>
          </div>
          <div className="stat">
            <div className="stat-value">
              {profile.avg_rating ? profile.avg_rating.toFixed(1) : '—'}
            </div>
            <div className="stat-label">Avg Rating</div>
          </div>
          <div className="stat">
            <div className="stat-value">
              {profile.genre_affinity ? Object.keys(profile.genre_affinity).length : 0}
            </div>
            <div className="stat-label">Genres</div>
          </div>
        </div>
      )}

      {profile?.genre_affinity && Object.keys(profile.genre_affinity).length > 0 && (
        <div className="genres-section">
          <h3>Top Genres</h3>
          <div className="genres-list">
            {Object.entries(profile.genre_affinity)
              .sort((a, b) => b[1] - a[1])
              .slice(0, 5)
              .map(([genre, score]) => (
                <div key={genre} className="genre-item">
                  <span className="genre-name">{genre}</span>
                  <div className="genre-bar">
                    <div
                      className="genre-fill"
                      style={{ width: `${Math.min(score * 100, 100)}%` }}
                    />
                  </div>
                </div>
              ))}
          </div>
        </div>
      )}

      <section className="profile-activity">
        <h3>Your collection</h3>
        <div className="activity-tabs">
          {[['liked', 'Liked'], ['rated', 'Rated'], ['watchlisted', 'Watchlist']].map(([key, label]) => (
            <button type="button" key={key} className={activityTab === key ? 'active' : ''} onClick={() => setActivityTab(key)}>{label}</button>
          ))}
        </div>
        <div className="activity-list">
          {Object.values(userInteractions).filter((item) => activityTab === 'rated' ? item.rating != null : item[activityTab]).map((item) => (
            <div className="activity-item" key={item.movie_id}>
              <span className="activity-title">{item.title || `Movie ${item.movie_id}`}</span>
              {activityTab === 'rated' && <span className="activity-rating">★ {item.rating}/5</span>}
            </div>
          ))}
          {!Object.values(userInteractions).some((item) => activityTab === 'rated' ? item.rating != null : item[activityTab]) && <p className="activity-empty">Nothing here yet. Explore a title and add it to your collection.</p>}
        </div>
      </section>
    </div>
  );
}

export default UserProfile;
