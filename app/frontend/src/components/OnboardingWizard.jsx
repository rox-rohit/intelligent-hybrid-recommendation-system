import { useEffect, useState } from 'react';
import { movieApi, userApi } from '../api';

function OnboardingWizard({ userId, onComplete }) {
  const [genres, setGenres] = useState([]);
  const [selected, setSelected] = useState([]);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    movieApi.getGenres()
      .then((data) => setGenres(data.genres || []))
      .catch((err) => setError(err.message || 'Could not load genres'))
      .finally(() => setLoading(false));
  }, []);

  function toggle(genre) {
    setSelected((current) => current.includes(genre)
      ? current.filter((item) => item !== genre)
      : [...current, genre]);
  }

  async function submit() {
    if (selected.length < 3) {
      setError('Choose at least three genres to personalize your first picks.');
      return;
    }
    setSaving(true);
    setError('');
    try {
      await userApi.onboard(userId, { genres: selected, ratings: [] });
      onComplete?.();
    } catch (err) {
      setError(err.message || 'Could not save your preferences');
    } finally {
      setSaving(false);
    }
  }

  return (
    <section style={styles.panel} aria-labelledby="onboarding-title">
      <p style={styles.eyebrow}>A quick start</p>
      <h2 id="onboarding-title" style={styles.title}>What do you like watching?</h2>
      <p style={styles.copy}>Pick at least three genres. You can change your taste profile as you rate movies.</p>
      {loading ? <p style={styles.copy}>Loading genres…</p> : (
        <div style={styles.genres}>
          {genres.map((genre) => (
            <button key={genre} type="button" onClick={() => toggle(genre)}
              aria-pressed={selected.includes(genre)}
              style={{ ...styles.genre, ...(selected.includes(genre) ? styles.selected : {}) }}>
              {genre}
            </button>
          ))}
        </div>
      )}
      {error && <p role="alert" style={styles.error}>{error}</p>}
      <button type="button" onClick={submit} disabled={loading || saving} style={styles.submit}>
        {saving ? 'Saving…' : `Continue (${selected.length} selected)`}
      </button>
    </section>
  );
}

const styles = {
  panel: { marginBottom: 'var(--space-6)', padding: 'var(--space-6)', background: 'var(--color-surface)', border: '1px solid var(--color-border)', borderRadius: 'var(--radius-xl)' },
  eyebrow: { color: 'var(--color-primary)', fontSize: 'var(--font-size-sm)', fontWeight: 700, textTransform: 'uppercase' },
  title: { color: 'var(--color-text)', fontSize: 'var(--font-size-2xl)', margin: 'var(--space-2) 0' },
  copy: { color: 'var(--color-text-muted)', marginBottom: 'var(--space-4)' },
  genres: { display: 'flex', flexWrap: 'wrap', gap: 'var(--space-2)' },
  genre: { padding: 'var(--space-2) var(--space-4)', borderRadius: '999px', border: '1px solid var(--color-border)', background: 'var(--color-surface-elevated)', color: 'var(--color-text)', cursor: 'pointer' },
  selected: { background: 'var(--color-primary)', color: '#fff', borderColor: 'var(--color-primary)' },
  error: { color: 'var(--color-error)', marginTop: 'var(--space-3)' },
  submit: { marginTop: 'var(--space-5)', padding: 'var(--space-3) var(--space-5)', border: 0, borderRadius: 'var(--radius-lg)', background: 'var(--color-primary)', color: '#fff', fontWeight: 700, cursor: 'pointer' },
};

export default OnboardingWizard;
