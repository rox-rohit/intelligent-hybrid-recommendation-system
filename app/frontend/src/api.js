/** API client for PersonalAI backend */

const API_BASE = '/api';

class ApiError extends Error {
  constructor(message, status, data) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.data = data;
  }
}

async function request(endpoint, options = {}) {
  const url = `${API_BASE}${endpoint}`;
  const config = {
    headers: {
      'Content-Type': 'application/json',
      ...options.headers,
    },
    ...options,
  };

  if (config.body && typeof config.body === 'object') {
    config.body = JSON.stringify(config.body);
  }

  const response = await fetch(url, config);

  if (!response.ok) {
    let errorData;
    try {
      errorData = await response.json();
    } catch {
      errorData = { detail: response.statusText };
    }
    throw new ApiError(
      errorData.detail || `HTTP ${response.status}`,
      response.status,
      errorData
    );
  }

  if (response.status === 204) {
    return null;
  }

  return response.json();
}

// ── User API ──────────────────────────────────────────────────────────────

export const userApi = {
  create: (username, email) => request('/users', {
    method: 'POST',
    body: { username, email },
  }),

  login: (username) => request(`/users/login?username=${encodeURIComponent(username)}`, {
    method: 'POST',
  }),

  get: (userId) => request(`/users/${userId}`),

  getProfile: (userId) => request(`/users/${userId}/profile`),
  onboard: (userId, data) => request(`/users/${userId}/onboarding`, { method: 'POST', body: data }),

  getInteractions: (userId, limit = 500) => request(`/users/${userId}/interactions/detail?limit=${limit}`),
};

// ── Interaction API ───────────────────────────────────────────────────────

export const interactionApi = {
  create: (userId, interaction) => request(`/users/${userId}/interactions`, {
    method: 'POST',
    body: interaction,
  }),

  list: (userId, limit = 50) => request(`/users/${userId}/interactions?limit=${limit}`),
};

// ── Recommendation API ────────────────────────────────────────────────────

export const recommendationApi = {
  get: (userId, { k = 10, method = 'auto', mode = 'personalized', diversity = 1 } = {}) =>
    request(`/users/${userId}/recommendations?k=${k}&method=${method}&mode=${mode}&diversity=${diversity}`),
};

// ── Movie Search API ──────────────────────────────────────────────────────

export const movieApi = {
  search: (params = {}) => {
    const searchParams = new URLSearchParams();
    if (params.query) searchParams.set('query', params.query);
    if (params.genres?.length) searchParams.set('genres', params.genres.join(','));
    if (params.minYear) searchParams.set('min_year', params.minYear);
    if (params.maxYear) searchParams.set('max_year', params.maxYear);
    if (params.minRating) searchParams.set('min_rating', params.minRating);
    if (params.limit) searchParams.set('limit', params.limit);
    return request(`/movies/search?${searchParams.toString()}`);
  },

  get: (movieId) => request(`/movies/${movieId}`),

  getGenres: () => request('/genres'),
};

// ── Health API ────────────────────────────────────────────────────────────

export const healthApi = {
  check: () => request('/health'),
};

export { ApiError };
