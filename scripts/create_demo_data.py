"""Demo data generator — creates realistic demo data for testing without full training.

Run this once to populate the database with demo users, interactions, and movies
so the frontend works immediately.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import random
from datetime import datetime, timedelta, timezone

import pandas as pd

from src.api.database import get_db, init_db
from src.api.models import Interaction, User, UserProfile
from src.utils.paths import PROCESSED_DATA_DIR


# Demo user configurations
DEMO_USERS = [
    {"username": "alice", "email": "alice@example.com"},
    {"username": "bob", "email": "bob@example.com"},
    {"username": "carol", "email": "carol@example.com"},
    {"username": "demo", "email": "demo@personalai.dev"},
]

# Predefined popular movie IDs from MovieLens (well-known movies)
DEMO_MOVIES = [
    (2571, "Matrix, The (1999)", ["Action", "Sci-Fi"]),
    (296, "Pulp Fiction (1994)", ["Crime", "Drama"]),
    (593, "Silence of the Lambs, The (1991)", ["Crime", "Horror", "Thriller"]),
    (1210, "Star Wars: Episode IV - A New Hope (1977)", ["Action", "Adventure", "Sci-Fi"]),
    (356, "Forrest Gump (1994)", ["Comedy", "Drama", "Romance"]),
    (260, "Star Wars: Episode V - The Empire Strikes Back (1980)", ["Action", "Adventure", "Sci-Fi"]),
    (589, "Terminator 2: Judgment Day (1991)", ["Action", "Sci-Fi", "Thriller"]),
    (1196, "Star Wars: Episode VI - Return of the Jedi (1983)", ["Action", "Adventure", "Sci-Fi"]),
    (541, "Blade Runner (1982)", ["Action", "Sci-Fi", "Thriller"]),
    (318, "Shawshank Redemption, The (1994)", ["Drama"]),
    (1265, "Groundhog Day (1993)", ["Comedy", "Fantasy", "Romance"]),
    (110, "Braveheart (1995)", ["Action", "Drama", "War"]),
    (590, "Dances with Wolves (1990)", ["Adventure", "Drama", "Western"]),
    (2987, "Amelie (2001)", ["Comedy", "Romance"]),
    (2628, "Princess Bride, The (1987)", ["Adventure", "Comedy", "Fantasy", "Romance"]),
]


def load_movie_metadata():
    """Try to load real movie data from processed files."""
    try:
        movies = pd.read_parquet(PROCESSED_DATA_DIR / "movies.parquet")
        return movies
    except Exception:
        return None


def create_demo_data():
    """Create demo users, profiles, and interactions."""
    init_db()

    movies_df = load_movie_metadata()
    has_real_movies = movies_df is not None

    with get_db() as db:
        # Create users
        user_ids = []
        for user_data in DEMO_USERS:
            existing = db.query(User).filter(User.username == user_data["username"]).first()
            if existing:
                print(f"User {user_data['username']} already exists (id={existing.id})")
                user_ids.append(existing.id)
                continue

            user = User(
                username=user_data["username"],
                email=user_data["email"],
                created_at=datetime.now(timezone.utc) - timedelta(days=random.randint(30, 365)),
            )
            db.add(user)
            db.flush()

            # Create empty profile
            profile = UserProfile(user_id=user.id)
            db.add(profile)

            user_ids.append(user.id)
            print(f"Created user: {user.username} (id={user.id})")

        db.commit()

        # Create interactions for each user
        interaction_types = [
            ("like", 5.0),
            ("watch", 4.5),
            ("complete", 5.0),
            ("view", 3.5),
            ("watchlist", None),
            ("skip", 1.5),
        ]

        for user_id in user_ids:
            # Check existing interactions
            existing_count = db.query(Interaction).filter(Interaction.user_id == user_id).count()
            if existing_count > 0:
                print(f"User {user_id} already has {existing_count} interactions")
                continue

            # Generate 15-30 interactions per user
            n_interactions = random.randint(15, 30)
            selected_movies = random.sample(DEMO_MOVIES, min(n_interactions, len(DEMO_MOVIES)))

            for i, (movie_id, title, genres) in enumerate(selected_movies):
                itype, rating = random.choice(interaction_types)

                # More recent interactions for later items
                days_ago = random.randint(0, 90)
                timestamp = datetime.now(timezone.utc) - timedelta(
                    days=days_ago, hours=random.randint(0, 23)
                )

                interaction = Interaction(
                    user_id=user_id,
                    movie_id=movie_id,
                    interaction_type=itype,
                    rating=rating,
                    timestamp=timestamp,
                )
                db.add(interaction)

            # Update profile with interaction count
            profile = db.query(UserProfile).filter(UserProfile.user_id == user_id).first()
            if profile:
                profile.n_interactions = n_interactions
                profile.avg_rating = round(random.uniform(3.0, 4.5), 1)
                profile.last_updated = datetime.now(timezone.utc)

                # Generate genre affinities based on interacted movies
                genre_affinity = {}
                for _, _, genres in selected_movies:
                    for g in genres:
                        genre_affinity[g] = genre_affinity.get(g, 0) + 1

                # Normalize
                total = sum(genre_affinity.values())
                if total > 0:
                    profile.genre_affinity = {
                        g: round(c / total, 2) for g, c in genre_affinity.items()
                    }

            print(f"Created {n_interactions} interactions for user {user_id}")

        db.commit()
        print("\n✅ Demo data created successfully!")


if __name__ == "__main__":
    create_demo_data()