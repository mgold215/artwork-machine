import unittest
from unittest.mock import MagicMock
from artwork_machine.upload.metadata import generate_metadata


class TestGenerateMetadata(unittest.TestCase):

    def _make_direction(self):
        """Create a mock CreativeDirection with typical values."""
        d = MagicMock()
        d.tagline = "Feel the future"
        d.art_style = "Brutalist neon"
        return d

    def _make_features(self):
        """Create a mock AudioFeatures with typical values."""
        f = MagicMock()
        f.bpm = 126.0
        f.key = "F minor"
        f.mood_tags = ["energetic", "upbeat"]
        f.genre_hints = ["future house", "melodic house"]
        f.duration = 210.0
        return f

    def test_title_format(self):
        """Title should be: artist - album (Official Visualizer)."""
        meta = generate_metadata(
            artist="moodmixformat",
            album="SIDEWAYZ",
            direction=self._make_direction(),
            features=self._make_features(),
        )
        self.assertEqual(meta['title'], "moodmixformat - SIDEWAYZ (Official Visualizer)")

    def test_description_contains_spotify_link(self):
        """Description should include the Spotify artist link."""
        meta = generate_metadata(
            artist="moodmixformat",
            album="SIDEWAYZ",
            direction=self._make_direction(),
            features=self._make_features(),
        )
        self.assertIn("open.spotify.com", meta['description'])

    def test_description_contains_linktree(self):
        """Description should include the Linktree link."""
        meta = generate_metadata(
            artist="moodmixformat",
            album="SIDEWAYZ",
            direction=self._make_direction(),
            features=self._make_features(),
        )
        self.assertIn("linktr.ee/moodmixformat", meta['description'])

    def test_tags_include_artist_and_genre(self):
        """Tags should include artist name and genre hints."""
        meta = generate_metadata(
            artist="moodmixformat",
            album="SIDEWAYZ",
            direction=self._make_direction(),
            features=self._make_features(),
        )
        self.assertIn("moodmixformat", meta['tags'])
        self.assertIn("future house", meta['tags'])

    def test_tags_max_length(self):
        """Total tag string should not exceed 500 characters."""
        meta = generate_metadata(
            artist="moodmixformat",
            album="SIDEWAYZ",
            direction=self._make_direction(),
            features=self._make_features(),
        )
        total_chars = sum(len(t) for t in meta['tags'])
        self.assertLessEqual(total_chars, 500)

    def test_category_is_music(self):
        """Category should be '10' (Music)."""
        meta = generate_metadata(
            artist="moodmixformat",
            album="SIDEWAYZ",
            direction=self._make_direction(),
            features=self._make_features(),
        )
        self.assertEqual(meta['category'], '10')

    def test_description_contains_bpm_and_key(self):
        """Description should include BPM and key from audio features."""
        meta = generate_metadata(
            artist="moodmixformat",
            album="SIDEWAYZ",
            direction=self._make_direction(),
            features=self._make_features(),
        )
        self.assertIn("126", meta['description'])
        self.assertIn("F minor", meta['description'])


if __name__ == '__main__':
    unittest.main()
