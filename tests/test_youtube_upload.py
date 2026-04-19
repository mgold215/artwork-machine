import unittest
from unittest.mock import patch, MagicMock
from artwork_machine.upload.youtube import upload_video, get_youtube_client


class TestGetYoutubeClient(unittest.TestCase):

    @patch('artwork_machine.upload.youtube.build')
    @patch('artwork_machine.upload.youtube.Credentials')
    def test_creates_client_with_refresh_token(self, mock_creds_class, mock_build):
        """Should create a YouTube client using the refresh token from env."""
        mock_creds = MagicMock()
        mock_creds.valid = True
        mock_creds_class.return_value = mock_creds

        with patch.dict('os.environ', {
            'YOUTUBE_REFRESH_TOKEN': 'test_refresh',
            'YOUTUBE_CLIENT_ID': 'test_id',
            'YOUTUBE_CLIENT_SECRET': 'test_secret',
        }):
            client = get_youtube_client()

        mock_build.assert_called_once_with('youtube', 'v3', credentials=mock_creds)
        self.assertEqual(client, mock_build.return_value)


class TestUploadVideo(unittest.TestCase):

    @patch('artwork_machine.upload.youtube.get_youtube_client')
    def test_calls_insert_with_correct_metadata(self, mock_get_client):
        """upload_video should call videos().insert() with proper metadata."""
        mock_yt = MagicMock()
        mock_get_client.return_value = mock_yt

        # Mock the insert chain: videos().insert().next_chunk()
        mock_insert = MagicMock()
        mock_yt.videos.return_value.insert.return_value = mock_insert
        mock_insert.next_chunk.return_value = (None, {'id': 'video123'})

        metadata = {
            'title': 'moodmixformat - TEST (Official Visualizer)',
            'description': 'Test description',
            'tags': ['moodmixformat', 'test'],
            'category': '10',
        }

        # Create a small temp file to simulate video
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            f.write(b'fake video data')
            video_path = f.name

        try:
            result = upload_video(video_path, metadata)
            # Should have called videos().insert()
            mock_yt.videos.return_value.insert.assert_called_once()
            self.assertEqual(result, 'https://youtu.be/video123')
        finally:
            os.unlink(video_path)

    @patch('artwork_machine.upload.youtube.get_youtube_client')
    def test_returns_none_on_missing_token(self, mock_get_client):
        """Should return None if YouTube credentials are not configured."""
        mock_get_client.side_effect = KeyError('YOUTUBE_REFRESH_TOKEN')

        result = upload_video('/fake/path.mp4', {'title': 'test'})
        self.assertIsNone(result)


if __name__ == '__main__':
    unittest.main()
