import json
from unittest.mock import Mock

import pytest
import requests

from chess_self import fetch

PGN = b'[Event "Test"]\n[White "MonisChess"]\n[Black "Other"]\n[Date "2026.01.02"]\n[Result "1-0"]\n\n1. e4 { [%clk 0:01:00] } e5 1-0\n'


def response(content=PGN, status=200, headers=None):
    r = requests.Response()
    r.status_code = status
    r._content = content
    r._content_consumed = True
    r.url = "https://lichess.org/api/games/user/MonisChess"
    r.headers.update(headers or {})
    return r


@pytest.fixture
def http(monkeypatch):
    session = Mock()
    session.headers = {}
    session.__enter__ = Mock(return_value=session)
    session.__exit__ = Mock(return_value=False)
    monkeypatch.setattr(fetch.requests, "Session", lambda: session)
    monkeypatch.setattr(fetch.time, "sleep", Mock())
    return session


def test_download_resume_and_tamper(tmp_path, http):
    http.get.return_value = response()
    result = fetch.fetch_games("lichess", "MonisChess", tmp_path)
    assert result['complete'] and result['total_games'] == 1
    record = result['files'][0]
    assert record['games_with_clocks'] == 1
    assert record['first_game_date'] == '2026.01.02'
    assert http.headers['User-Agent']
    assert http.get.call_args.kwargs['timeout'] == fetch.TIMEOUT
    assert http.get.call_args.kwargs['params'] == dict(clocks='true', evals='false', opening='false', sort='dateAsc')
    assert fetch.fetch_games('lichess', 'MonisChess', tmp_path)['total_games'] == 1
    assert http.get.call_count == 1
    (tmp_path / record['file']).write_bytes(b'altered')
    with pytest.raises(ValueError, match='altered'):
        fetch.fetch_games('lichess', 'MonisChess', tmp_path)


@pytest.mark.parametrize('username', ['../escape', '/tmp/a', 'a/b', '.', 'a?b', 'a\\b', ''])
def test_username_path_safety(tmp_path, username):
    with pytest.raises(ValueError):
        fetch.fetch_games('lichess', username, tmp_path)


@pytest.mark.parametrize('body,headers', [(b'<html>blocked</html>', {}), (b'{"error":"no"}', {}), (PGN, {'Content-Type': 'text/html'}), (b'[Event "Bad"]\n[White "A"]\n[Black "B"]\n\n1. e5 *', {})])
def test_invalid_content_not_persisted(tmp_path, http, body, headers):
    http.get.return_value = response(body, headers=headers)
    result = fetch.fetch_games('lichess', 'MonisChess', tmp_path)
    assert not result['complete'] and result['failures']
    assert not list(tmp_path.glob('*.pgn'))


def test_empty_lichess_export(tmp_path, http):
    http.get.return_value = response(b'')
    result = fetch.fetch_games('lichess', 'MonisChess', tmp_path)
    assert result['complete'] and result['total_games'] == 0
    assert (tmp_path / result['files'][0]['file']).read_bytes() == b''


@pytest.mark.parametrize('status,attempts', [(404, 1), (401, 1), (429, 4), (503, 4), (302, 1)])
def test_http_errors(tmp_path, http, status, attempts):
    http.get.side_effect = [response(b'error', status)] * attempts
    result = fetch.fetch_games('lichess', 'MonisChess', tmp_path)
    assert not result['complete']
    assert http.get.call_count == attempts
    assert not list(tmp_path.glob('*.pgn'))


def test_transient_and_retry_after(http):
    http.get.side_effect = [requests.Timeout(), response(b'', 429, {'Retry-After':'65'}), response()]
    assert fetch._request(http, 'https://lichess.org').content == PGN
    fetch.time.sleep.assert_any_call(65)


def test_month_failure_and_resume(tmp_path, http):
    base = 'https://api.chess.com/pub/player/monisverse/games/'
    index = json.dumps({'archives': [base+'2025/01', base+'2025/02']}).encode()
    http.get.side_effect = [response(index), response(b'no', 404), response()]
    result = fetch.fetch_games('chess.com', 'MonisVerse', tmp_path)
    assert result['total_games'] == 1 and not result['complete']
    assert result['failures'][0]['url'].endswith('2025/01/pgn')
    http.get.side_effect = [response()]
    result = fetch.fetch_games('chess.com', 'MonisVerse', tmp_path)
    assert result['complete'] and result['total_games'] == 2
    assert http.get.call_count == 4


def test_unsafe_archive_url(tmp_path, http):
    http.get.return_value = response(b'{"archives":["https://evil.example/pgn"]}')
    result = fetch.fetch_games('chess.com', 'MonisVerse', tmp_path)
    assert not result['complete'] and http.get.call_count == 1


def test_untracked_export_is_not_overwritten(tmp_path, http):
    path = tmp_path / 'lichess-monischess-all.pgn'
    path.write_bytes(PGN)
    with pytest.raises(FileExistsError):
        fetch.fetch_games('lichess', 'MonisChess', tmp_path)
    assert path.read_bytes() == PGN
    http.get.assert_not_called()


def test_manifest_path_escape(tmp_path, http):
    (tmp_path / 'lichess-monischess.manifest.json').write_text(json.dumps({
        'platform':'lichess', 'username':'MonisChess', 'files':[{'file':'../secret','sha256':'x'}]}))
    with pytest.raises(ValueError, match='Unsafe'):
        fetch.fetch_games('lichess', 'MonisChess', tmp_path)


def test_atomic_collision_preserves_original(tmp_path):
    path = tmp_path / 'file.pgn'
    path.write_bytes(b'original')
    with pytest.raises(FileExistsError):
        fetch._atomic(path, b'new')
    assert path.read_bytes() == b'original'
    assert not list(tmp_path.glob('.download-*'))
