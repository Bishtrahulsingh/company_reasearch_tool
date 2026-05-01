import datetime
import httpx

from app.config.settings import settings

GITHUB_TOKEN = settings.GITHUB_TOKEN

headers = {
    "Authorization": f"Bearer {GITHUB_TOKEN}",
    "Accept": "application/vnd.github+json"
}


async def get_star_count(url):
    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers)
        return response.json().get("stargazers_count", 0)

async def get_recent_releases(url):
    url = f"{url}/releases"
    async with httpx.AsyncClient(headers=headers) as client:
        response = await client.get(url)
        return response.json()

async def get_recent_commits(url,params):
    url =f"{url}/commits"
    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers,params=params)
        return response.json()

async def get_github_activity(repo, since_days:int=7)->str:
    owner, name = repo.split("/")
    since = (datetime.datetime.utcnow() - datetime.timedelta(days=since_days)).isoformat()
    url = f"https://api.github.com/repos/{owner}/{name}"

    params = {
        "since": since,
        "per_page": 5
    }

    commits = await get_recent_commits(url,params)
    releases = await get_recent_releases(url)
    stars = await get_star_count(url)

    lines = [f'stars: {stars}.']

    for c in commits[:3]:
        msg = c["commit"]["message"].split("\n")[0]
        date = c["commit"]["author"]["date"][:10]
        lines.append(f"On {date}, commit: {msg}.")

    for r in releases[:2]:
        name = r.get("name") or r.get("tag_name")
        date = r.get("published_at", "")[:10]
        if date:
            lines.append(f"Release {name} published on {date}.")

    return " ".join(lines)