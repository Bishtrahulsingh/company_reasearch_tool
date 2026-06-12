import asyncio
import datetime
from typing import Dict

import httpx
import trafilatura

from app.config.settings import settings
from app.core.models import WebSearchResult

url = "https://google.serper.dev/search"

def parse_date(date_str):
    if not date_str:
        return None
    for fmt in ("%b %d, %Y", "%Y-%m-%d", "%B %d, %Y"):
        try:
            return datetime.datetime.strptime(date_str, fmt).date()
        except ValueError:
            continue
    return None

async def extract_from_url(client, site_url:str):
    try:
        response = await client.get(site_url, timeout=8)
        text = trafilatura.extract(response.text)
        return text
    except Exception:
        return None

async def search_web(query, company)->WebSearchResult:
    payload = {
        "q": f"{query} for {company}"
    }

    headers = {
        'X-API-KEY': settings.X_API_KEY,
        'Content-Type': 'application/json'
    }

    async with httpx.AsyncClient(timeout=15) as client:
        response = await client.post(url, json=payload, headers=headers)
        data = response.json()
        summary = data.get('answerBox',{}).get('snippet','')
        organic = data.get('organic',[])

        items = []
        for item in organic[:2]:
            items.append({
                'title': item.get('title'),
                'text': None,
                'url' : item.get('link'),
                'published_at': parse_date(item.get('date')),
                'company' : company,
            })

        tasks = [extract_from_url(client,item.get('url')) for item in items]
        try:
            contents = await asyncio.wait_for(asyncio.gather(*tasks), timeout=12)
        except asyncio.TimeoutError:
            contents = [None] * len(tasks)

        for item,content in zip(items,contents):
            item['text'] = content

        return WebSearchResult(summary= summary,items= items)