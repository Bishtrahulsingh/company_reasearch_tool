import asyncio

import httpx
import trafilatura

from app.config.settings import settings

url = "https://google.serper.dev/search"

async def extract_from_url(client, site_url:str):
    try:
        response = await client.get(site_url, timeout=10)
        text = trafilatura.extract(response.text)
        return text
    except Exception:
        return None

async def search_web(query, company):
    payload = {
        "q": f"{query} for {company}"
    }
    headers = {
        'X-API-KEY': settings.X_API_KEY,
        'Content-Type': 'application/json'
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(url, json=payload, headers=headers)
        data = response.json()
        summary = data.get('answerBox',{}).get('snippet')
        organic = data.get('organic',[])

        items = []
        for item in organic:
            items.append({
                'title': item.get('title'),
                'text': None,
                'url' : item.get('link'),
                'published_at': item.get('date'),
                'company' : company,
            })

        tasks = [extract_from_url(client,item.get('url')) for item in items]
        contents = await asyncio.gather(*tasks)

        for item,content in zip(items,contents):
            item['text'] = content
            print(item,end="\n-------------------\n")

        return {
            'summary': summary,
            'items':items
        }