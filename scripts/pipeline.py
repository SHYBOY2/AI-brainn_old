#!/usr/bin/env python3
"""
pipeline.py — runs once per GitHub Actions cron tick (every 6 hours).
Handles:
  1. Pulling Discord messages (Voice, Links, Text, Journals, and Queries).
  2. Resolving YouTube transcripts and web article contents.
  3. Journaling directly to journal/ directory.
  4. Answering queries grounded in wiki/ knowledge and posting replies back to Discord.
  5. Summarizing raw files into structured wiki pages via Groq.
  6. Rebuilding index.md and log.md deterministically.
"""

import os
import re
import json
import time
import shutil
import traceback
from pathlib import Path
from datetime import datetime, timezone

import requests
import yaml
import trafilatura
from youtube_transcript_api import YouTubeTranscriptApi

# --- Config & Directories ------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "raw"
WIKI_DIR = ROOT / "wiki"
JOURNAL_DIR = ROOT / "journal"
PROCESSED_DIR = ROOT / "processed"
STATE_DIR = ROOT / ".state"
LOG_FILE = ROOT / "log.md"
INDEX_FILE = ROOT / "index.md"
ERROR_LOG = ROOT / "error_log.md"
OFFSET_FILE = STATE_DIR / "discord_offset.json"

DISCORD_TOKEN = os.environ["DISCORD_BOT_TOKEN"]
DISCORD_CHANNEL_ID = os.environ["DISCORD_CHANNEL_ID"]
GROQ_API_KEY = os.environ["GROQ_API_KEY"]
GROQ_CHAT_MODEL = "llama-3.3-70b-versatile"
GROQ_WHISPER_MODEL = "whisper-large-v3-turbo"

TIME_BUDGET_SECONDS = 240
START_TIME = time.time()

YT_URL_RE = re.compile(r"(youtube\.com/watch\?v=|youtu\.be/)([\w-]+)")
URL_RE = re.compile(r"https?://\S+")

for d in (RAW_DIR, WIKI_DIR, JOURNAL_DIR, PROCESSED_DIR, STATE_DIR):
    d.mkdir(parents=True, exist_ok=True)
    
(STATE_DIR / ".gitkeep").touch(exist_ok=True)

def time_left():
    return TIME_BUDGET_SECONDS - (time.time() - START_TIME)

# --- Discord Helper ------------------------------------------------------
def send_discord_message(content: str):
    """Sends a response back to the Discord channel, chunking if over 2000 chars."""
    headers = {
        "Authorization": f"Bot {DISCORD_TOKEN}",
        "Content-Type": "application/json"
    }
    url = f"https://discord.com/api/v10/channels/{DISCORD_CHANNEL_ID}/messages"
    
    # Discord 2000 character limit handling
    chunks = [content[i:i+1900] for i in range(0, len(content), 1900)]
    for chunk in chunks:
        try:
            r = requests.post(url, headers=headers, json={"content": chunk}, timeout=15)
            r.raise_for_status()
        except Exception as e:
            print(f"Failed to post reply to Discord: {e}")

# --- Stage 1: Pull from Discord ------------------------------------------
def load_offset():
    if OFFSET_FILE.exists():
        return json.loads(OFFSET_FILE.read_text()).get("offset", None)
    return None

def save_offset(offset):
    OFFSET_FILE.write_text(json.dumps({"offset": offset}))

def transcribe_audio(audio_bytes: bytes) -> str:
    resp = requests.post(
        "https://api.groq.com/openai/v1/audio/transcriptions",
        headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
        files={"file": ("voice.ogg", audio_bytes)},
        data={"model": GROQ_WHISPER_MODEL},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["text"]

def slugify(title: str) -> str:
    return re.sub(r"[^a-z0-9-]", "", title.lower().replace(" ", "-"))

def write_journal_entry(text: str):
    now = datetime.now(timezone.utc)
    date_str = now.strftime("%Y-%m-%d")
    ts_str = now.strftime("%Y%m%dT%H%M%SZ")
    
    first_line = text.split("\n")[0][:40]
    clean_title = re.sub(r"^(journal:?\s*)", "", first_line, flags=re.IGNORECASE).strip() or "entry"
    slug = slugify(clean_title)
    
    file_name = f"{date_str}-{slug}.md"
    path = JOURNAL_DIR / file_name
    
    frontmatter = {
        "date": date_str,
        "title": clean_title,
        "type": "journal",
        "created": ts_str
    }
    body = f"---\n{yaml.safe_dump(frontmatter, sort_keys=False)}---\n\n{text}\n"
    path.write_text(body, encoding="utf-8")
    append_log(f"Journal logged: `{file_name}`")

def write_raw_note(text: str, note_type: str):
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    frontmatter = {
        "type": note_type,
        "status": "raw",
        "captured": ts,
        "attempts": 0,
    }
    path = RAW_DIR / f"{ts}-{note_type}.md"
    body = f"---\n{yaml.safe_dump(frontmatter, sort_keys=False)}---\n\n{text}\n"
    path.write_text(body, encoding="utf-8")

def pull_discord_updates():
    offset = load_offset()
    headers = {"Authorization": f"Bot {DISCORD_TOKEN}"}
    params = {"limit": 50}
    if offset:
        params["after"] = offset

    r = requests.get(
        f"https://discord.com/api/v10/channels/{DISCORD_CHANNEL_ID}/messages",
        headers=headers,
        params=params,
        timeout=30
    )
    r.raise_for_status()
    messages = r.json()
    
    if not messages:
        return

    messages.sort(key=lambda x: int(x["id"]))
    new_offset = offset

    for msg in messages:
        # Ignore messages sent by bots to avoid loops
        if msg.get("author", {}).get("bot"):
            continue

        new_offset = msg["id"]
        
        # Audio Attachments
        if msg.get("attachments"):
            for att in msg["attachments"]:
                if att["filename"].endswith(".ogg") or att.get("content_type", "").startswith("audio/"):
                    audio_bytes = requests.get(att["url"], timeout=60).content
                    text = transcribe_audio(audio_bytes)
                    
                    if text.lower().startswith("journal"):
                        write_journal_entry(text)
                    elif text.lower().startswith("query") or text.startswith("?"):
                        write_raw_note(text, "query")
                    else:
                        write_raw_note(text, "voice")
        
        # Text or Links
        if msg.get("content"):
            content = msg["content"].strip()
            if content:
                if content.lower().startswith("journal"):
                    write_journal_entry(content)
                elif content.lower().startswith("query") or content.startswith("?"):
                    write_raw_note(content, "query")
                else:
                    write_raw_note(content, "text" if not URL_RE.search(content) else "link")

    if new_offset:
        save_offset(new_offset)

# --- Stage 2: Resolve Links & YouTube ------------------------------------
def resolve_link_content(url: str) -> str:
    try:
        m = YT_URL_RE.search(url)
        if m:
            transcript = YouTubeTranscriptApi().fetch(m.group(2))
            return " ".join(seg.text for seg in transcript)
        downloaded = trafilatura.fetch_url(url)
        return trafilatura.extract(downloaded) if downloaded else ""
    except Exception as e:
        print(f"Warning: Failed to resolve {url}: {e}")
        return f"[Failed to fetch content from URL: {url}]"

def resolve_pending_links():
    for path in RAW_DIR.glob("*.md"):
        text = path.read_text(encoding="utf-8")
        fm, body = split_frontmatter(text)
        if fm.get("type") == "link" and fm.get("status") == "raw":
            url_match = URL_RE.search(body)
            if url_match:
                resolved = resolve_link_content(url_match.group(0))
                if resolved:
                    fm["status"] = "resolved"
                    write_frontmatted(path, fm, resolved)

# --- Stage 3: Summarization & 2-Way Query Processing ---------------------
def split_frontmatter(text: str):
    if text.startswith("---"):
        _, fm_block, body = text.split("---", 2)
        return (yaml.safe_load(fm_block) or {}), body.strip()
    return {}, text.strip()

def write_frontmatted(path: Path, fm: dict, body: str):
    path.write_text(f"---\n{yaml.safe_dump(fm, sort_keys=False)}---\n\n{body}\n", encoding="utf-8")

def chunk_text(text: str, max_chars: int = 6000):
    for i in range(0, len(text), max_chars):
        yield text[i:i + max_chars]

def call_groq_chat(prompt: str) -> str:
    resp = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
        json={
            "model": GROQ_CHAT_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
        },
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]

def get_wiki_context() -> str:
    """Combines existing wiki pages to serve as context for user queries."""
    context_blocks = []
    for page in WIKI_DIR.glob("*.md"):
        text = page.read_text(encoding="utf-8")
        _, body = split_frontmatter(text)
        context_blocks.append(f"--- Note: {page.stem} ---\n{body[:1500]}")
    return "\n\n".join(context_blocks[:15])  # Cap at 15 notes for context length safety

def process_query(path: Path):
    """Answers a user query using Wiki knowledge, posts to Discord, and saves to Wiki."""
    _, question = split_frontmatter(path.read_text(encoding="utf-8"))
    clean_question = re.sub(r"^(query:?\s*|\?\s*)", "", question, flags=re.IGNORECASE).strip()
    
    wiki_context = get_wiki_context()
    
    prompt = f"""You are the System Archivist for a Second Brain knowledge base.
Answer the user's question accurately using ONLY the knowledge provided below from their wiki notes.
If the wiki notes do not contain enough context, answer using your general knowledge, but explicitly state what came from the wiki vs general knowledge.

User Question: {clean_question}

Knowledge Base Context:
{wiki_context if wiki_context else "No wiki notes available yet."}
"""
    answer = call_groq_chat(prompt)
    
    # Send answer back to Discord!
    discord_reply = f"**Query:** {clean_question}\n\n**Answer:**\n{answer}"
    send_discord_message(discord_reply)
    
    # Save Query & Answer into Wiki
    slug = slugify(clean_question[:30])
    wiki_path = WIKI_DIR / f"query-{slug}.md"
    
    frontmatter = {
        "title": f"Query: {clean_question[:40]}",
        "tags": ["query", "qa"],
        "created": datetime.now(timezone.utc).isoformat(timespec="minutes"),
        "status": "processed"
    }
    
    wiki_path.write_text(
        f"---\n{yaml.safe_dump(frontmatter, sort_keys=False)}---\n\n"
        f"**Question:** {clean_question}\n\n"
        f"## Answer\n{answer}\n",
        encoding="utf-8"
    )
    
    shutil.move(str(path), str(PROCESSED_DIR / path.name))
    append_log(f"Answered Query: `{clean_question[:30]}` -> posted to Discord")

def summarize_one(path: Path) -> dict:
    fm, body = split_frontmatter(path.read_text(encoding="utf-8"))
    partials = [
        call_groq_chat(
            "Summarize the key facts and ideas in this text as dense "
            f"markdown notes, no fluff:\n\n{chunk}"
        )
        for chunk in chunk_text(body)
    ]
    combined = "\n\n".join(partials)
    synth = call_groq_chat(
        "Turn these notes into ONE wiki page. Output strict JSON with "
        "keys title, tags (list), body (markdown). Notes:\n\n" + combined
    )
    
    synth_clean = synth.strip()
    if synth_clean.startswith("```"):
        synth_clean = re.sub(r"^```(?:json)?\n|\n```$", "", synth_clean).strip()
        
    try:
        page = json.loads(synth_clean)
    except json.JSONDecodeError:
        page = {"title": path.stem, "tags": [], "body": synth_clean}

    return page

def process_raw_files():
    for path in sorted(RAW_DIR.glob("*.md")):
        if time_left() <= 0:
            break
        fm, _ = split_frontmatter(path.read_text(encoding="utf-8"))
        if fm.get("status") not in ("raw", "resolved"):
            continue
        if fm.get("attempts", 0) >= 3:
            continue  

        try:
            # Check if this is a Q&A query
            if fm.get("type") == "query":
                process_query(path)
                continue

            # Standard Note Summarization
            page = summarize_one(path)
            slug = slugify(page["title"])
            wiki_path = WIKI_DIR / f"{slug}.md"
            frontmatter = {
                "title": page["title"],
                "tags": page.get("tags", []),
                "source": path.name,
                "created": datetime.now(timezone.utc).isoformat(timespec="minutes"),
                "status": "processed",
            }
            wiki_path.write_text(
                f"---\n{yaml.safe_dump(frontmatter, sort_keys=False)}---\n\n"
                f"{page['body']}\n",
                encoding="utf-8",
            )
            shutil.move(str(path), str(PROCESSED_DIR / path.name))
            append_log(f"Processed `{path.name}` -> [[{page['title']}]]")
        except Exception:
            fm["attempts"] = fm.get("attempts", 0) + 1
            body_text = split_frontmatter(path.read_text(encoding="utf-8"))[1]
            write_frontmatted(path, fm, body_text)
            with open(ERROR_LOG, "a", encoding="utf-8") as f:
                f.write(f"\n---\n{datetime.now()}: {path.name}\n{traceback.format_exc()}\n")

# --- Stage 4: Deterministic Bookkeeping ----------------------------------
def append_log(entry: str):
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"- {ts} | {entry}\n")

def rebuild_index():
    by_tag = {}
    for page in sorted(WIKI_DIR.glob("*.md")):
        fm, _ = split_frontmatter(page.read_text(encoding="utf-8"))
        for tag in fm.get("tags", []):
            by_tag.setdefault(tag, []).append(fm.get("title", page.stem))
    lines = ["# Index\n"]
    for tag in sorted(by_tag):
        lines.append(f"\n## {tag}")
        lines += [f"- [[{t}]]" for t in sorted(by_tag[tag])]
    INDEX_FILE.write_text("\n".join(lines), encoding="utf-8")

if __name__ == "__main__":
    pull_discord_updates()
    resolve_pending_links()
    process_raw_files()
    rebuild_index()
