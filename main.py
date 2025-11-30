import os
import sys
import asyncio
import json
import re
import time
import traceback
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from collections import Counter, defaultdict

# 3rd-party libs
try:
    from bs4 import BeautifulSoup
except Exception:
    print("Missing dependency 'beautifulsoup4'. Install: pip install beautifulsoup4")
    raise

try:
    import requests
except Exception:
    print("Missing dependency 'requests'. Install: pip install requests")
    raise

try:
    from dotenv import load_dotenv
except Exception:
    print("Missing dependency 'python-dotenv'. Install: pip install python-dotenv")
    raise

# Playwright import (async)
try:
    from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError
except Exception:
    print("Missing dependency 'playwright'. Install: pip install playwright  AND run 'playwright install'")
    raise

# ADK agent imports (kept as in your original file)
from google.adk.agents import Agent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.genai import types

# ---------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("propel_seo_playwright")

# ---------------------------------------------------------
# Load environment
# ---------------------------------------------------------
load_dotenv()

if "GOOGLE_API_KEY" not in os.environ:
    logger.error("🔑 GOOGLE_API_KEY not found in environment. Please set the environment variable.")
    sys.exit(1)
else:
    logger.info("✅ GOOGLE_API_KEY found.")

GOOGLE_CX = os.environ.get("GOOGLE_CX")
ALLOW_EXTERNAL_SEARCH = os.environ.get("ALLOW_EXTERNAL_SEARCH", "0") == "1"

# Optional LinkedIn auth (use carefully)
LINKEDIN_EMAIL = os.environ.get("LINKEDIN_EMAIL")
LINKEDIN_PASSWORD = os.environ.get("LINKEDIN_PASSWORD")

# Playwright runtime options
HEADLESS = os.environ.get("HEADLESS", "0") == "1"
PLAYWRIGHT_TIMEOUT = int(os.environ.get("PLAYWRIGHT_TIMEOUT", "30"))

# Retry config for Gemini
retry_config = types.HttpRetryOptions(
    attempts=5,
    exp_base=7,
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504]
)

# ---------------------------------------------------------
# Utilities
# ---------------------------------------------------------
def get_link_url() -> str:
    if len(sys.argv) < 2:
        logger.error("❗ Usage: python main.py <any_content_url>")
        sys.exit(1)
    return sys.argv[1].strip()

def extract_platform_name(url: str) -> str:
    try:
        parsed = re.sub(r'https?://', '', url).split('/')[0].lower()
        if parsed.startswith('www.'):
            parsed = parsed[4:]
        platform = re.sub(r'\.\w{2,4}(\.\w{2,4})?$', '', parsed)
        return platform.capitalize()
    except Exception:
        return "Unknown"

# Simple fetch fallback (requests + BeautifulSoup)
HTTP_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36"
}
REQUEST_TIMEOUT = 12

def fetch_with_requests(url: str, max_retries: int = 2) -> Optional[str]:
    last_exc = None
    for attempt in range(max_retries + 1):
        try:
            r = requests.get(url, headers=HTTP_HEADERS, timeout=REQUEST_TIMEOUT)
            r.raise_for_status()
            return r.text
        except Exception as e:
            last_exc = e
            logger.warning(f"Requests fetch attempt {attempt+1} failed: {e}")
            time.sleep(1 + attempt * 0.5)
    logger.error(f"Requests fetch failed after retries: {last_exc}")
    return None

# Lightweight tokenizer and keyword extraction (used for the agent context)
STOPWORDS = set("""
a about above after again against all am an and any are as at be because been before being below between both
but by can could did do does doing down during each few for from further had has have having he her here hers
him his how i if in into is it its just me more most my nor not of off on once only or other our out over
own same she should so some such than that the their them then there these they this those through to too under
until up very was we were what when where which while who will with you your
""".split())

WORD_RE = re.compile(r"[A-Za-z0-9']{2,}")

def tokenize(text: str) -> List[str]:
    return [t.lower() for t in WORD_RE.findall(text or "")]

def extract_top_keywords(texts: List[str], top_n: int = 12) -> List[Tuple[str, int]]:
    counter = Counter()
    for t in texts:
        for tok in tokenize(t):
            if tok in STOPWORDS:
                continue
            if tok.isdigit():
                continue
            counter[tok] += 1
    return counter.most_common(top_n)

# Readability (simple heuristics)
def estimate_syllables(word: str) -> int:
    word = word.lower()
    vowels = "aeiouy"
    if len(word) <= 3:
        return 1
    syll = 0
    prev_v = False
    for ch in word:
        is_v = ch in vowels
        if is_v and not prev_v:
            syll += 1
        prev_v = is_v
    if word.endswith("e"):
        syll = max(1, syll - 1)
    return max(1, syll)

def flesch_reading_ease(text: str) -> Optional[float]:
    if not text or len(text.split()) < 50:
        return None
    sentences = [s for s in re.split(r'[.!?]+', text) if s.strip()]
    words = tokenize(text)
    syllables = sum(estimate_syllables(w) for w in words)
    asl = len(words) / max(1, len(sentences))
    asw = syllables / max(1, len(words))
    score = 206.835 - 1.015 * asl - 84.6 * asw
    return round(score, 2)

# ---------------------------------------------------------
# Niche & Competitor Analysis Helpers (new: Option C)
# ---------------------------------------------------------
def infer_niche_from_scraped(scraped: Dict[str, Any], top_keywords: List[Tuple[str, int]]) -> Dict[str, Any]:
    """
    Derive a safe niche analysis entirely from scraped page content and keywords.
    Returns a dict with:
      - primary_niche: list[str]
      - secondary_niche: list[str]
      - adjacent_creator_categories: list[str]
      - behavior_patterns: list[str]
      - content_gaps: list[str]
      - recommendations: list[str]
    """
    # Gather sources
    headings = scraped.get("headings") or []
    paragraphs = scraped.get("paragraphs") or []
    posts = scraped.get("posts") or []
    profile = scraped.get("linkedin_profile") or {}

    # Compose a combined text corpus
    corpus = " ".join(headings + paragraphs + [p.get("text","") for p in posts] + [profile.get("headline","") or "", profile.get("about","") or ""])
    tokens = tokenize(corpus)
    # Use top_keywords passed in, but also compute frequent nouns/adjectives heuristically by token frequency
    keyword_terms = [k for k,_ in top_keywords] if top_keywords else []
    freq = Counter(tok for tok in tokens if tok not in STOPWORDS and not tok.isdigit())
    most_common = [w for w,_ in freq.most_common(20)]

    # Heuristic mapping to niches (safe, no external assumptions)
    primary_niche = []
    secondary_niche = []
    adjacent = set()

    # match simple domain words to niche buckets (this is deterministic and local)
    niche_map = {
        "product": "Product Management",
        "pm": "Product Management",
        "startup": "Startup Advising",
        "founder": "Startup Advising",
        "leadership": "Leadership & Management",
        "team": "Leadership & Management",
        "ai": "AI & Machine Learning",
        "data": "Data & Analytics",
        "strategy": "Tech Strategy",
        "metrics": "Product Metrics",
        "growth": "Growth & GTM",
        "culture": "Organizational Culture",
        "agency": "High-Agency / Productivity"
    }

    # detect presence
    for token in most_common[:60]:
        if token in niche_map:
            candidate = niche_map[token]
            if candidate not in primary_niche:
                primary_niche.append(candidate)
    # ensure some defaults if profile contains matching phrases in headline/about
    headline = (profile.get("headline") or "").lower()
    about = (profile.get("about") or "").lower()
    for key, val in niche_map.items():
        if key in headline or key in about:
            if val not in primary_niche:
                primary_niche.append(val)

    # derive secondary niches from keywords and tokens
    for token in keyword_terms:
        t = token.lower()
        if t in niche_map and niche_map[t] not in primary_niche:
            secondary_niche.append(niche_map[t])

    # fill fallback niches if empty using common tokens
    if not primary_niche:
        # heuristically infer from most_common tokens mapping
        for token in most_common[:30]:
            if token in niche_map:
                primary_niche.append(niche_map[token])
        # if still empty, use a conservative default
        if not primary_niche:
            primary_niche = ["Product Management Thought Leadership"]

    # adjacent creator categories (general behavioral buckets)
    if "Product Management" in primary_niche or "Product Metrics" in primary_niche:
        adjacent.update(["Product Educators", "Product Framework Creators", "PM Case-Study Authors"])
    if "Startup Advising" in primary_niche:
        adjacent.update(["Startup Advisors", "Founder Mentors", "Early-Stage Operators"])
    if "Leadership & Management" in primary_niche or "High-Agency / Productivity" in primary_niche:
        adjacent.update(["Leadership Coaches", "Org Design Practitioners", "Team-Building Creators"])
    if "AI & Machine Learning" in primary_niche:
        adjacent.update(["AI Practitioners", "AI Strategy Writers"])

    # behavior patterns (observed norms in the niche, inferred from posts / headlines)
    behavior_patterns = []
    # find presence of long-form essays vs short posts by heuristic
    avg_paragraph_len = sum(len(p.split()) for p in paragraphs)/max(1, len(paragraphs))
    if avg_paragraph_len > 120:
        behavior_patterns.append("Long-form analytical essays & threads")
    else:
        behavior_patterns.append("Short insights & tactical posts")
    # posts frequency presence
    if posts and len(posts) >= 3:
        behavior_patterns.append("Uses topical posts and some hashtags (when present)")
    else:
        behavior_patterns.append("Low posting frequency of short posts; emphasis on evergreen articles")
    # visuals/tactics
    if any(len(h) > 20 for h in headings):
        behavior_patterns.append("Uses educational headings and explicit frameworks")
    else:
        behavior_patterns.append("Sparse visual or slide-based content observed")

    # content gaps (conservative, evidence-based)
    content_gaps = []
    # Hashtag usage
    all_hashtags = set()
    for p in posts:
        for h in p.get("hashtags", []):
            all_hashtags.add(h.lower().strip("#"))
    if not all_hashtags:
        content_gaps.append("Low / no hashtag use — discoverability is limited")
    # CTAs and repurposing
    if not any("call to action" in (p.get("text","").lower()) or "download" in (p.get("text","").lower()) for p in posts):
        content_gaps.append("Few explicit CTAs or lead magnets in posts")
    # visual content
    if not scraped.get("images"):
        content_gaps.append("Little to no image/visual content; consider carousels or infographics")
    # short-form presence
    if avg_paragraph_len > 150 and (not posts or len(posts) < 3):
        content_gaps.append("Content is long-form and may miss short-form audiences; repurpose for micro-content")

    # recommendations (actionable, derived from gaps)
    recommendations = []
    if "Low / no hashtag use — discoverability is limited" in content_gaps:
        recommendations.append("Add 3-6 strategic hashtags per post mixing broad (#ProductManagement) and niche (#HighAgency)")
    if "Few explicit CTAs or lead magnets in posts" in content_gaps:
        recommendations.append("Add one clear CTA in 30% of posts (question, poll, or downloadable checklist)")
    if "Little to no image/visual content; consider carousels or infographics" in content_gaps:
        recommendations.append("Convert top articles into 6–8 slide carousels for LinkedIn and repurpose as short videos")
    if "Content is long-form and may miss short-form audiences; repurpose for micro-content" in content_gaps:
        recommendations.append("Repurpose long articles into 4–6 short posts and a weekly short-form video")

    # compact result
    niche_analysis = {
        "primary_niche": list(dict.fromkeys(primary_niche))[:6],
        "secondary_niche": list(dict.fromkeys(secondary_niche))[:6],
        "adjacent_creator_categories": list(adjacent)[:8],
        "behavior_patterns": behavior_patterns[:8],
        "content_gaps": content_gaps[:8],
        "recommendations": recommendations[:8],
        "observed_hashtags": sorted(list(all_hashtags))[:20],
        "most_common_tokens": most_common[:20]
    }
    return niche_analysis

def analyze_competitor_gap(current_hashtags: str, competitor_hashtags: str, page_top_keywords: List[Tuple[str,int]] = None, posts_hashtags: List[str] = None) -> str:
    """
    Enhanced gap analysis:
    - Uses competitor_hashtags when provided
    - Also considers page_top_keywords and posts_hashtags to compute missed opportunities
    - Returns safe summary string
    """
    current_list = {tag.strip('#').lower() for tag in re.findall(r'#\w+', (current_hashtags or "").lower())}
    competitor_list = {tag.strip('#').lower() for tag in re.findall(r'#\w+', (competitor_hashtags or "").lower())}
    missing = competitor_list - current_list

    # Also infer missing tags from page keywords
    keyword_based_missing = set()
    if page_top_keywords:
        # page_top_keywords is list of (term, count)
        for term, _ in page_top_keywords[:12]:
            # if keyword isn't a hashtag currently, suggest it as a hashtag
            if term.lower() not in current_list:
                keyword_based_missing.add(term.lower())

    # include hashtags from posts sample
    posts_missing = set()
    if posts_hashtags:
        for h in posts_hashtags:
            hnorm = h.lower().strip("#")
            if hnorm not in current_list:
                posts_missing.add(hnorm)

    combined_missing = set(list(missing) + list(keyword_based_missing) + list(posts_missing))

    if not combined_missing:
        return "[COMPETITOR GAP ANALYSIS] No missing hashtags or keyword-based hashtag opportunities found based on provided inputs."

    # compute a simple gap score (higher when more items missing)
    gap_score = min(100, max(30, 70 + len(combined_missing)*2))
    top_missing = ", ".join(f"#{t}" for t in list(combined_missing)[:8])
    return f"[COMPETITOR GAP ANALYSIS: Score {gap_score}/100] Suggested missing hashtags/terms: {top_missing}. Recommendation: test 2-4 of these in next posts and measure reach."

# ---------------------------------------------------------
# Optional: Google Custom Search (kept conservative)
# ---------------------------------------------------------
def google_custom_search(query: str, api_key: str, cx: str, num: int = 5) -> Dict[str, Any]:
    try:
        params = {"key": api_key, "cx": cx, "q": query, "num": min(num, 10)}
        r = requests.get("https://www.googleapis.com/customsearch/v1", params=params, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.warning(f"Google Custom Search failed: {e}")
        return {"error": str(e)}

# ---------------------------------------------------------
# Playwright Stealth Helpers (JS snippets to mask automation)
# ---------------------------------------------------------
_STEALTH_JS = """
// Pass the webdriver test
Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
// Mock chrome runtime
window.chrome = window.chrome || { runtime: {} };
// Languages
Object.defineProperty(navigator, 'languages', { get: () => ['en-US', 'en'] });
// Plugins
Object.defineProperty(navigator, 'plugins', { get: () => [1,2,3,4,5] });
// Permissions
const originalQuery = window.navigator.permissions.query;
window.navigator.permissions.__query = originalQuery;
window.navigator.permissions.query = parameters => (
  parameters.name === 'notifications' ? Promise.resolve({ state: Notification.permission }) : originalQuery(parameters)
);
"""

# ---------------------------------------------------------
# Playwright scraping logic (async) - unchanged aside from structure
# ---------------------------------------------------------
async def scrape_with_playwright(url: str, depth: str = "C", headless: bool = False, login: bool = False) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "url": url,
        "scraped_with": "playwright",
        "error": None,
        "title": "",
        "meta_description": "",
        "meta_keywords": "",
        "headings": [],
        "paragraphs": [],
        "images": [],
        "links": [],
        "schema_jsonld": [],
        "linkedin_profile": None,
        "posts": [],
    }

    try:
        logger.info("Launching Playwright browser...")
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=headless, args=["--no-sandbox", "--disable-dev-shm-usage"])
            context = await browser.new_context(user_agent=HTTP_HEADERS["User-Agent"], locale="en-US")
            page = await context.new_page()
            await page.add_init_script(_STEALTH_JS)

            if login and LINKEDIN_EMAIL and LINKEDIN_PASSWORD and "linkedin.com" in url.lower():
                logger.info("Attempting LinkedIn login (credentials provided).")
                try:
                    await page.goto("https://www.linkedin.com/login", timeout=PLAYWRIGHT_TIMEOUT * 1000)
                    await page.fill('input[id="username"]', LINKEDIN_EMAIL)
                    await page.fill('input[id="password"]', LINKEDIN_PASSWORD)
                    await page.click('button[type="submit"]')
                    await page.wait_for_timeout(2000)
                except Exception as e:
                    logger.warning(f"LinkedIn login attempt had an issue: {e}")

            logger.info(f"Playwright navigating to {url} ...")
            try:
                await page.goto(url, timeout=PLAYWRIGHT_TIMEOUT * 1000, wait_until="networkidle")
            except PlaywrightTimeoutError:
                logger.warning("Playwright navigation timed out; attempting to continue with available DOM.")
            except Exception as e:
                logger.error(f"Playwright navigation failed: {e}")
                await browser.close()
                result["error"] = f"Playwright navigation failed: {e}"
                return result

            await page.wait_for_timeout(1200)
            html = await page.content()
            soup = BeautifulSoup(html, "html.parser")

            title = soup.title.string.strip() if soup.title and soup.title.string else ""
            meta_desc = ""
            meta_tag = soup.find("meta", attrs={"name": "description"})
            if meta_tag and meta_tag.get("content"):
                meta_desc = meta_tag["content"].strip()
            if not meta_desc:
                og = soup.find("meta", attrs={"property": "og:description"}) or soup.find("meta", attrs={"name": "og:description"})
                if og and og.get("content"):
                    meta_desc = og["content"].strip()

            headings = [h.get_text(" ", strip=True) for h in soup.find_all(re.compile("^h[1-6]$"))]
            paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p") if p.get_text(strip=True)]

            images = []
            for img in soup.find_all("img"):
                src = img.get("src") or img.get("data-src") or ""
                alt = img.get("alt") or ""
                images.append({"src": src, "alt": alt})

            links = []
            for a in soup.find_all("a"):
                href = a.get("href")
                text = a.get_text(" ", strip=True)
                if href:
                    links.append({"href": href, "text": text})

            schema_jsonld = []
            for tag in soup.find_all("script", {"type": "application/ld+json"}):
                try:
                    schema_jsonld.append(tag.string.strip())
                except Exception:
                    pass

            result.update({
                "title": title,
                "meta_description": meta_desc,
                "meta_keywords": (soup.find("meta", attrs={"name":"keywords"}) or {}).get("content", ""),
                "headings": headings,
                "paragraphs": paragraphs,
                "images": images,
                "links": links,
                "schema_jsonld": schema_jsonld,
            })

            if "linkedin.com/in/" in url.lower() or "linkedin.com/pub/" in url.lower():
                profile = {"name": None, "headline": None, "about": None, "location": None, "experience": [], "education": [], "skills": [], "featured": []}
                try:
                    h1 = soup.find("h1")
                    if h1:
                        profile["name"] = h1.get_text(" ", strip=True)
                        sib = h1.find_next_sibling()
                        if sib:
                            profile["headline"] = sib.get_text(" ", strip=True)[:300]
                    about_labels = soup.find_all(lambda tag: tag.name in ["section", "div"] and ("about" in (tag.get("id") or "").lower() or "about" in " ".join(tag.get("class", []) if tag.get("class") else [])))
                    if about_labels:
                        profile["about"] = about_labels[0].get_text(" ", strip=True)[:2000]
                    experiences = []
                    for ex in soup.find_all(lambda tag: tag.name == "li" and "experience" in (tag.get("class") or []) or (tag.find("span") and "experience" in tag.get_text(" ").lower())):
                        experiences.append(ex.get_text(" ", strip=True))
                    if not experiences:
                        exp_header = soup.find(lambda tag: tag.name in ["h2","h3"] and "experience" in tag.get_text(" ").lower())
                        if exp_header:
                            uls = exp_header.find_next("ul")
                            if uls:
                                for li in uls.find_all("li"):
                                    experiences.append(li.get_text(" ", strip=True))
                    profile["experience"] = experiences[:25]
                    educations = []
                    edu_header = soup.find(lambda tag: tag.name in ["h2","h3"] and "education" in tag.get_text(" ").lower())
                    if edu_header:
                        ul = edu_header.find_next("ul")
                        if ul:
                            for li in ul.find_all("li"):
                                educations.append(li.get_text(" ", strip=True))
                    profile["education"] = educations[:10]
                    skills = []
                    skill_tags = soup.find_all(lambda tag: tag.name in ["span","li","button"] and tag.get_text(strip=True) and len(tag.get_text(strip=True)) < 60 and ("skill" in " ".join(tag.get("class", []) if tag.get("class") else []) or "#" in tag.get_text()))
                    for s in skill_tags:
                        txt = s.get_text(" ", strip=True)
                        if len(txt.split()) <= 6:
                            skills.append(txt)
                    profile["skills"] = list(dict.fromkeys(skills))[:50]
                    featured = []
                    feat_header = soup.find(lambda tag: tag.name in ["h2","h3"] and "featured" in tag.get_text(" ").lower())
                    if feat_header:
                        feat_div = feat_header.find_next_sibling()
                        if feat_div:
                            for item in feat_div.find_all(["a","div"]):
                                txt = item.get_text(" ", strip=True)
                                if txt:
                                    featured.append(txt)
                    profile["featured"] = featured[:20]

                    result["linkedin_profile"] = profile
                except Exception as e:
                    logger.warning(f"LinkedIn profile parsing partial failure: {e}")

            if depth.upper() == "C":
                posts = []
                try:
                    post_containers = soup.find_all(lambda tag: tag.name in ["div","article"] and (tag.find("span") or tag.find("time")))
                    for pc in post_containers:
                        text = pc.get_text(" ", strip=True)
                        if not text or len(text.split()) < 6:
                            continue
                        hashtags = re.findall(r'#\w+', text)
                        engagement = {}
                        likes = re.search(r'([\d,\.]+)\s+likes?', pc.get_text(" ", strip=True).lower())
                        comments = re.search(r'([\d,\.]+)\s+comments?', pc.get_text(" ", strip=True).lower())
                        if likes:
                            engagement["likes"] = likes.group(1)
                        if comments:
                            engagement["comments"] = comments.group(1)
                        posts.append({"text": text[:2000], "hashtags": hashtags, "engagement": engagement})
                        if len(posts) >= 25:
                            break
                except Exception as e:
                    logger.debug(f"Post parsing had non-fatal error: {e}")
                result["posts"] = posts

            await browser.close()
            return result

    except Exception as e:
        logger.error(f"Playwright scrape failed: {e}")
        traceback.print_exc()
        result["error"] = str(e)
        return result

# ---------------------------------------------------------
# Agents: same design as you had, adjusted prompts to use structured JSON only
# ---------------------------------------------------------
root_agent = Agent(
    name="ViralContentStrategist",
    model=Gemini(model="gemini-2.5-flash", retry_options=retry_config),
    description="Ruthlessly analyzes content and generates the final A+ type report.",
    instruction="""
You are a Ruthless Viral Content Strategist (A+ Level). Generate a professional Markdown report.
Use ONLY the provided structured JSON context. Do NOT fabricate competitor names or metrics.
If competitor_snippets is not provided, use the 'niche_analysis' field in the JSON to produce a robust Competitor & Niche Analysis.
Required sections: Content Breakdown & SEO Score; Engagement & Hook Strategy (hook rating 1-10); Competitor & Niche Analysis (use niche_analysis if no competitor_snippets); Hashtag Audit & Gap Analysis (use gap_analysis and niche_analysis.observed_hashtags); Actionable Content Repurposing + Rewritten Viral Post Example.
Return valid Markdown only.
""",
    tools=[]
)

qa_agent = Agent(
    name="QualityAssuranceAgent",
    model=Gemini(model="gemini-2.5-flash", retry_options=retry_config),
    description="Expert editor and formatter. Cleans and finalizes the report.",
    instruction="""
You are an expert editor and formatter. Receive a raw Markdown report and:
1. Remove tool/agent metadata and diagnostic logs.
2. Proofread and fix grammar/spelling.
3. Ensure output is purely Markdown starting with the first heading.
4. Do not include any names from the prompt in competitor sections.
Return only the final clean Markdown.
""",
    tools=[]
)

logger.info("✅ Gemini agents configured (root + QA).")

# ---------------------------------------------------------
# Main pipeline (async) with Option C (combined upgrade)
# ---------------------------------------------------------
async def process_link():
    output_filename = "propel_seo_report.md"
    link_url = get_link_url()
    platform_name = extract_platform_name(link_url)
    tznow = datetime.now().astimezone().strftime("%A, %B %d, %Y at %I:%M %p %Z")

    strategy_runner = InMemoryRunner(agent=root_agent)
    qa_runner = InMemoryRunner(agent=qa_agent)

    header_md = (
        f"# P.R.O.P.E.L. Digital Growth SEO Report 📈\n\n"
        f"**Input URL:** `{link_url}`  \n"
        f"**Platform Detected:** `{platform_name}`  \n"
        f"**Report Generated Date:** `{tznow}`\n\n---\n\n"
    )

    try:
        # 1) Attempt Playwright scrape (ultimate depth)
        logger.info("Starting Playwright scrape (Ultimate depth, stealth enabled)...")
        try:
            pw_result = await scrape_with_playwright(link_url, depth="C", headless=HEADLESS, login=bool(LINKEDIN_EMAIL and LINKEDIN_PASSWORD))
        except Exception as e:
            logger.error(f"Playwright raised an exception: {e}")
            pw_result = {"error": str(e)}

        # 2) If Playwright failed or returned little data, fallback to requests scraping
        if not pw_result or pw_result.get("error") or (not pw_result.get("paragraphs") and not pw_result.get("title")):
            logger.info("Playwright returned limited/no data — falling back to requests+BeautifulSoup scrape.")
            html = fetch_with_requests(link_url, max_retries=2)
            fallback = {"url": link_url, "scraped_with": "requests", "error": None, "title": "", "meta_description": "", "paragraphs": [], "headings": [], "images": [], "links": [], "schema_jsonld": []}
            if html:
                soup = BeautifulSoup(html, "html.parser")
                fallback["title"] = soup.title.string.strip() if soup.title and soup.title.string else ""
                md = soup.find("meta", attrs={"name":"description"})
                fallback["meta_description"] = md.get("content", "").strip() if md and md.get("content") else ""
                fallback["headings"] = [h.get_text(" ", strip=True) for h in soup.find_all(re.compile("^h[1-6]$"))]
                fallback["paragraphs"] = [p.get_text(" ", strip=True) for p in soup.find_all("p") if p.get_text(strip=True)]
                fallback["images"] = [{"src": img.get("src"), "alt": img.get("alt","")} for img in soup.find_all("img")]
                fallback["links"] = [{"href":a.get("href"), "text":a.get_text(" ", strip=True)} for a in soup.find_all("a") if a.get("href")]
                fallback["schema_jsonld"] = [tag.string for tag in soup.find_all("script", {"type":"application/ld+json"}) if tag.string]
                pw_result = fallback
            else:
                logger.warning("Requests fallback failed as well; continuing with minimal context.")

        scraped = pw_result

        # 3) Compose simple analysis artifacts
        all_text = " ".join((scraped.get("headings") or []) + (scraped.get("paragraphs") or []))
        if not all_text.strip():
            all_text = (scraped.get("title") or "") + " " + (scraped.get("meta_description") or "")

        top_keywords = extract_top_keywords((scraped.get("paragraphs") or []) + [scraped.get("title", ""), scraped.get("meta_description", "")], top_n=12)
        readability = flesch_reading_ease(all_text)

        # 4) Competitor discovery (optional)
        competitor_data = None
        gap_analysis_text = "[COMPETITOR GAP ANALYSIS] External competitor data not available; gap analysis skipped."
        if ALLOW_EXTERNAL_SEARCH and GOOGLE_CX:
            try:
                logger.info("Attempting Google Custom Search for competitor discovery (external).")
                query = f"site:linkedin.com \"{platform_name}\" \"product\" OR \"product management\""
                gres = google_custom_search(query, os.environ["GOOGLE_API_KEY"], GOOGLE_CX, num=5)
                if gres and not gres.get("error") and gres.get("items"):
                    items = [{"title": it.get("title"), "snippet": it.get("snippet"), "link": it.get("link")} for it in gres.get("items", [])]
                    competitor_data = {"results": items}
                    comp_tags = set()
                    for it in items:
                        comp_tags.update(re.findall(r'#\w+', it.get("snippet","")))
                    if comp_tags:
                        # posts_hashtags derived as list
                        posts_hashtags = []
                        if scraped.get("posts"):
                            for p in scraped["posts"]:
                                for h in p.get("hashtags", []):
                                    posts_hashtags.append(h)
                        gap_analysis_text = analyze_competitor_gap(scraped.get("meta_keywords",""), ", ".join(list(comp_tags)), page_top_keywords=top_keywords, posts_hashtags=posts_hashtags)
                    else:
                        gap_analysis_text = "[COMPETITOR GAP ANALYSIS] Competitor snippets did not contain hashtags; analysis skipped."
                else:
                    logger.info("No competitor results or search returned error — skipping external competitor analysis.")
            except Exception as e:
                logger.warning(f"Competitor discovery failed: {e}")

        # Otherwise, if LinkedIn posts were scraped, we can compute a local hashtag set to analyze
        posts_hashtags_list = []
        if scraped.get("posts"):
            comp_tags = set()
            for p in scraped["posts"]:
                for h in p.get("hashtags", []):
                    comp_tags.add(h.lower().strip("#"))
                    posts_hashtags_list.append(h)
            if comp_tags:
                gap_analysis_text = analyze_competitor_gap(scraped.get("meta_keywords",""), ", ".join(list(comp_tags)), page_top_keywords=top_keywords, posts_hashtags=posts_hashtags_list)

        # 4b) Infer niche analysis from the scraped content (this is always done and never fabricates external names)
        niche_analysis = infer_niche_from_scraped(scraped, top_keywords)

        # 5) Prepare JSON context for agent (structured, minimal)
        strategy_input = {
            "target_url": link_url,
            "title": scraped.get("title","")[:500],
            "meta_description": scraped.get("meta_description","")[:1000],
            "sample_headings": scraped.get("headings", [])[:8],
            "sample_paragraphs": (scraped.get("paragraphs") or [])[:6],
            "top_keywords": [k for k,_ in top_keywords],
            "readability_score": readability,
            "competitor_available": bool(competitor_data),
            "competitor_snippets": competitor_data.get("results") if competitor_data else None,
            "gap_analysis": gap_analysis_text,
            "linkedin_profile": scraped.get("linkedin_profile"),
            "posts_sample": (scraped.get("posts") or [])[:10],
            # New fields to support guaranteed niche analysis
            "niche_analysis": niche_analysis,
        }

        # 6) Run strategy agent with strict instructions (note: agent told to use niche_analysis if competitor missing)
        final_report_query = f"""
Using ONLY the structured JSON context provided (do NOT infer or fabricate any external data), generate a Markdown SEO + virality report.
Sections required:
## Content Breakdown & SEO Score
## Engagement & Hook Strategy (CRITICAL AUDIT)
## Competitor & Niche Analysis
## Hashtag Audit & Gap Analysis
## Actionable Content Repurposing

Important instructions for Competitor & Niche Analysis:
- If "competitor_snippets" is provided, summarize themes from them (do NOT invent names or numbers).
- If "competitor_snippets" is NOT provided, use the provided "niche_analysis" object for a robust, evidence-based Competitor & Niche Analysis. Do NOT fabricate competitor names or external metrics.
- Use "niche_analysis" fields: primary_niche, secondary_niche, adjacent_creator_categories, behavior_patterns, content_gaps, recommendations, observed_hashtags.
- Use "gap_analysis" and "niche_analysis.observed_hashtags" for the Hashtag Audit & Gap Analysis section.

Return valid Markdown only. Here is the JSON context:
{json.dumps(strategy_input, ensure_ascii=False, indent=2)}
"""
        logger.info("⏳ Running strategy agent to compile the draft report...")
        strategy_runner = InMemoryRunner(agent=root_agent)
        strategy_response = await strategy_runner.run_debug(final_report_query)
        raw_agent_markdown = ""
        try:
            if hasattr(strategy_response, "text") and strategy_response.text:
                raw_agent_markdown = strategy_response.text
            elif hasattr(strategy_response, "output") and strategy_response.output:
                raw_agent_markdown = strategy_response.output
            else:
                raw_agent_markdown = str(strategy_response)
        except Exception:
            raw_agent_markdown = str(strategy_response)
        raw_agent_markdown = raw_agent_markdown.strip()
        if not raw_agent_markdown:
            raw_agent_markdown = "## Error\nStrategy agent returned no content."

        # 7) QA agent cleanup
        qa_query = f"""
Proofread and clean the Markdown report below. Remove any agent/tool metadata and return only clean Markdown starting with the first heading.

--- INPUT START ---
{raw_agent_markdown}
--- INPUT END ---
"""
        logger.info("⏳ Running QA agent to finalize the report...")
        qa_runner = InMemoryRunner(agent=qa_agent)
        qa_response = await qa_runner.run_debug(qa_query)
        final_clean_markdown = ""
        try:
            if hasattr(qa_response, "text") and qa_response.text:
                final_clean_markdown = qa_response.text
            elif hasattr(qa_response, "output") and qa_response.output:
                final_clean_markdown = qa_response.output
            else:
                final_clean_markdown = str(qa_response)
        except Exception:
            final_clean_markdown = str(qa_response)
        final_clean_markdown = final_clean_markdown.strip()

        # 8) Write report
        final_report_content = header_md + final_clean_markdown
        with open(output_filename, "w", encoding="utf-8") as f:
            f.write(final_report_content)
        logger.info(f"🎉 SUCCESS! Report written to {output_filename}")

        # Print preview
        preview = "\n".join(final_report_content.splitlines()[:40])
        print("\n--- A+ REPORT PREVIEW (First 40 lines) ---")
        print(preview)
        print("...")
        print("-----------------------------------------")

    except Exception as e:
        logger.error("Fatal error in pipeline: %s", e)
        traceback.print_exc()

# ---------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------
if __name__ == "__main__":
    try:
        asyncio.run(process_link())
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    except Exception as e:
        logger.error("Fatal synchronous error: %s", e)
        traceback.print_exc()
