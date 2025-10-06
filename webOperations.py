import os
import requests
from dotenv import load_dotenv
from urllib.parse import quote_plus
from snapshot_operations import fetch_snapshot_data, check_snapshot_status

# -------------------------------------------------------------------------
# Load environment variables
# -------------------------------------------------------------------------
load_dotenv()

# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------
REDDIT_DISCOVERY_DATASET = "gd_lvz8ah06191smkebj4"
REDDIT_POSTS_DATASET = "gd_lvzdpsdlw09j6t702"
BRIGHTDATA_API_BASE = "https://api.brightdata.com"

# -------------------------------------------------------------------------
# 🧩 Helper: General API Request Wrapper
# -------------------------------------------------------------------------
def _send_api_request(url: str, **kwargs) -> dict | None:
    """Send a POST request to the Bright Data API with authentication headers."""
    api_key = os.getenv("BRIGHTDATA_API_KEY")
    if not api_key:
        raise EnvironmentError("Missing BRIGHTDATA_API_KEY in environment variables.")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(url, headers=headers, **kwargs)
        response.raise_for_status()
        return response.json()

    except requests.exceptions.RequestException as err:
        print(f"⚠️ Request failed: {err}")
        return None
    except Exception as err:
        print(f"❌ Unexpected error during API call: {err}")
        return None


# -------------------------------------------------------------------------
# 🌐 Search: Google / Bing SERP
# -------------------------------------------------------------------------
def serp_search(query: str, engine: str = "google") -> dict | None:
    """
    Perform a SERP (Search Engine Results Page) search using Bright Data proxy.

    Args:
        query (str): The search query.
        engine (str): The search engine ("google" or "bing").

    Returns:
        dict | None: Extracted search results with 'knowledge' and 'organic' sections.
    """
    base_urls = {
        "google": "https://www.google.com/search",
        "bing": "https://www.bing.com/search",
    }

    if engine not in base_urls:
        raise ValueError(f"Invalid search engine '{engine}'. Supported: google, bing")

    api_url = f"{BRIGHTDATA_API_BASE}/request"
    payload = {
        "zone": "ai_agent2",
        "url": f"{base_urls[engine]}?q={quote_plus(query)}&brd_json=1",
        "format": "raw",
    }

    response_data = _send_api_request(api_url, json=payload)
    if not response_data:
        return None

    return {
        "knowledge": response_data.get("knowledge", {}),
        "organic": response_data.get("organic", []),
    }


# -------------------------------------------------------------------------
# 🧠 Snapshot Utility: Trigger → Wait → Download
# -------------------------------------------------------------------------
def _execute_snapshot_pipeline(
    trigger_url: str,
    params: dict,
    payload: list,
    task_name: str = "snapshot",
) -> list | None:
    """Trigger a Bright Data snapshot and handle progress + download."""
    trigger_response = _send_api_request(trigger_url, params=params, json=payload)
    if not trigger_response:
        print(f"❌ Failed to trigger {task_name} snapshot.")
        return None

    snapshot_id = trigger_response.get("snapshot_id")
    if not snapshot_id:
        print(f"⚠️ No snapshot_id returned for {task_name}.")
        return None

    # Wait for snapshot completion
    if not check_snapshot_status(snapshot_id):
        print(f"❌ {task_name.capitalize()} snapshot did not complete successfully.")
        return None

    # Download completed snapshot
    return fetch_snapshot_data(snapshot_id)


# -------------------------------------------------------------------------
# 🔎 Reddit Search by Keyword
# -------------------------------------------------------------------------
def reddit_search_api(
    keyword: str,
    date: str = "All time",
    sort_by: str = "Hot",
    num_of_posts: int = 75,
) -> dict | None:
    """
    Search Reddit posts matching a given keyword using Bright Data dataset.

    Args:
        keyword (str): The keyword to search for.
        date (str): Time range (e.g., "Past month", "All time").
        sort_by (str): Sorting preference ("Hot", "New", etc.).
        num_of_posts (int): Number of posts to retrieve.

    Returns:
        dict | None: Parsed post data and total count.
    """
    trigger_url = f"{BRIGHTDATA_API_BASE}/datasets/v3/trigger"

    params = {
        "dataset_id": REDDIT_DISCOVERY_DATASET,
        "include_errors": "true",
        "type": "discover_new",
        "discover_by": "keyword",
    }

    payload = [
        {
            "keyword": keyword,
            "date": date,
            "sort_by": sort_by,
            "num_of_posts": num_of_posts,
        }
    ]

    snapshot_data = _execute_snapshot_pipeline(trigger_url, params, payload, "reddit search")
    if not snapshot_data:
        return None

    parsed_posts = [
        {"title": post.get("title"), "url": post.get("url")}
        for post in snapshot_data
        if isinstance(post, dict)
    ]

    return {
        "parsed_posts": parsed_posts,
        "total_found": len(parsed_posts),
    }


# -------------------------------------------------------------------------
# 💬 Reddit Comment Retrieval
# -------------------------------------------------------------------------
def reddit_post_retrieval(
    urls: list[str],
    days_back: int = 10,
    load_all_replies: bool = False,
    comment_limit: str = "",
) -> dict | None:
    """
    Retrieve comments from Reddit posts.

    Args:
        urls (list[str]): List of Reddit post URLs.
        days_back (int): Limit comments by date range.
        load_all_replies (bool): Whether to load all nested replies.
        comment_limit (str): Max comment count per post.

    Returns:
        dict | None: Extracted comments with total count.
    """
    if not urls:
        print("⚠️ No Reddit URLs provided.")
        return None

    trigger_url = f"{BRIGHTDATA_API_BASE}/datasets/v3/trigger"

    params = {
        "dataset_id": REDDIT_POSTS_DATASET,
        "include_errors": "true",
    }

    payload = [
        {
            "url": url,
            "days_back": days_back,
            "load_all_replies": load_all_replies,
            "comment_limit": comment_limit,
        }
        for url in urls
    ]

    snapshot_data = _execute_snapshot_pipeline(trigger_url, params, payload, "reddit comments")
    if not snapshot_data:
        return None

    parsed_comments = [
        {
            "comment_id": comment.get("comment_id"),
            "content": comment.get("comment"),
            "date": comment.get("date_posted"),
        }
        for comment in snapshot_data
        if isinstance(comment, dict)
    ]

    return {
        "comments": parsed_comments,
        "total_retrieved": len(parsed_comments),
    }