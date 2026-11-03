"""The real size of a product, looked up on the web by Muse Spark.

"Will the IKEA KIVIK 3-seat sofa fit?": a photo only gives an estimate of a
size, while the maker publishes the real one. Muse Spark searches the web
(the API's ``web_search`` tool, billed per search on top of the tokens) and
answers with the dimensions it found and the page it took them from, which
PIVOT quotes. Nothing is invented: without a source, there is no size.

The user may also give a link to the product's page (a shop, a Marketplace
ad). PIVOT reads what the page says about itself (its title, description and
product data, as a link preview does) and gives it to Muse Spark with the
link. Only public web addresses are read, once, a few kilobytes at most: a
page behind a login (as most Marketplace ads are) gives nothing, and the
search goes on with what the user said.
"""

from __future__ import annotations

import html
import ipaddress
import json
import re
import socket
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from urllib.parse import urljoin, urlparse

import httpx

from pivot.meta.client import MetaConfig, call_with_retry

# Web search grounding: $2.50 per 1,000 searches (dev.meta.ai pricing, Nov 1, 2026).
SEARCH_PRICE = 2.50 / 1000

LOOK_UP_PROMPT = (
    "Find the official dimensions of this product, from its maker or a shop selling it: {product}. "
    "{page}"
    "Answer with JSON only: {{\"name\": full product name, \"width_cm\": ..., \"depth_cm\": ..., "
    "\"height_cm\": ..., \"source\": address of the page the dimensions come from}}, the width "
    "being the longest horizontal side. If you find no dimensions on a page, answer "
    "{{\"name\": null}}.")
# Given with a link: what its page says about itself.
PAGE_PROMPT = ("The user gave this link to it: {url}. {said}If these words give its dimensions "
               "(an ad often does: W/B/L for width, D/T/P for depth, H for height), use them, "
               "with the link as the source; else open the page if you can, or search for the "
               "product it describes. ")
# Reading a page given by the user: at most this many bytes, redirects and seconds.
_PAGE_BYTES = 500_000
_REDIRECTS = 3
_TIMEOUT = 6.0
# Longest description of a page passed on, in characters.
_PAGE_CHARS = 1500
_USER_AGENT = "Mozilla/5.0 (compatible; PIVOT; +https://github.com/MarinaBeguin/PIVOT)"


@dataclass(frozen=True)
class ProductSize:
    """A product's size, as published.

    Attributes:
        name: Full product name.
        size: Width (long side), depth and height, in meters.
        source: Address of the page the dimensions come from.
    """

    name: str
    size: tuple[float, float, float]
    source: str


@dataclass(frozen=True)
class LookUp:
    """Result of :func:`look_up_size`, with what it cost.

    Attributes:
        product: The size found, or None.
        cost: Dollars spent (tokens and searches).
    """

    product: ProductSize | None
    cost: float


def parse_size(text: str) -> ProductSize | None:
    """Reads the model's answer.

    Args:
        text: The answer, JSON possibly wrapped in prose or a code block.

    Returns:
        The size, or None if the answer has no complete size with a source.
    """
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match is None:
        return None
    try:
        data = json.loads(match.group(0))
        size = tuple(float(data[key]) / 100 for key in ("width_cm", "depth_cm", "height_cm"))
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None
    source = str(data.get("source") or "")
    if not data.get("name") or not source.startswith("http") or min(size) <= 0:
        return None
    width, depth = max(size[0], size[1]), min(size[0], size[1])
    return ProductSize(str(data["name"]), (width, depth, size[2]), source)


def is_public(url: str) -> bool:
    """Tests whether an address may be read: http(s), on a host of the public internet.

    Args:
        url: The address.

    Returns:
        False for another scheme, or a host that resolves to a private, loopback,
        link-local or reserved address (the server's own network is never read).
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        return False
    try:
        addresses = {info[4][0] for info in socket.getaddrinfo(parsed.hostname, None)}
    except (socket.gaierror, UnicodeError):
        return False
    for address in addresses:
        ip = ipaddress.ip_address(str(address).split("%", 1)[0])
        if not ip.is_global or ip.is_multicast:
            return False
    return bool(addresses)


def describe_page(text: str) -> str:
    """Keeps what a page says about itself: title, description, product data.

    Args:
        text: The page's HTML.

    Returns:
        Those lines, plain text, at most a few hundred words; empty if none.
    """
    found: list[str] = []
    for key in ("og:title", "og:description", "twitter:title", "twitter:description",
                "description", "product:brand"):
        pattern = (r"<meta[^>]+(?:property|name)=[\"']" + re.escape(key)
                   + r"[\"'][^>]*content=[\"']([^\"']*)")
        found += re.findall(pattern, text, re.IGNORECASE)
    title = re.search(r"<title[^>]*>(.*?)</title>", text, re.IGNORECASE | re.DOTALL)
    if title:
        found.append(title.group(1))
    # Product data for search engines (schema.org JSON-LD): name, sizes.
    for block in re.findall(r"<script[^>]+application/ld\+json[^>]*>(.*?)</script>", text,
                            re.IGNORECASE | re.DOTALL):
        found += re.findall(r'"(?:name|description|width|depth|height)"\s*:\s*"([^"]{1,300})"',
                            block)
    lines: list[str] = []
    for line in (" ".join(html.unescape(item).split()) for item in found):
        if line and line not in lines:
            lines.append(line)
    return " | ".join(lines)[:_PAGE_CHARS]


def read_page(url: str) -> str:
    """Reads what a public page says about itself, as a link preview does.

    Args:
        url: The page, given by the user.

    Returns:
        Its title, description and product data; empty if it cannot be read
        (private address, login wall, timeout).
    """
    try:
        with httpx.Client(timeout=_TIMEOUT, follow_redirects=False, headers={
                "User-Agent": _USER_AGENT,
                "Accept-Language": "en, fr;q=0.8"}) as http:
            for _ in range(_REDIRECTS + 1):
                if not is_public(url):
                    return ""
                with http.stream("GET", url) as response:
                    if response.is_redirect and "location" in response.headers:
                        url = urljoin(url, response.headers["location"])
                        continue
                    if response.status_code != 200:
                        return ""
                    body = b""
                    for chunk in response.iter_bytes():
                        body += chunk
                        if len(body) >= _PAGE_BYTES:
                            break
                    return describe_page(body.decode(response.encoding or "utf-8",
                                                     errors="replace"))
    except httpx.HTTPError:
        return ""
    return ""


def look_up_size(config: MetaConfig, product: str, url: str = "", client: Any = None,
                 prices: tuple[float, float] = (1.25, 4.25),
                 reader: Callable[[str], str] = read_page) -> LookUp:
    """Asks Muse Spark, with web search, for a product's published size.

    Args:
        config: API configuration; ``agent_model`` is used.
        product: What to look up, e.g. ``"IKEA KIVIK 3-seat sofa"``; may be empty
            with a link.
        url: A link the user gave to the product's page, if any.
        client: SDK client; created from ``config`` if None.
        prices: Dollars per million input and output tokens.
        reader: Reads what a page says about itself (see :func:`read_page`).

    Returns:
        The size found (or None), and what the request cost.
    """
    client = client if client is not None else config.client()
    page = ""
    if url:
        said = reader(url)
        page = PAGE_PROMPT.format(url=url, said=f"Its page says: {said}. " if said else "")
    prompt = LOOK_UP_PROMPT.format(product=product or "the product of the link below",
                                   page=page)
    response = call_with_retry(lambda: client.responses.create(
        model=config.agent_model, input=prompt, tools=[{"type": "web_search"}]))
    usage = getattr(response, "usage", None)
    tokens = ((getattr(usage, "input_tokens", 0) or 0, getattr(usage, "output_tokens", 0) or 0)
              if usage is not None else (0, 0))
    searches = sum(1 for item in getattr(response, "output", None) or []
                   if getattr(item, "type", "") == "web_search_call")
    cost = (tokens[0] * prices[0] + tokens[1] * prices[1]) / 1e6 + searches * SEARCH_PRICE
    return LookUp(parse_size(getattr(response, "output_text", "") or ""), cost)
