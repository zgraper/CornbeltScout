try:
    from ddgs import DDGS
    print("ddgs found")
except ImportError:
    print("ddgs NOT found")

try:
    from duckduckgo_search import DDGS
    print("duckduckgo_search found")
except ImportError:
    print("duckduckgo_search NOT found")
