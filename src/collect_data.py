import feedparser
import requests
from bs4 import BeautifulSoup
import csv
import time

SPORT_FEEDS = [
    "https://www.espn.com/espn/rss/news",
    "https://feeds.bbci.co.uk/sport/rss.xml",
]

POLITICS_FEEDS = [
    "http://feeds.bbci.co.uk/news/politics/rss.xml",
    "https://rss.nytimes.com/services/xml/rss/nyt/Politics.xml",
]


def extract_article_text(url):
    try:
        r = requests.get(url, timeout=8, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(r.text, "html.parser")

        paragraphs = soup.find_all("p")
        text = " ".join(p.get_text() for p in paragraphs)

        return text.strip()

    except Exception as e:
        print("Failed:", url, e)
        return ""


def collect_from_feeds(feed_list, label, writer, limit_per_feed=150):
    for feed_url in feed_list:
        print(f"\nReading feed: {feed_url}")
        feed = feedparser.parse(feed_url)

        count = 0

        for entry in feed.entries:
            if count >= limit_per_feed:
                break

            if "link" not in entry:
                continue

            url = entry.link
            text = extract_article_text(url)

            if len(text) > 800:   # keep only meaningful articles
                writer.writerow([text, label])
                count += 1
                print("Saved:", label, count)

            time.sleep(0.5)


def main():
    with open("data/dataset.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "label"])

        collect_from_feeds(SPORT_FEEDS, "sports", writer)
        collect_from_feeds(POLITICS_FEEDS, "politics", writer)

    print("\n✅ Dataset collection complete")


if __name__ == "__main__":
    main()
