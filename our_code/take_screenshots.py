"""Launch Streamlit headless and screenshot the 3 pages via Playwright.

Takes full-page screenshots (Playwright's full_page=True), but first scrolls
through the whole page to trigger lazy-rendered plot images, then waits for
every <img> in the document to be fully loaded, then scrolls back to top.

Outputs:
  our_code/screenshots/page1_explore.png
  our_code/screenshots/page2_cluster.png
  our_code/screenshots/page3_classify.png
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright, Page

URL = "http://127.0.0.1:8501"
HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "screenshots"
OUT_DIR.mkdir(exist_ok=True)

VIEWPORT = {"width": 1280, "height": 900}


def wait_for_all_images_loaded(page: Page, timeout_ms: int = 30_000) -> None:
    """Block until every <img> has naturalHeight > 0 (or timeout)."""
    page.wait_for_function(
        """() => {
            const imgs = Array.from(document.images);
            if (imgs.length === 0) return true;
            return imgs.every(i => i.complete && i.naturalHeight > 0);
        }""",
        timeout=timeout_ms,
    )


def wait_for_streamlit_idle(page: Page, timeout_ms: int = 60_000) -> None:
    """Wait for Streamlit to finish rendering: networkidle + no spinner."""
    page.wait_for_load_state("networkidle", timeout=timeout_ms)
    try:
        page.wait_for_selector(
            '[data-testid="stStatusWidget"]', state="hidden", timeout=3_000
        )
    except Exception:
        pass
    time.sleep(0.5)


def slow_scroll_whole_page(page: Page, step_px: int = 600, pause_s: float = 0.3) -> None:
    """Scroll from top to bottom in chunks to trigger lazy-rendered plots."""
    page.evaluate("window.scrollTo(0, 0)")
    time.sleep(0.2)
    height = page.evaluate("document.body.scrollHeight")
    y = 0
    while y < height:
        y += step_px
        page.evaluate(f"window.scrollTo(0, {y})")
        time.sleep(pause_s)
        # recompute height — rendering may have grown the page
        new_height = page.evaluate("document.body.scrollHeight")
        if new_height > height:
            height = new_height
    # ensure we really hit the bottom
    page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
    time.sleep(pause_s)


def prepare_and_shot(page: Page, out_path: Path) -> None:
    """Streamlit nests the real scroll inside a container, so `full_page=True`
    often clips at viewport height.  Workaround: measure the rendered content
    height (from the Streamlit block container), resize the viewport to match,
    then take a regular screenshot."""
    wait_for_streamlit_idle(page)
    slow_scroll_whole_page(page)
    wait_for_streamlit_idle(page)
    wait_for_all_images_loaded(page)

    # Find the actual content height.  Try multiple candidates: Streamlit's
    # main block container, the body, the documentElement.
    content_height = int(page.evaluate(
        """() => {
            const candidates = [
                document.querySelector('[data-testid="stMain"]'),
                document.querySelector('[data-testid="stAppViewContainer"]'),
                document.querySelector('section.main'),
                document.body,
                document.documentElement,
            ];
            let h = 0;
            for (const el of candidates) {
                if (!el) continue;
                h = Math.max(h, el.scrollHeight, el.offsetHeight);
            }
            return h;
        }"""
    ))
    desired_h = min(max(content_height + 60, 900), 8000)
    page.set_viewport_size({"width": VIEWPORT["width"], "height": desired_h})
    # give Streamlit a beat to re-lay out at the taller viewport
    time.sleep(0.8)
    wait_for_streamlit_idle(page)
    wait_for_all_images_loaded(page)
    page.evaluate("window.scrollTo(0, 0)")
    time.sleep(0.3)
    page.screenshot(path=str(out_path), full_page=True)
    # reset viewport for the next page
    page.set_viewport_size(VIEWPORT)
    time.sleep(0.3)


def click_sidebar_page(page: Page, label: str) -> None:
    sidebar = page.locator('section[data-testid="stSidebar"]')
    sidebar.get_by_text(label, exact=True).first.click()
    wait_for_streamlit_idle(page)


def main() -> None:
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        context = browser.new_context(
            viewport=VIEWPORT,
            device_scale_factor=1,
        )
        page = context.new_page()

        page.goto(URL, wait_until="networkidle", timeout=60_000)
        wait_for_streamlit_idle(page)

        # Page 1: Explore Wallets (default page)
        prepare_and_shot(page, OUT_DIR / "page1_explore.png")
        print(f"[ss] saved {OUT_DIR / 'page1_explore.png'}", flush=True)

        # Page 2: Cluster Wallets
        click_sidebar_page(page, "Cluster Wallets")
        prepare_and_shot(page, OUT_DIR / "page2_cluster.png")
        print(f"[ss] saved {OUT_DIR / 'page2_cluster.png'}", flush=True)

        # Page 3: Classify a Wallet
        click_sidebar_page(page, "Classify a Wallet")
        # default selectbox auto-picks first option; streamlit reruns and renders
        # the comparison chart.  Still, give it an extra moment.
        wait_for_streamlit_idle(page)
        time.sleep(1.5)
        prepare_and_shot(page, OUT_DIR / "page3_classify.png")
        print(f"[ss] saved {OUT_DIR / 'page3_classify.png'}", flush=True)

        context.close()
        browser.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ss] ERROR: {e}", file=sys.stderr)
        raise
