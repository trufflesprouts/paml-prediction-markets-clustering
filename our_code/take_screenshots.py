"""Launch Streamlit headless and screenshot the 3 pages via Playwright.

Assumes:
  - Streamlit is running at http://127.0.0.1:8501
  - V2 artifacts exist in our_code/results/

Outputs:
  our_code/screenshots/page1_explore.png
  our_code/screenshots/page2_cluster.png
  our_code/screenshots/page3_classify.png
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

URL = "http://127.0.0.1:8501"
HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "screenshots"
OUT_DIR.mkdir(exist_ok=True)

VIEWPORT = {"width": 1280, "height": 900}


def wait_for_streamlit_idle(page, timeout_ms: int = 30_000) -> None:
    """Wait until Streamlit finishes rendering (spinner gone)."""
    page.wait_for_load_state("networkidle", timeout=timeout_ms)
    # also wait a beat for any residual rerun spinner
    try:
        page.wait_for_selector('[data-testid="stStatusWidget"]', state="hidden", timeout=3000)
    except Exception:
        pass
    time.sleep(1.0)


def click_sidebar_page(page, label: str) -> None:
    """Click the sidebar radio option by its visible label."""
    sidebar = page.locator('section[data-testid="stSidebar"]')
    sidebar.get_by_text(label, exact=True).first.click()
    wait_for_streamlit_idle(page)


def main() -> None:
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        context = browser.new_context(viewport=VIEWPORT)
        page = context.new_page()

        page.goto(URL, wait_until="networkidle", timeout=60_000)
        wait_for_streamlit_idle(page, timeout_ms=60_000)

        # Page 1: Explore Wallets (default page)
        # Scroll down so PCA shows; then take full_page screenshot.
        page.evaluate("window.scrollTo(0, 0)")
        time.sleep(0.5)
        # Give histograms + PCA time to render
        wait_for_streamlit_idle(page, timeout_ms=60_000)
        page.screenshot(path=str(OUT_DIR / "page1_explore.png"), full_page=True)
        print(f"[ss] saved page1_explore.png", flush=True)

        # Page 2: Cluster Wallets
        click_sidebar_page(page, "Cluster Wallets")
        wait_for_streamlit_idle(page, timeout_ms=60_000)
        # default mode is "Load V2 K=5 (default)"
        page.screenshot(path=str(OUT_DIR / "page2_cluster.png"), full_page=True)
        print(f"[ss] saved page2_cluster.png", flush=True)

        # Page 3: Classify a Wallet
        click_sidebar_page(page, "Classify a Wallet")
        wait_for_streamlit_idle(page, timeout_ms=60_000)
        # a wallet is auto-selected (first option in the dropdown), so the
        # classification output should already be rendered.
        page.screenshot(path=str(OUT_DIR / "page3_classify.png"), full_page=True)
        print(f"[ss] saved page3_classify.png", flush=True)

        context.close()
        browser.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ss] ERROR: {e}", file=sys.stderr)
        raise
