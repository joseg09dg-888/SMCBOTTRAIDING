"""Create Pepperstone MT5 demo account via web automation."""
import asyncio, json, os, sys
sys.path.insert(0, r'C:\Users\jose-\projects\trading_agent')

DEMO_EMAIL    = "joseg09.dg@gmail.com"
DEMO_PASSWORD = "IMSMCbot2026!"
DEMO_NAME     = "Jose SMCBot"
DEMO_PHONE    = "+1234567890"

async def create_account():
    try:
        from playwright.async_api import async_playwright
    except ImportError:
        print("Installing playwright...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "playwright", "-q"])
        subprocess.check_call([sys.executable, "-m", "playwright", "install", "chromium", "--with-deps"])
        from playwright.async_api import async_playwright

    print("Opening Pepperstone demo account form...")
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        page = await browser.new_page()

        # Try Pepperstone demo form
        await page.goto("https://pepperstone.com/en/trade-with-us/accounts/demo/", timeout=30000)
        await page.wait_for_timeout(3000)
        print(f"Page: {page.url}")
        print(f"Title: {await page.title()}")

        # Try to find form fields
        forms = await page.query_selector_all("input[type=email], input[name*=email], input[placeholder*=Email]")
        print(f"Email fields found: {len(forms)}")

        if not forms:
            # Try alternative URL
            await page.goto("https://pepperstone.com/en/open-live-account/?type=demo", timeout=30000)
            await page.wait_for_timeout(3000)
            print(f"Alt page: {page.url}")
            forms = await page.query_selector_all("input[type=email]")
            print(f"Email fields: {len(forms)}")

        if forms:
            await forms[0].fill(DEMO_EMAIL)
            print(f"Filled email: {DEMO_EMAIL}")

            # Try to fill name
            name_fields = await page.query_selector_all("input[name*=name], input[placeholder*=Name]")
            if name_fields:
                await name_fields[0].fill(DEMO_NAME)
                print("Filled name")

            # Take screenshot
            await page.screenshot(path="logs/pepperstone_form.png")
            print("Screenshot saved: logs/pepperstone_form.png")

        # Try ICMarkets alternative
        print("\nTrying ICMarkets demo form...")
        await page.goto("https://icmarkets.com/global/en/open-trading-account/demo/", timeout=30000)
        await page.wait_for_timeout(3000)
        print(f"ICMarkets: {page.url}")
        await page.screenshot(path="logs/icmarkets_form.png")
        print("ICMarkets screenshot saved")

        # Wait a bit for user to see
        await page.wait_for_timeout(5000)
        await browser.close()

    print("\nManual option: The screenshots show the forms.")
    print("To create demo account quickly, go to:")
    print("  Pepperstone: https://pepperstone.com/en/trade-with-us/accounts/demo/")
    print("  ICMarkets:   https://icmarkets.com/global/en/open-trading-account/demo/")
    print("  Or in MT5: File -> Open Account -> search 'ICMarkets' or 'Pepperstone'")

asyncio.run(create_account())
