"""Login to ICMarkets secure portal and get MT5 account number."""
from playwright.sync_api import sync_playwright
import re, time, os
os.makedirs('logs', exist_ok=True)

with sync_playwright() as p:
    browser = p.chromium.launch(headless=False)
    page = browser.new_page()

    print("Going to ICMarkets login...")
    page.goto("https://secure.icmarkets.com/Account/LogOn", timeout=30000)
    time.sleep(3)
    print(f"URL: {page.url}")

    # Fill form using evaluate to avoid selector issues
    try:
        page.evaluate("""() => {
            const inputs = document.querySelectorAll('input');
            for (const inp of inputs) {
                if (inp.type === 'email' || inp.name === 'Email') inp.value = 'joseg09.dg@gmail.com';
                if (inp.type === 'password' || inp.name === 'Password') inp.value = 'IMSMCbot*1';
            }
        }""")
        print("Form filled via evaluate")
    except Exception as e:
        print(f"evaluate error: {e}")

    page.screenshot(path="logs/ic_before.png")

    # Submit via JS
    try:
        page.evaluate("document.querySelector('form').submit()")
    except:
        page.keyboard.press("Enter")

    time.sleep(8)  # Wait for redirect
    print(f"After submit: {page.url}")
    page.screenshot(path="logs/ic_after.png")

    # Extract all data
    content = page.content()
    numbers = list(set(re.findall(r'\b\d{7,9}\b', content)))
    icservers = list(set(re.findall(r'ICMarkets\w+', content)))
    print(f"Numbers: {numbers[:15]}")
    print(f"ICMarkets refs: {icservers}")

    # Try accounts page
    for url in ["https://secure.icmarkets.com/TradingAccount",
                "https://secure.icmarkets.com/Account",
                "https://secure.icmarkets.com/"]:
        try:
            page.goto(url, timeout=15000)
            time.sleep(4)
            c = page.content()
            nums = list(set(re.findall(r'\b\d{7,9}\b', c)))
            srvs = list(set(re.findall(r'ICMarkets\w+', c)))
            print(f"\n{url}: nums={nums[:8]} servers={srvs}")
            page.screenshot(path=f"logs/ic_{url.split('/')[-1] or 'home'}.png")
            if nums:
                break
        except Exception as e:
            print(f"{url}: {e}")

    time.sleep(3)
    browser.close()
    print("\nDone")
