"""Get ICMarkets MT5 credentials from client portal."""
import asyncio, sys, os
sys.path.insert(0, r'C:\Users\jose-\projects\trading_agent')
os.makedirs('logs', exist_ok=True)

EMAIL    = "joseg09.dg@gmail.com"
PASSWORD = "IMSMCbot*1"

async def main():
    from playwright.async_api import async_playwright

    async with async_playwright() as p:
        # Use headless=False so we can see what's happening
        browser = await p.chromium.launch(headless=False)
        ctx = await browser.new_context(
            viewport={"width": 1280, "height": 720},
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0"
        )
        page = await ctx.new_page()

        # Try ICMarkets client area login
        for login_url in [
            "https://secure.icmarkets.com/login",
            "https://client.icmarkets.com/login",
            "https://www.icmarkets.com/en/login/",
        ]:
            try:
                print(f"Trying: {login_url}")
                await page.goto(login_url, timeout=20000)
                await page.wait_for_timeout(2000)
                print(f"  URL: {page.url}")

                # Look for email/password fields
                email_f = await page.query_selector("input[type='email'], input[name='email'], input[name='username']")
                pwd_f   = await page.query_selector("input[type='password']")

                if email_f and pwd_f:
                    print(f"  Login form found!")
                    await email_f.fill(EMAIL)
                    await pwd_f.fill(PASSWORD)
                    await page.screenshot(path="logs/ic_login.png")

                    # Submit
                    submit = await page.query_selector("button[type='submit'], input[type='submit'], button:has-text('Login'), button:has-text('Sign in')")
                    if submit:
                        await submit.click()
                        await page.wait_for_timeout(5000)
                        print(f"  After submit: {page.url}")
                        await page.screenshot(path="logs/ic_after_login.png")

                        # Look for MT5 account info
                        content = await page.content()
                        if any(x in content.lower() for x in ["mt5", "metatrader", "account", "login"]):
                            print("  Account page detected!")
                            await page.screenshot(path="logs/ic_account.png")

                            # Try to find login number
                            import re
                            logins = re.findall(r'\b\d{7,8}\b', content)
                            if logins:
                                print(f"  Possible MT5 logins: {logins[:5]}")

                        break
            except Exception as e:
                print(f"  Error: {e}")
                continue

        await page.wait_for_timeout(5000)
        print("\nScreenshots saved in logs/")
        print("Check: ic_login.png, ic_after_login.png, ic_account.png")
        await browser.close()

asyncio.run(main())
