"""Login to ICMarkets and get MT5 demo account credentials."""
import asyncio, sys, time
sys.path.insert(0, r'C:\Users\jose-\projects\trading_agent')

EMAIL    = "joseg09.dg@gmail.com"
PASSWORD = "IMSMCbot*1"

async def main():
    from playwright.async_api import async_playwright

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False, slow_mo=500)
        page = await browser.new_page()

        # 1. Try to login to ICMarkets client portal
        print("Abriendo ICMarkets portal...")
        await page.goto("https://www.icmarkets.com/en/login/", timeout=30000)
        await page.wait_for_timeout(3000)
        print(f"URL: {page.url}")
        await page.screenshot(path="logs/ic_step1.png")

        # 2. Fill login form
        email_field = await page.query_selector("input[type=email], input[name*=email], input[placeholder*=Email]")
        if email_field:
            await email_field.fill(EMAIL)
            print(f"Email filled: {EMAIL}")

        pwd_field = await page.query_selector("input[type=password]")
        if pwd_field:
            await pwd_field.fill(PASSWORD)
            print("Password filled")

            # Submit
            submit = await page.query_selector("button[type=submit], input[type=submit]")
            if submit:
                await submit.click()
                await page.wait_for_timeout(5000)
                print(f"After login: {page.url}")
                await page.screenshot(path="logs/ic_step2.png")

        # 3. Look for MT5 account details
        content = await page.content()
        if "account" in content.lower() or "mt5" in content.lower():
            print("Account page detected")
            await page.screenshot(path="logs/ic_accounts.png")

        # 4. Try to open a new demo account if needed
        demo_link = await page.query_selector("a[href*=demo], button:has-text('Demo')")
        if demo_link:
            await demo_link.click()
            await page.wait_for_timeout(3000)
            await page.screenshot(path="logs/ic_demo.png")
            print("Demo account page opened")

        # 5. Try direct MT5 login via mt5.initialize
        print("\nTrying MT5 Python connection...")
        import MetaTrader5 as mt5
        time.sleep(2)
        ok = mt5.initialize()
        print(f"mt5.initialize(): {ok} | {mt5.last_error()}")
        if ok:
            info = mt5.account_info()
            print(f"CONNECTED! Login:{info.login} Balance:{info.balance} Server:{info.server}")
            mt5.shutdown()
        else:
            # Try with ICMarkets server names
            for srv in ["ICMarketsSC-Demo", "ICMarketsSC-Demo02", "ICMarketsEU-Demo",
                        "ICMarketsAU-Demo", "ICMarkets-Demo"]:
                mt5.shutdown()
                time.sleep(0.5)
                ok2 = mt5.initialize(server=srv)
                print(f"  {srv}: {ok2} | {mt5.last_error()}")
                if ok2:
                    info = mt5.account_info()
                    print(f"  CONNECTED! Login:{info.login} Balance:{info.balance}")
                    mt5.shutdown()
                    break

        print("\nScreenshots saved in logs/")
        print("ic_step1.png, ic_step2.png, ic_accounts.png")
        await page.wait_for_timeout(3000)
        await browser.close()

asyncio.run(main())
