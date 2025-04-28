Why Cypress doesn’t “see” your existing cookies

Cypress always spins-up its own, temporary Chrome profile in /tmp (or %TEMP%) so every run starts with a clean slate.
Playwright, by contrast, can launch a real browser profile (--user-data-dir) by default, so whatever cookies your normal Chrome session already has are instantly available.

To make Cypress reuse those same profile cookies you have two practical options:

Option	When to use	Effort	Downsides
A. Inject the cookies programmatically (cy.session / cy.setCookie)	Works in any CI, keeps tests isolated	Low ↗	You still have to fetch / hard-code cookie values (like the GSSSO flow we built earlier).
B. Point Cypress at your real Chrome profile with --user-data-dir	You already have a valid SSO session in Chrome & just want to piggy-back on it	Medium	Breaks test isolation (state leaks)Won’t work in a clean CI runner unless you sync the profile folderProfile path is OS- and user-specific

Below is Option B—forcing Cypress to launch Chrome with your existing profile so all cookies, local-storage, etc. carry over exactly like Playwright.

⸻

1 · Locate your Chrome Beta profile folder

OS	Default user-data-dir	Profile name
Windows	C:\Users\<YOU>\AppData\Local\Google\Chrome Beta\User Data	usually Default, or Profile 1, …
macOS	~/Library/Application Support/Google/Chrome Beta	same
Linux	~/.config/google-chrome-beta	same

Open Chrome ➜ chrome://version to confirm the exact paths and profile name.

⸻

2 · Add a before:browser:launch hook in cypress.config.js

const { defineConfig } = require('cypress');
const path = require('path');

module.exports = defineConfig({
  e2e: {
    setupNodeEvents(on, config) {
      on('before:browser:launch', (browser = {}, launchOptions) => {
        if (browser.name === 'chrome' || browser.name === 'chromium') {
          //
          // 1. Path to your *existing* Chrome Beta profile
          //
          const userDir = process.platform === 'win32'
            ? 'C:\\Users\\<YOU>\\AppData\\Local\\Google\\Chrome Beta\\User Data'
            : process.platform === 'darwin'
              ? path.join(process.env.HOME, 'Library/Application Support/Google/Chrome Beta')
              : path.join(process.env.HOME, '.config/google-chrome-beta');

          // 2. (Optional) choose a profile folder inside that dir
          const profileName = 'Default';        // or 'Profile 1', etc.

          launchOptions.args.push(`--user-data-dir=${userDir}`);
          launchOptions.args.push(`--profile-directory=${profileName}`);

          // Optional: don’t let Cypress clear cookies between specs
          config.isTextTerminal && (config.trashAssetsBeforeRuns = false);
        }
        return launchOptions;
      });

      return config;
    },

    baseUrl: 'https://qa.ibdweb.site.gs.com',
    chromeWebSecurity: false
  }
});

Replace <YOU> and profileName with your actual values.

⸻

3 · Run Cypress with Chrome Beta and keep the window visible

# Windows (PowerShell)
npx cypress run --headed `
  --browser "C:\Program Files\Google\Chrome Beta\Application\chrome.exe"

# macOS / Linux
npx cypress run --headed \
  --browser "/Applications/Google Chrome Beta.app/Contents/MacOS/Google Chrome Beta"

	•	Because we overrode before:browser:launch, Cypress will reuse the specified profile and inherit every cookie and login token already present.
	•	Want to verify? Add cy.visit('/') and inspect DevTools—your auth cookies should already be there before any test code runs.

⸻

Common pitfalls

Symptom	Fix
Cypress still opens a blank temp profile	Double-check that the launchOptions.args.push(...) lines actually run (add console.log(launchOptions.args) in the hook).
“Another Chrome instance is running” error	Make sure the normal Chrome window is closed before launching Cypress; Chrome won’t allow two processes to lock the same profile simultaneously.
Works locally but fails in CI	CI machines don’t have your profile directory. Use Option A (programmatic cookie injection) for CI.
SSO expires mid-test	Even with the shared profile, Kerberos tokens might time-out. Consider refreshing them with cy.session() at the start of each spec.



⸻

TL;DR

Add a before:browser:launch hook that pushes
--user-data-dir=<path> and --profile-directory=<name> to Chrome’s arguments, then run Cypress with --headed --browser <Chrome Beta path>. Cypress will launch your real Chrome profile, carrying over the exact cookies Playwright sees.
