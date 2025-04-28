Below is a drop-in replacement for every file you need to replicate the token-based login flow you found in the older e2e repo.
	1.	cypress/plugins/index.ts — obtains fresh GSSSO + IBDLogin cookies before each run
	2.	cypress/support/data-utils.ts — Cypress custom command that sets those cookies and opens the app
	3.	cypress/support/index.ts — preserves the cookies between tests
	4.	cypress.env.json — where you store the Kerberos user / site URL once (no hard-coding in code)
	5.	Test usage — simply call cy.open_dealSetup() (or rename as you like) at the start of any spec

Everything stays in TypeScript; if you run JavaScript just drop the types.

⸻

1 ▸ cypress/plugins/index.ts

import { defineConfig } from 'cypress';
import { execSync }   from 'child_process';
import * as fs        from 'fs';
import * as path      from 'path';

function getAccessCookies() {
  //
  // 1. Hit the PingFederate / Kerberos endpoint exactly once
  // 2. Capture -k (self-signed) cookies into a temp jar
  // 3. Extract the GSSSO + IBDLogin chips and hand them back
  //
  const { QA_IDP_URL, CLIENT_ID, REDIRECT_URI } = process.env;

  const curlCmd = `curl -s -k -L --negotiate -u ":" \
    -b /tmp/cj.txt -c /tmp/cj.txt \
    "${QA_IDP_URL}?nonce=1234&scope=openid&response_type=id_token%20token&client_id=${CLIENT_ID}&access_token_manager_id=RefDefault&IdpAdapterId=Kerberos&response_mode=form_post&state=xyz&redirect_uri=${REDIRECT_URI}"`;

  execSync(curlCmd);

  const jar = fs.readFileSync('/tmp/cj.txt', 'utf8');
  const gssso     = /GSSSO\s+([^\s]+)/.exec(jar)?.[1];
  const ibdLogin  = /IBDLogin\s+([^\s]+)/.exec(jar)?.[1];

  if (!gssso || !ibdLogin)
    throw new Error('Kerberos handshake failed – cookies not found');

  return { gssso, ibdLogin };
}

export default defineConfig({
  e2e: {
    setupNodeEvents(on, config) {
      // Pull fresh cookies and inject into Cypress env
      const cookies = getAccessCookies();
      config.env.gssso    = cookies.gssso;
      config.env.ibdLogin = cookies.ibdLogin;
      return config;
    },
    baseUrl: 'https://qa.ibdweb.site.gs.com',
    chromeWebSecurity: false
  }
});



⸻

2 ▸ cypress/support/data-utils.ts

/// <reference types="cypress" />

Cypress.Commands.add('open_dealSetup', () => {
  const gssso    = Cypress.env('gssso');
  const ibdLogin = Cypress.env('ibdLogin');

  if (!gssso || !ibdLogin) {
    throw new Error('Missing gssso / ibdLogin cookies in Cypress.env');
  }

  cy.setCookie('GSSSO',    gssso);
  cy.setCookie('IBDLogin', ibdLogin);

  // IMPORTANT: call cy.visit *after* the cookies exist
  cy.visit('/ecm-syndicate-link/#/deal/dealSetup/');
});

Tip: Rename the command if your first route isn’t the deal-setup page.

⸻

3 ▸ cypress/support/index.ts

import './data-utils';

// keep auth cookies alive across every test
Cypress.Cookies.defaults({
  preserve: ['GSSSO', 'IBDLogin']
});



⸻

4 ▸ cypress.env.json

{
  "QA_IDP_URL"  : "https://idfs-qa.gs.com/as/authorization.oauth2",
  "CLIENT_ID"   : "f04c2914dfc04b29a6a08e1683295bba",
  "REDIRECT_URI": "https://qa.ibdweb.site.gs.com/tokenLogin?authLevel=30000"
}

If you can’t commit this file, set the 3 vars in your shell instead:

export QA_IDP_URL=...
export CLIENT_ID=...
export REDIRECT_URI=...



⸻

5 ▸ Updating your spec files

At the top of any test, swap the cy.visit(...) with one line:

it('Create Deal', () => {
  cy.open_dealSetup();      // <- cookie injection + visit
  // ...rest of your test exactly as before
});



⸻

Running

# interactive UI, Chrome Beta
npx cypress open  --browser "/Applications/Google Chrome Beta.app/Contents/MacOS/Google Chrome Beta"

# headless CI run, Chrome Beta
npx cypress run   --browser "/Applications/Google Chrome Beta.app/Contents/MacOS/Google Chrome Beta" --headless

Troubleshooting checklist

Symptom	Fix
Missing gssso / ibdLogin thrown	Make sure the Kerberos machine you’re running on can negotiate (VPN, kinit, etc.).
curl: (60) SSL certificate	Keep -k flag (ignore self-signed) or add CA cert to trust store.
Cookies set but still 401	Confirm cookies appear in DevTools Application → Cookies before the first XHR.

Once the cookies land, your Cypress tests will stay authenticated just like the Playwright originals.
